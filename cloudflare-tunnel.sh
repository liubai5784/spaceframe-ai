#!/usr/bin/env bash
# ============================================================
# Cloudflare Tunnel 一键部署
# 把本地 8000 端口的空间刚架 Agent 挂到你的域名下（HTTPS）
# 前提：
#   1. 域名已托管到 Cloudflare
#   2. 后端已在本地运行（uvicorn app.main:app --port 8000，见 deploy.sh）
#   3. 本机可访问外网（隧道是出站连接，无需公网 IP / 无需开放端口）
#
# 用法：
#   方式1（浏览器授权）：
#       bash cloudflare-tunnel.sh <你的域名> <子域名>
#       例：bash cloudflare-tunnel.sh example.com spaceframe
#       -> 部署后访问 https://spaceframe.example.com
#
#   方式2（API Token，无需浏览器）：
#       CLOUDFLARE_API_TOKEN=xxx bash cloudflare-tunnel.sh <你的域名> <子域名>
#       （Token 在 Cloudflare 控制台创建：My Profile -> API Tokens ->
#         创建 Custom Token，权限选 Zone:DNS:Edit + Zone:Zone:Read，
#         区域选你的域名）
# ============================================================
set -euo pipefail

DOMAIN="${1:-}"
SUB="${2:-spaceframe}"
PORT="${PORT:-8000}"
TUNNEL_NAME="spaceframe-ai"
HOSTNAME="${SUB}.${DOMAIN}"
TOKEN="${CLOUDFLARE_API_TOKEN:-}"

if [ -z "$DOMAIN" ]; then
  echo "用法: bash cloudflare-tunnel.sh <你的域名> [子域名]"
  echo "例:   bash cloudflare-tunnel.sh example.com spaceframe"
  exit 1
fi

echo "==> [1/5] 安装 cloudflared"
if ! command -v cloudflared > /dev/null; then
  curl -sL --output /tmp/cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb
fi
cloudflared --version

echo "==> [2/5] 认证 Cloudflare 账号"
if [ -n "$TOKEN" ]; then
  echo "使用 API Token 模式"
  export TUNNEL_API_TOKEN="$TOKEN"
else
  echo "请在弹出的浏览器中完成授权（若无法弹出，复制终端里的链接到浏览器打开）"
  cloudflared tunnel login
fi

echo "==> [3/5] 创建隧道（已存在则复用）"
if [ -n "$TOKEN" ]; then
  cloudflared tunnel --api-token "$TOKEN" create "$TUNNEL_NAME" 2>/dev/null \
    || echo "隧道已存在，继续"
else
  cloudflared tunnel create "$TUNNEL_NAME" 2>/dev/null || echo "隧道已存在，继续"
fi

echo "==> [4/5] 配置 DNS 路由: $HOSTNAME -> 隧道"
if [ -n "$TOKEN" ]; then
  cloudflared tunnel --api-token "$TOKEN" route dns "$TUNNEL_NAME" "$HOSTNAME" 2>/dev/null \
    || echo "DNS 记录已存在，继续"
else
  cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" 2>/dev/null \
    || echo "DNS 记录已存在，继续"
fi

echo "==> [5/5] 注册 systemd 服务（开机自启 + 崩溃自动重启）"
TUNNEL_ID=$(cloudflared tunnel list --name "$TUNNEL_NAME" 2>/dev/null \
            | awk -v n="$TUNNEL_NAME" '$0 ~ n {print $1}' | head -1)
if [ -z "$TUNNEL_ID" ]; then
  echo "❌ 找不到隧道 ID，请检查第 3 步输出"
  exit 1
fi

# 生成运行配置
mkdir -p /etc/cloudflared
cat > /etc/cloudflared/config.yml <<EOF
tunnel: $TUNNEL_ID
credentials-file: /root/.cloudflared/$TUNNEL_ID.json
ingress:
  - hostname: $HOSTNAME
    service: http://localhost:$PORT
  - service: http_status:404
EOF

cat > /etc/systemd/system/cloudflared-spaceframe.service <<EOF
[Unit]
Description=Cloudflare Tunnel for Space Frame AI
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cloudflared tunnel run --config /etc/cloudflared/config.yml $TUNNEL_NAME
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cloudflared-spaceframe > /dev/null 2>&1
systemctl restart cloudflared-spaceframe

echo ""
echo "✅ 隧道已启动，网页端地址：https://$HOSTNAME"
echo ""
echo "验证："
echo "  systemctl status cloudflared-spaceframe   # 隧道状态"
echo "  journalctl -u cloudflared-spaceframe -f   # 隧道日志"
echo "  curl https://$HOSTNAME/api/health         # 应返回 {\"status\":\"ok\",...}"
echo ""
echo "⚠️  首次生效可能需要 1-2 分钟（DNS + 证书下发）。"
echo "    Cloudflare 免费版国内访问走海外节点，速度一般，答辩演示请提前实测。"
