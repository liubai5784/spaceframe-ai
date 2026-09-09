#!/usr/bin/env bash
# ============================================================
# 空间刚架智能计算 Agent —— 腾讯云 Ubuntu 一键部署脚本
# 适用：Ubuntu 20.04 / 22.04（4核4GB 足够）
# 用法：
#   sudo bash deploy.sh
# 可选环境变量：
#   PORT=8000          监听端口（默认 8000）
#   LLM_API_KEY=xxx    大模型 API Key（不填则使用规则解析模式）
# ============================================================
set -euo pipefail

REPO_URL="https://github.com/liubai5784/spaceframe-ai.git"
APP_DIR="/opt/spaceframe-ai"
PORT="${PORT:-8000}"
LLM_KEY="${LLM_API_KEY:-}"

echo "==> [1/5] 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git curl unzip > /dev/null

echo "==> [2/5] 拉取项目代码（已存在则跳过）"
if [ -d "$APP_DIR/src" ]; then
  echo "检测到已有代码，跳过拉取（如需更新请手动 git pull）"
elif [ -d "$APP_DIR/.git" ]; then
  cd "$APP_DIR" && git pull --ff-only
else
  # 国内服务器访问 GitHub 不稳定：加大缓冲 + 重试 3 次，仍失败则走镜像
  git config --global http.postBuffer 524288000
  fetch_ok=0
  for i in 1 2 3; do
    echo "尝试 git clone（第 $i 次）..."
    if git clone --depth 1 "$REPO_URL" "$APP_DIR" 2>/tmp/git_err.log; then
      fetch_ok=1; break
    fi
    sleep 2
  done
  if [ "$fetch_ok" != "1" ]; then
    echo "git 直连失败，改用镜像下载 zip ..."
    mkdir -p "$APP_DIR"
    curl -fsSL --connect-timeout 20 -o /tmp/sf.zip \
      "https://gh-proxy.com/https://github.com/liubai5784/spaceframe-ai/archive/refs/heads/main.zip" \
      || curl -fsSL --connect-timeout 20 -o /tmp/sf.zip \
      "https://codeload.github.com/liubai5784/spaceframe-ai/zip/refs/heads/main" \
      || { echo "❌ GitHub 直连与镜像均失败，请手动下载后重试"; exit 1; }
    rm -rf "$APP_DIR"
    unzip -q /tmp/sf.zip -d /tmp/sf_extract
    mv /tmp/sf_extract/spaceframe-ai-main "$APP_DIR"
  fi
fi

echo "==> [3/5] 创建 Python 环境并安装依赖"
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "==> [4/5] 注册 systemd 服务（开机自启 + 崩溃自动重启）"
cat > /etc/systemd/system/spaceframe-ai.service <<EOF
[Unit]
Description=Space Frame AI Agent Web Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=3
Environment=LLM_API_KEY=$LLM_KEY

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable spaceframe-ai > /dev/null 2>&1
systemctl restart spaceframe-ai

echo "==> [5/5] 检查服务状态"
sleep 2
if systemctl is-active --quiet spaceframe-ai; then
  PUBLIC_IP=$(curl -s -m 3 https://api.ipify.org || hostname -I | awk '{print $1}')
  echo ""
  echo "✅ 部署成功！网页端地址：http://${PUBLIC_IP}:${PORT}"
  echo ""
  echo "⚠️  重要：请在腾讯云控制台 -> 轻量应用服务器/云服务器 -> 防火墙/安全组"
  echo "    放行 TCP ${PORT} 端口，否则外网无法访问。"
  echo ""
  echo "常用命令："
  echo "  systemctl status spaceframe-ai    # 查看状态"
  echo "  journalctl -u spaceframe-ai -f    # 实时日志"
  echo "  systemctl restart spaceframe-ai   # 重启"
else
  echo "❌ 服务启动失败，查看日志：journalctl -u spaceframe-ai -n 50"
  exit 1
fi
