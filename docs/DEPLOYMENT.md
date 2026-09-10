# 部署记录（生产环境）

> 本文档记录 2026-09-10 实际部署结果，供答辩演示与后续维护参考。
> 文中不包含任何密钥，密钥一律通过 systemd drop-in 注入。

## 环境

| 项 | 值 |
|---|---|
| 服务器 | 腾讯云轻量应用服务器，Ubuntu，4核4GB/40GB |
| 系统用户 | lighthouse |
| 项目目录 | `/opt/spaceframe-ai` |
| 后端服务 | `spaceframe-ai.service`（uvicorn，端口 8000） |
| 隧道服务 | `cloudflared-spaceframe.service` |
| 隧道名 / ID | `spaceframe-ai` / `2d516356-82f0-4333-98cf-9bf0b3435765` |
| 公网地址 | **https://spaceframe.escfpig.cn** |
| 大模型 | DeepSeek `deepseek-chat`（OpenAI 兼容接口） |

## 部署架构

```
用户浏览器 ──HTTPS──> Cloudflare 边缘（域名 spaceframe.escfpig.cn）
                          │ Cloudflare Tunnel（出站 QUIC/HTTP2 连接，无需开放入站端口）
                          ▼
                   cloudflared（腾讯云服务器，systemd 常驻）
                          │ http://localhost:8000
                          ▼
                   FastAPI (app.main) ──> 空间刚架有限元内核（数值计算）
                          └──────────────> DeepSeek API（自然语言解析/结果解释）
```

## 部署步骤摘要

1. **拉取代码**：`deploy.sh`（国内网络已加强：git 重试 + 缓冲 + 镜像 zip 兜底）
   - 代码目录 `/opt/spaceframe-ai`，Python venv：`.venv`
   - systemd：`spaceframe-ai.service`（开机自启 + 崩溃重启）
2. **Cloudflare Tunnel**：
   - `cloudflared tunnel login`（浏览器授权，生成 `~/.cloudflared/cert.pem`）
   - `cloudflared tunnel create spaceframe-ai`
   - `cloudflared tunnel route dns spaceframe-ai spaceframe.escfpig.cn`
   - 配置文件：`~/.cloudflared/config.yml`（ingress → `http://localhost:8000`，兜底 404）
3. **LLM 配置**（Drop-in 方式，不动主配置）：
   - `/etc/systemd/system/spaceframe-ai.service.d/llm.conf`：
     ```
     [Service]
     Environment=LLM_API_KEY=<你的Key>
     Environment=LLM_BASE_URL=https://api.deepseek.com/v1
     Environment=LLM_MODEL=deepseek-chat
     ```

## 常用命令

```bash
systemctl status spaceframe-ai          # 后端状态
systemctl status cloudflared-spaceframe # 隧道状态
journalctl -u spaceframe-ai -f          # 后端日志
journalctl -u cloudflared-spaceframe -f # 隧道日志
curl -s https://spaceframe.escfpig.cn/api/health   # 健康检查（llm:true 表示 LLM 已配置）
```

## 排障经验（踩过的坑）

| 问题 | 原因 | 解决 |
|---|---|---|
| `git clone` GnuTLS -110 | 国内访问 GitHub TLS 中断 | 重试 + 加大 postBuffer + gh-proxy 镜像 zip 兜底 |
| cloudflared 秒退 (exit 0) | `--config` 是 `tunnel` 命令层参数，不能放 `run` 后；且默认值即正确路径 | `cloudflared tunnel run spaceframe-ai`（不带 --config） |
| `flag provided but not defined: -api-token` | 新版 cloudflared 移除 `--api-token` 用法 | 改用 `cloudflared tunnel login` 浏览器授权 |
| ICMP proxy 警告 | GID 不在 ping_group_range | 无害，忽略 |
| UDP/QUIC 部分失败 | 云厂商出站 UDP 受限 | 自动降级 HTTP/2，不影响 |
| 错误码 1033 | 隧道无活跃连接（cloudflared 未跑起来） | 修复 ExecStart 后恢复 |

## 验证命令（答辩前自检）

```bash
curl -s https://spaceframe.escfpig.cn/api/health
curl -s -X POST https://spaceframe.escfpig.cn/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"8x6x4m两层刚架 柱HW300 梁HN400 顶部荷载20kN/m2"}'
```
