# Cloudflare + 自有域名部署指南

本项目的网页端由「静态前端（web/）」和「Python 计算后端（FastAPI）」组成。
Cloudflare Pages 只能托管静态文件，因此需要把计算请求转发到运行后端的服务器。

两种推荐路线：

- **路线 A：Cloudflare Tunnel（最简单，推荐）**——后端整体通过隧道挂到你的域名下，无需开放服务器端口、自动 HTTPS。
- **路线 B：Cloudflare Pages + Worker 代理**——前端由 Pages 自动构建（连 GitHub 仓库），API 请求由 Worker 转发到后端服务器。

---

## 路线 A：Cloudflare Tunnel（推荐）

### 前提
1. 域名已托管到 Cloudflare（在 Cloudflare 添加站点并把域名的 NS 记录改为 Cloudflare 提供的两个 NS）。
2. 腾讯云服务器已运行本项目后端（见 `deploy.sh`，或手动 `uvicorn app.main:app --host 127.0.0.1 --port 8000`）。

### 步骤（在腾讯云服务器上执行）

```bash
# 1. 安装 cloudflared
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 2. 登录 Cloudflare 账号（会弹出授权链接，浏览器打开并授权域名）
cloudflared tunnel login

# 3. 创建隧道（名字自定，例如 spaceframe）
cloudflared tunnel create spaceframe

# 4. 配置隧道：把域名指向本地 8000 端口
cloudflared tunnel route dns spaceframe <你的子域名>. <你的域名>
# 例如：cloudflared tunnel route dns spaceframe spaceframe.example.com

# 5. 运行隧道
cloudflared tunnel run spaceframe
```

更规范的用法是写成 systemd 服务（`/etc/systemd/system/cloudflared.service`），
或用 `cloudflared tunnel --url http://localhost:8000` 的快速模式（无需登录建隧道，
适合临时演示，但域名需用 `--hostname` 指定并配置 CNAME）。

### 完成
浏览器打开 `https://<你的域名>` 即可使用，全程 HTTPS，无需在腾讯云安全组开放 8000 端口。

> 注意：Cloudflare 免费版国内访问走海外节点，速度可能一般；答辩演示建议提前测试。

---

## 路线 B：Cloudflare Pages + Worker 代理

### 1. 前端发布到 Pages
1. Cloudflare 控制台 → Workers 和 Pages → 创建 → Pages → 连接 Git 仓库。
2. 选择 `liubai5784/spaceframe-ai`，构建配置：
   - 构建命令：`echo "static site"`（本项目前端无构建步骤）
   - 输出目录：`web`
3. 部署完成后得到一个 `xxx.pages.dev` 地址。
4. 在「自定义域」里绑定你的域名（需域名已在 Cloudflare）。

### 2. Worker 代理 API 请求
前端页面里的请求是相对路径 `/api/*`，需要在 Pages 的同域下把它们转发到后端服务器。
用 Pages Functions 或在 Pages 项目「设置 → 函数」中放一个 `functions/api/[[path]].js`：

```js
// functions/api/[[path]].js
export async function onRequest(context) {
  const url = new URL(context.request.url);
  const backend = 'http://<腾讯云公网IP>:8000';   // 后端地址
  const target = backend + url.pathname + url.search;
  return fetch(target, {
    method: context.request.method,
    headers: context.request.headers,
    body: context.request.method === 'GET' ? undefined : await context.request.arrayBuffer(),
  });
}
```

（把 `functions/` 目录放到仓库根目录并推送，Pages 会自动使用 Functions。）

### 3. 后端服务器
在腾讯云运行 `deploy.sh`（或手动启动 uvicorn），并**在安全组放行 TCP 8000 端口**
（Worker 服务器需要能访问到后端）。后端已配置 CORS 放行 `*`，跨域调用不会被拦截。

---

## 常见问题

- **访问很慢 / 打不开**：先确认后端 `curl http://<IP>:8000/api/health` 返回
  `{"status":"ok",...}`；Tunnel 路线确认 `cloudflared tunnel list` 状态为 healthy。
- **页面能开但点分析没反应**：浏览器 F12 → Network，看 `/api/analyze` 请求是否报
  CORS 或 502；Worker 路线确认后端端口已放行。
- **想换端口**：后端默认 8000，改 `deploy.sh` 里的 `PORT` 环境变量，或直接改 uvicorn 命令。
