# trzju.github.io (Hexo)

## 本地环境
- Node.js 20.x（与 GitHub Actions 一致，见 `.nvmrc`）
- Git

## 安装依赖
```bash
npm install
```

如果你本机 `npm` 生命周期脚本报错，再使用：

```bash
npm install --ignore-scripts
```

## 本地预览
```bash
npm run clean
npm run build
npm run server
```

默认访问 `http://localhost:4000`。

## 部署到 GitHub Pages（gh-pages 分支）
1. 先确保当前仓库远程地址可推送（你有权限）。
2. 执行：

```bash
npm run clean
npm run build
npm run deploy
```

部署配置在 `_config.yml`：
- `deploy.type: git`
- `deploy.repo: https://github.com/trzju/trzju.github.io.git`
- `deploy.branch: gh-pages`

如果你继续使用仓库内的 GitHub Actions 工作流（`.github/workflows/pages.yml`），也可以只提交源码到 `main`，由 Actions 自动构建并发布。
