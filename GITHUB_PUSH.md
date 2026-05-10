# 推送到 GitHub（首次）

本目录已用 `git init` 并做好 `.gitignore`。你需要在 GitHub 上**新建空仓库**（不要勾选添加 README），然后执行：

```bash
cd Lab2
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

若使用 SSH：

```bash
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git push -u origin main
```

登录可用浏览器 OAuth、Personal Access Token，或配置 SSH 公钥。课程要求报告里附上 **Git 仓库链接** 即可。
