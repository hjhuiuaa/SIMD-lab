# 推送到 GitHub

本目录为 Git 仓库根目录，已配置 `.gitignore`（不含大体积数据与课件 PDF）。

远程仓库：**https://github.com/hjhuiuaa/SIMD-lab**

首次推送（在 `Lab2` 目录下）：

```bash
git remote add origin https://github.com/hjhuiuaa/SIMD-lab.git
git branch -M main
git push -u origin main
```

若已添加过 `origin`，只需：

```bash
git push -u origin main
```

使用 SSH 时将 URL 换成 `git@github.com:hjhuiuaa/SIMD-lab.git`。登录可用浏览器、Personal Access Token 或 SSH 公钥。课程报告里附上仓库链接即可。
