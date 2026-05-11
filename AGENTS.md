## 图片压缩

- PNG → progressive JPEG q85，使用 `png2jpg.py`
- 用法：`python png2jpg.py cartoon-01.png cartoon-02.png cartoon-03.png`
- 原则：漫画/线稿风格用 JPEG q85 足够，progressive=True 实现渐进加载

## 本地预览

- 使用 `npx wrangler dev` 模拟 Cloudflare 生产环境
- timeout 3600s 给用户充裕的审阅时间，确认后再部署
- 用法：`npx wrangler dev`（见 wrangler.toml）
