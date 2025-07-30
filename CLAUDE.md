（交流可以用英文，本文档中文，保留这句）

# 项目说明

## 项目目标
写一个 markdown 到 html 的转换脚本。
markdown 有多个，典型组织为 index.md + chapter1.md + ...
转出的 html 需要统一风格跳转正常（需开 headless mode 验证）
html 风格现代，mobile friendly

## 技术方案

### 核心功能
1. **Markdown 解析**: 使用 Python markdown 库，支持扩展功能（如表格、代码高亮等），特别要支持好 latex（并能容忍一些小错，如 opening $$ 但 closing 是$。以及 latex 里的 _ 没有转义）
2. **批量转换**: 递归处理目录中所有 .md 文件，保持原目录结构
3. **链接处理**: 自动将 .md 链接转换为 .html 链接
4. **导航生成**: 自动生成侧边栏导航和上一页/下一页按钮

### 样式设计
- **框架**: 自定义轻量级 CSS，无外部依赖
- **响应式**: Mobile-first 设计，支持各种屏幕尺寸
- **主题**: 默认亮色主题，预留深色模式支持
- **布局**: 左侧固定导航栏（桌面端），顶部汉堡菜单（移动端）

### 项目结构
```
md_to_html/
├── convert.py          # 主转换脚本
├── templates/          # HTML 模板
│   └── base.html      # 基础模板
├── static/            # 静态资源
│   ├── style.css      # 主样式文件
│   └── script.js      # 导航交互脚本
├── cache/             # 缓存目录（.gitignore）
├── output/            # 输出目录（.gitignore）
└── test_data/         # 测试 markdown 文件
```

### 使用方式
```bash
# 基础转换
python convert.py test_data/ output/

# 清除缓存后转换
python convert.py test_data/ output/ --clear-cache

# 指定单个文件
python convert.py test_data/index.md output/

```

### 缓存机制
- 基于文件内容的 MD5 哈希缓存
- 缓存 HTML 片段而非完整页面
- 支持手动清除缓存选项

### 测试验证
- 使用 Playwright 进行 headless 浏览器测试
- 验证所有内部链接可访问
- 检查响应式布局在不同视口下的表现
- 确保导航功能正常工作

