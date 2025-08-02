#!/usr/bin/env python3
"""
Markdown to HTML converter with LaTeX support and navigation generation.
"""

import os
import sys
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil

import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


class LaTeXPreprocessor(Preprocessor):
    """Preprocessor to handle LaTeX math with tolerance for common errors."""
    
    def run(self, lines):
        text = '\n'.join(lines)
        
        # Fix common LaTeX errors
        # 1. Handle mismatched $$ and $
        text = self._fix_display_math(text)
        
        # 2. Escape underscores in LaTeX expressions
        text = self._escape_underscores_in_latex(text)
        
        # 3. Fix tables that need blank lines
        text = self._fix_tables(text)
        
        # 4. Fix code blocks in lists
        text = self._fix_code_blocks_in_lists(text)
        
        return text.split('\n')
    
    def _fix_display_math(self, text):
        """Fix mismatched display math delimiters."""
        # Pattern to find display math blocks that start with $$ but end with single $
        pattern = r'\$\$([^$]+?)\$(?!\$)'
        text = re.sub(pattern, r'$$\1$$', text)
        return text
    
    def _escape_underscores_in_latex(self, text):
        """Escape underscores within LaTeX expressions."""
        # Handle display math $$...$$
        def escape_in_display(match):
            content = match.group(1)
            content = content.replace('_', '\\_')
            return f'$${content}$$'
        
        # Handle inline math $...$
        def escape_in_inline(match):
            content = match.group(1)
            # Avoid matching display math that we've already processed
            if content.startswith('$'):
                return match.group(0)
            content = content.replace('_', '\\_')
            return f'${content}$'
        
        # Process display math first
        text = re.sub(r'\$\$(.*?)\$\$', escape_in_display, text, flags=re.DOTALL)
        # Then process inline math (avoiding display math)
        text = re.sub(r'(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)', escape_in_inline, text)
        
        return text
    
    def _fix_tables(self, text):
        """Ensure tables have blank lines before and after them."""
        lines = text.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line looks like a table separator
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                # Look back to find the header
                if i > 0 and '|' in lines[i-1]:
                    # Check if there's already a blank line before the table
                    if i > 1 and new_lines and new_lines[-1].strip():
                        new_lines.append('')  # Add blank line before table
                    
                    # Add the header
                    if new_lines and new_lines[-1] == lines[i-1]:
                        pass  # Already added
                    else:
                        new_lines.append(lines[i-1])
                    
                    # Add the separator
                    new_lines.append(line)
                    
                    # Add table rows
                    j = i + 1
                    while j < len(lines) and '|' in lines[j] and lines[j].strip():
                        new_lines.append(lines[j])
                        j += 1
                    
                    # Add blank line after table if needed
                    if j < len(lines) and lines[j].strip():
                        new_lines.append('')
                    
                    i = j - 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            
            i += 1
        
        return '\n'.join(new_lines)
    
    def _fix_code_blocks_in_lists(self, text):
        """Fix code blocks that are inside lists by ensuring proper indentation."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        processed = False
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts with spaces followed by ```
            if re.match(r'^(\s+)```', line) and not processed:
                indent_match = re.match(r'^(\s+)', line)
                current_indent = len(indent_match.group(1))
                
                # For markdown, code blocks in lists need to be indented by 8 spaces (or 2 tabs)
                # to be recognized as code blocks rather than continuation of the list item
                if current_indent >= 4 and current_indent < 8:
                    # This might be a code block in a list that needs more indentation
                    # Look back to see if we're in a list context
                    in_list_context = False
                    for j in range(max(0, i-5), i):
                        if re.match(r'^(\s*)[-*+\d]+[\.\)]\s+', lines[j]):
                            in_list_context = True
                            break
                    
                    if in_list_context:
                        # We're in a list, need to indent to 8 spaces total
                        spaces_to_add = 8 - current_indent
                        fixed_lines.append(' ' * spaces_to_add + line)
                        i += 1
                        processed = True
                        
                        # Process code block content
                        while i < len(lines):
                            current_line = lines[i]
                            # Check if this is a closing ``` with similar indentation
                            if re.match(r'^(\s+)```\s*$', current_line):
                                # Add proper indentation to closing ```
                                fixed_lines.append(' ' * 8 + '```')
                                i += 1
                                break
                            else:
                                # Indent content lines
                                if current_line.strip():
                                    fixed_lines.append(' ' * spaces_to_add + current_line)
                                else:
                                    fixed_lines.append('')
                                i += 1
                        processed = False
                        continue
                
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)


class LaTeXExtension(Extension):
    """Extension to add LaTeX preprocessing."""
    
    def extendMarkdown(self, md):
        md.preprocessors.register(LaTeXPreprocessor(md), 'latex_fix', 25)


class MarkdownConverter:
    """Main converter class for Markdown to HTML conversion."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'conversion_cache.json'
        self.cache = self._load_cache()
        
        # Initialize markdown with extensions
        self.md = markdown.Markdown(extensions=[
            'extra',  # tables, fenced code blocks, etc.
            'codehilite',  # syntax highlighting
            'toc',  # table of contents
            'sane_lists',  # better list handling
            LaTeXExtension(),  # our custom LaTeX handler
        ])
    
    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Get MD5 hash of file content."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _convert_links(self, html: str) -> str:
        """Convert .md links to .html links."""
        # Pattern to match markdown links
        pattern = r'href="([^"]+\.md)(#[^"]+)?"'
        
        def replace_link(match):
            link = match.group(1)
            anchor = match.group(2) or ''
            html_link = link[:-3] + '.html'
            return f'href="{html_link}{anchor}"'
        
        return re.sub(pattern, replace_link, html)
    
    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                return line.strip()[2:].strip()
        return 'Untitled'
    
    def _build_navigation(self, current_file: Path, all_files: List[Path]) -> Dict:
        """Build navigation structure."""
        nav = {
            'current': str(current_file),
            'files': [],
            'prev': None,
            'next': None
        }
        
        # Sort files for consistent ordering
        sorted_files = sorted(all_files)
        
        # Build file list with titles
        for f in sorted_files:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                title = self._extract_title(content)
                nav['files'].append({
                    'path': str(f.relative_to(f.parent.parent) if f.parent != f.parent.parent else f.name),
                    'title': title,
                    'active': f == current_file
                })
        
        # Find prev/next
        current_idx = sorted_files.index(current_file)
        if current_idx > 0:
            prev_file = sorted_files[current_idx - 1]
            with open(prev_file, 'r', encoding='utf-8') as f:
                prev_title = self._extract_title(f.read())
            nav['prev'] = {
                'path': str(prev_file.relative_to(prev_file.parent.parent) if prev_file.parent != prev_file.parent.parent else prev_file.name),
                'title': prev_title
            }
        
        if current_idx < len(sorted_files) - 1:
            next_file = sorted_files[current_idx + 1]
            with open(next_file, 'r', encoding='utf-8') as f:
                next_title = self._extract_title(f.read())
            nav['next'] = {
                'path': str(next_file.relative_to(next_file.parent.parent) if next_file.parent != next_file.parent.parent else next_file.name),
                'title': next_title
            }
        
        return nav
    
    def convert_file(self, input_file: Path, output_file: Path, all_files: List[Path], use_cache: bool = True) -> bool:
        """Convert a single markdown file to HTML."""
        # Check cache
        file_hash = self._get_file_hash(input_file)
        cache_key = str(input_file)
        
        if use_cache and cache_key in self.cache and self.cache[cache_key]['hash'] == file_hash:
            # Use cached content
            content_html = self.cache[cache_key]['content']
            print(f"Using cached content for {input_file}")
        else:
            # Convert markdown to HTML
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Reset markdown instance to clear any state
            self.md.reset()
            content_html = self.md.convert(content)
            
            # Cache the result
            self.cache[cache_key] = {
                'hash': file_hash,
                'content': content_html
            }
            self._save_cache()
            print(f"Converted {input_file}")
        
        # Convert links
        content_html = self._convert_links(content_html)
        
        # Extract title
        with open(input_file, 'r', encoding='utf-8') as f:
            title = self._extract_title(f.read())
        
        # Build navigation
        nav = self._build_navigation(input_file, all_files)
        
        # Generate full HTML
        html = self._generate_html(title, content_html, nav, output_file)
        
        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return True
    
    def _generate_html(self, title: str, content: str, nav: Dict, output_file: Path) -> str:
        """Generate complete HTML with template."""
        # Calculate relative paths for assets
        depth = len(output_file.parent.parts) - len(Path('output').parts)
        asset_prefix = '../' * depth if depth > 0 else './'
        
        # Adjust navigation paths
        nav_html = self._generate_nav_html(nav, asset_prefix)
        prev_next_html = self._generate_prev_next_html(nav, asset_prefix)
        
        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{asset_prefix}assets/style.css">
    <link rel="stylesheet" href="{asset_prefix}assets/highlight.css">
    <script src="{asset_prefix}assets/script.js" defer></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: false,
                packages: {{'[+]': ['noerrors']}}
            }},
            options: {{
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }},
            loader: {{
                load: ['[tex]/noerrors']
            }}
        }};
    </script>
</head>
<body>
    <div class="container">
        <nav id="sidebar" class="sidebar">
            <div class="sidebar-header">
                <h3>目录</h3>
                <button id="sidebar-toggle" class="sidebar-toggle">
                    <span></span>
                    <span></span>
                    <span></span>
                </button>
            </div>
            {nav_html}
        </nav>
        
        <main class="content">
            <article>
                {content}
            </article>
            
            {prev_next_html}
        </main>
    </div>
</body>
</html>"""
    
    def _generate_nav_html(self, nav: Dict, asset_prefix: str) -> str:
        """Generate navigation HTML."""
        items = []
        for file_info in nav['files']:
            # For navigation, we want relative paths from root
            path = file_info['path'].replace('.md', '.html')
            # Remove any directory prefix for files in root
            if '/' not in path:
                nav_path = path
            else:
                nav_path = path
            active = 'active' if file_info['active'] else ''
            items.append(f'<li class="{active}"><a href="{asset_prefix}{nav_path}">{file_info["title"]}</a></li>')
        
        return f'<ul class="nav-list">{"".join(items)}</ul>'
    
    def _generate_prev_next_html(self, nav: Dict, asset_prefix: str) -> str:
        """Generate previous/next navigation HTML."""
        prev_html = ''
        next_html = ''
        
        if nav['prev']:
            path = nav['prev']['path'].replace('.md', '.html')
            prev_html = f'<a href="{asset_prefix}{path}" class="nav-link prev">← {nav["prev"]["title"]}</a>'
        
        if nav['next']:
            path = nav['next']['path'].replace('.md', '.html')
            next_html = f'<a href="{asset_prefix}{path}" class="nav-link next">{nav["next"]["title"]} →</a>'
        
        return f'<nav class="page-nav">{prev_html}{next_html}</nav>'
    
    def convert_directory(self, input_dir: Path, output_dir: Path, use_cache: bool = True):
        """Convert all markdown files in a directory."""
        # Find all markdown files
        md_files = list(input_dir.rglob('*.md'))
        
        if not md_files:
            print(f"No markdown files found in {input_dir}")
            return
        
        print(f"Found {len(md_files)} markdown files")
        
        # Convert each file
        for md_file in md_files:
            # Calculate relative path
            rel_path = md_file.relative_to(input_dir)
            output_file = output_dir / rel_path.with_suffix('.html')
            
            self.convert_file(md_file, output_file, md_files, use_cache)
        
        # Copy static assets
        self._copy_assets(output_dir)
        
        print(f"\nConversion complete! Output in {output_dir}")
    
    def _copy_assets(self, output_dir: Path):
        """Copy static assets to output directory."""
        assets_dir = output_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        # Create CSS file
        css_file = assets_dir / 'style.css'
        css_file.write_text(self._get_css_content())
        
        # Create JavaScript file
        js_file = assets_dir / 'script.js'
        js_file.write_text(self._get_js_content())
        
        # Create syntax highlighting CSS
        highlight_css = assets_dir / 'highlight.css'
        highlight_css.write_text(self._get_highlight_css())
        
        print("Static assets copied")
    
    def _get_css_content(self) -> str:
        """Get CSS content."""
        return """/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --text-color: #333;
    --bg-color: #fff;
    --sidebar-bg: #f8f9fa;
    --border-color: #e0e0e0;
    --code-bg: #f4f4f4;
    --link-color: #3498db;
    --link-hover: #2980b9;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--bg-color);
}

/* Container layout */
.container {
    display: flex;
    min-height: 100vh;
}

/* Sidebar */
.sidebar {
    width: 300px;
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
    padding: 20px;
    overflow-y: auto;
    position: fixed;
    height: 100vh;
    left: 0;
    top: 0;
    transition: transform 0.3s ease;
}

.sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
}

.sidebar-header h3 {
    color: var(--primary-color);
    font-size: 1.2rem;
}

.sidebar-toggle {
    display: none;
    background: none;
    border: none;
    cursor: pointer;
    padding: 5px;
    width: 30px;
    height: 30px;
    position: relative;
}

.sidebar-toggle span {
    display: block;
    width: 20px;
    height: 2px;
    background-color: var(--primary-color);
    margin: 4px 0;
    transition: 0.3s;
}

.nav-list {
    list-style: none;
}

.nav-list li {
    margin-bottom: 8px;
}

.nav-list a {
    color: var(--text-color);
    text-decoration: none;
    display: block;
    padding: 8px 12px;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.nav-list a:hover {
    background-color: rgba(52, 152, 219, 0.1);
    color: var(--link-color);
}

.nav-list .active a {
    background-color: var(--secondary-color);
    color: white;
}

/* Main content */
.content {
    flex: 1;
    margin-left: 300px;
    padding: 40px;
    max-width: 900px;
    width: 100%;
}

article {
    margin-bottom: 40px;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--primary-color);
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

h1 { font-size: 2.2em; border-bottom: 2px solid var(--border-color); padding-bottom: 0.3em; }
h2 { font-size: 1.8em; }
h3 { font-size: 1.5em; }
h4 { font-size: 1.3em; }
h5 { font-size: 1.1em; }
h6 { font-size: 1em; }

p {
    margin-bottom: 1em;
}

a {
    color: var(--link-color);
    text-decoration: none;
}

a:hover {
    color: var(--link-hover);
    text-decoration: underline;
}

/* Lists */
ul, ol {
    margin-bottom: 1em;
    padding-left: 2em;
}

li {
    margin-bottom: 0.5em;
}

/* Code blocks */
pre {
    background-color: var(--code-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 16px;
    overflow-x: auto;
    margin-bottom: 1em;
}

code {
    background-color: var(--code-bg);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    font-size: 0.9em;
}

pre code {
    background-color: transparent;
    padding: 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1em;
}

th, td {
    border: 1px solid var(--border-color);
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: var(--sidebar-bg);
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: rgba(0, 0, 0, 0.02);
}

/* Blockquotes */
blockquote {
    border-left: 4px solid var(--secondary-color);
    padding-left: 20px;
    margin: 1em 0;
    color: #666;
}

/* Page navigation */
.page-nav {
    display: flex;
    justify-content: space-between;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
}

.nav-link {
    display: inline-block;
    padding: 10px 20px;
    background-color: var(--sidebar-bg);
    border-radius: 4px;
    transition: background-color 0.2s;
}

.nav-link:hover {
    background-color: rgba(52, 152, 219, 0.1);
    text-decoration: none;
}

.nav-link.prev {
    margin-right: auto;
}

.nav-link.next {
    margin-left: auto;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .sidebar {
        transform: translateX(-100%);
        z-index: 1000;
    }
    
    .sidebar.active {
        transform: translateX(0);
    }
    
    .sidebar-toggle {
        display: block;
    }
    
    .content {
        margin-left: 0;
        padding: 20px;
    }
    
    .page-nav {
        flex-direction: column;
        gap: 10px;
    }
    
    .nav-link {
        width: 100%;
        text-align: center;
    }
    
    h1 { font-size: 1.8em; }
    h2 { font-size: 1.5em; }
    h3 { font-size: 1.3em; }
}

/* Dark mode preparation */
@media (prefers-color-scheme: dark) {
    :root {
        --primary-color: #ecf0f1;
        --secondary-color: #3498db;
        --text-color: #ecf0f1;
        --bg-color: #1a1a1a;
        --sidebar-bg: #2c3e50;
        --border-color: #34495e;
        --code-bg: #2c3e50;
    }
}

/* Math display */
.MathJax_Display {
    overflow-x: auto;
    overflow-y: hidden;
}"""
    
    def _get_js_content(self) -> str:
        """Get JavaScript content."""
        return """// Sidebar toggle for mobile
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        const isClickInside = sidebar.contains(event.target);
        const isToggleClick = sidebarToggle.contains(event.target);
        
        if (!isClickInside && !isToggleClick && sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
        }
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});"""
    
    def _get_highlight_css(self) -> str:
        """Get syntax highlighting CSS."""
        return """.codehilite { background: #f8f8f8; }
.codehilite .hll { background-color: #ffffcc }
.codehilite .c { color: #999988; font-style: italic } /* Comment */
.codehilite .err { color: #a61717; background-color: #e3d2d2 } /* Error */
.codehilite .k { color: #000000; font-weight: bold } /* Keyword */
.codehilite .o { color: #000000; font-weight: bold } /* Operator */
.codehilite .cm { color: #999988; font-style: italic } /* Comment.Multiline */
.codehilite .cp { color: #999999; font-weight: bold; font-style: italic } /* Comment.Preproc */
.codehilite .c1 { color: #999988; font-style: italic } /* Comment.Single */
.codehilite .cs { color: #999999; font-weight: bold; font-style: italic } /* Comment.Special */
.codehilite .gd { color: #000000; background-color: #ffdddd } /* Generic.Deleted */
.codehilite .ge { color: #000000; font-style: italic } /* Generic.Emph */
.codehilite .gr { color: #aa0000 } /* Generic.Error */
.codehilite .gh { color: #999999 } /* Generic.Heading */
.codehilite .gi { color: #000000; background-color: #ddffdd } /* Generic.Inserted */
.codehilite .go { color: #888888 } /* Generic.Output */
.codehilite .gp { color: #555555 } /* Generic.Prompt */
.codehilite .gs { font-weight: bold } /* Generic.Strong */
.codehilite .gu { color: #aaaaaa } /* Generic.Subheading */
.codehilite .gt { color: #aa0000 } /* Generic.Traceback */
.codehilite .kc { color: #000000; font-weight: bold } /* Keyword.Constant */
.codehilite .kd { color: #000000; font-weight: bold } /* Keyword.Declaration */
.codehilite .kn { color: #000000; font-weight: bold } /* Keyword.Namespace */
.codehilite .kp { color: #000000; font-weight: bold } /* Keyword.Pseudo */
.codehilite .kr { color: #000000; font-weight: bold } /* Keyword.Reserved */
.codehilite .kt { color: #445588; font-weight: bold } /* Keyword.Type */
.codehilite .m { color: #009999 } /* Literal.Number */
.codehilite .s { color: #dd1144 } /* Literal.String */
.codehilite .na { color: #008080 } /* Name.Attribute */
.codehilite .nb { color: #0086B3 } /* Name.Builtin */
.codehilite .nc { color: #445588; font-weight: bold } /* Name.Class */
.codehilite .no { color: #008080 } /* Name.Constant */
.codehilite .nd { color: #3c5d5d; font-weight: bold } /* Name.Decorator */
.codehilite .ni { color: #800080 } /* Name.Entity */
.codehilite .ne { color: #990000; font-weight: bold } /* Name.Exception */
.codehilite .nf { color: #990000; font-weight: bold } /* Name.Function */
.codehilite .nl { color: #990000; font-weight: bold } /* Name.Label */
.codehilite .nn { color: #555555 } /* Name.Namespace */
.codehilite .nt { color: #000080 } /* Name.Tag */
.codehilite .nv { color: #008080 } /* Name.Variable */
.codehilite .ow { color: #000000; font-weight: bold } /* Operator.Word */
.codehilite .w { color: #bbbbbb } /* Text.Whitespace */
.codehilite .mf { color: #009999 } /* Literal.Number.Float */
.codehilite .mh { color: #009999 } /* Literal.Number.Hex */
.codehilite .mi { color: #009999 } /* Literal.Number.Integer */
.codehilite .mo { color: #009999 } /* Literal.Number.Oct */
.codehilite .sb { color: #dd1144 } /* Literal.String.Backtick */
.codehilite .sc { color: #dd1144 } /* Literal.String.Char */
.codehilite .sd { color: #dd1144 } /* Literal.String.Doc */
.codehilite .s2 { color: #dd1144 } /* Literal.String.Double */
.codehilite .se { color: #dd1144 } /* Literal.String.Escape */
.codehilite .sh { color: #dd1144 } /* Literal.String.Heredoc */
.codehilite .si { color: #dd1144 } /* Literal.String.Interpol */
.codehilite .sx { color: #dd1144 } /* Literal.String.Other */
.codehilite .sr { color: #009926 } /* Literal.String.Regex */
.codehilite .s1 { color: #dd1144 } /* Literal.String.Single */
.codehilite .ss { color: #990073 } /* Literal.String.Symbol */
.codehilite .bp { color: #999999 } /* Name.Builtin.Pseudo */
.codehilite .vc { color: #008080 } /* Name.Variable.Class */
.codehilite .vg { color: #008080 } /* Name.Variable.Global */
.codehilite .vi { color: #008080 } /* Name.Variable.Instance */
.codehilite .il { color: #009999 } /* Literal.Number.Integer.Long */"""
    
    def clear_cache(self):
        """Clear the conversion cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.cache = {}
        print("Cache cleared")


def main():
    parser = argparse.ArgumentParser(description='Convert Markdown files to HTML with navigation')
    parser.add_argument('input', help='Input directory or file')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before conversion')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    converter = MarkdownConverter()
    
    if args.clear_cache:
        converter.clear_cache()
    
    if input_path.is_file() and input_path.suffix == '.md':
        # Single file conversion
        output_file = output_path / input_path.name.replace('.md', '.html')
        converter.convert_file(input_path, output_file, [input_path])
    elif input_path.is_dir():
        # Directory conversion
        converter.convert_directory(input_path, output_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
