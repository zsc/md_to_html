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


class LaTeXPreprocessor:
    """Preprocessor to handle LaTeX math with tolerance for common errors."""
    
    def __init__(self, md=None):
        """Initialize preprocessor (md parameter kept for compatibility)."""
        pass
    
    def run(self, lines):
        text = '\n'.join(lines)
        
        # Fix common LaTeX errors
        # 1. Fix multiline display math
        text = self._fix_multiline_display_math(text)
        
        # 2. Handle mismatched $$ and $
        text = self._fix_display_math(text)
        
        # 3. Escape underscores in LaTeX expressions
        text = self._escape_underscores_in_latex(text)
        
        # 4. Fix tables that need blank lines (including in blockquotes)
        text = self._fix_tables(text)
        
        # 5. Fix code blocks in lists
        text = self._fix_code_blocks_in_lists(text)
        
        # 6. Fix lists that need blank lines
        text = self._fix_lists(text)
        
        # 7. Convert bold numbered lists to proper list format
        text = self._convert_bold_lists(text)
        
        # 8. Add markdown="1" to HTML blocks that need markdown processing
        text = self._add_markdown_attribute(text)
        
        # Debug: print if preprocessor is being called
        import os
        if os.environ.get('DEBUG_PREPROCESSOR'):
            print("PREPROCESSOR CALLED!")
            print(f"First line after processing: {text.split(chr(10))[0] if text else 'EMPTY'}")
        
        return text.split('\n')
    
    def _fix_multiline_display_math(self, text):
        """Fix display math blocks that have multiple blank lines."""
        # First, handle cases where $$ is on its own line with content between
        pattern1 = r'\$\$\s*\n((?:(?!\$\$)[\s\S])*?)\n\s*\$\$'
        
        def fix_math(match):
            content = match.group(1)
            # Remove excessive blank lines within the math content
            lines = content.split('\n')
            # Keep only non-empty lines or single blank lines
            fixed_lines = []
            prev_blank = False
            for line in lines:
                if line.strip():
                    fixed_lines.append(line)
                    prev_blank = False
                elif not prev_blank:
                    fixed_lines.append(line)
                    prev_blank = True
            
            content = '\n'.join(fixed_lines).strip()
            return f'$$\n{content}\n$$'
        
        text = re.sub(pattern1, fix_math, text, flags=re.MULTILINE)
        
        # Also handle inline $$ with content on same line but closing $$ on separate line
        pattern2 = r'\$\$([^\$\n]+)\s*\n+\s*\$\$'
        text = re.sub(pattern2, r'$$\1$$', text, flags=re.MULTILINE)
        
        return text
    
    def _fix_display_math(self, text):
        """Fix mismatched display math delimiters."""
        # Pattern to find display math blocks that start with $$ but end with single $
        # Make sure we're really matching display math, not inline math
        # Look for $$ at word boundaries and ensure it's not preceded by non-whitespace
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Only process lines that contain $$
            if '$$' in line:
                # Use a more careful pattern that checks context
                # Match $$ followed by content and ending with single $ (not $$)
                pattern = r'(?:^|\s)\$\$([^$]+?)\$(?!\$)'
                if re.search(pattern, line):
                    lines[i] = re.sub(pattern, r' $$\1$$', line)
        return '\n'.join(lines)
    
    def _escape_underscores_in_latex(self, text):
        """Escape underscores and protect stars, backslashes, and braces within LaTeX expressions."""
        # Handle display math $$...$$
        def escape_in_display(match):
            content = match.group(1)
            # First, protect double backslashes in LaTeX (like \\ for line breaks in matrices)
            # We need to double them so markdown doesn't eat them
            content = content.replace('\\\\', '\\\\\\\\')
            # Protect curly braces in LaTeX (like \{ and \})
            content = content.replace('\\{', '\\\\{')
            content = content.replace('\\}', '\\\\}')
            # Only escape underscores that aren't already escaped
            # Replace _ with \_ but not \_ with \\_
            content = re.sub(r'(?<!\\)_', r'\\_', content)
            # Also protect stars from being interpreted as markdown
            content = re.sub(r'(?<!\\)\*', r'\\*', content)
            return f'$${content}$$'
        
        # Handle inline math $...$
        def escape_in_inline(match):
            content = match.group(1)
            # Avoid matching display math that we've already processed
            if content.startswith('$'):
                return match.group(0)
            # First, protect double backslashes in LaTeX
            content = content.replace('\\\\', '\\\\\\\\')
            # Protect curly braces in LaTeX
            content = content.replace('\\{', '\\\\{')
            content = content.replace('\\}', '\\\\}')
            # Only escape underscores that aren't already escaped
            content = re.sub(r'(?<!\\)_', r'\\_', content)
            # Also protect stars from being interpreted as markdown
            content = re.sub(r'(?<!\\)\*', r'\\*', content)
            return f'${content}$'
        
        # Process display math first
        text = re.sub(r'\$\$(.*?)\$\$', escape_in_display, text, flags=re.DOTALL)
        # Then process inline math (avoiding display math)
        # Fixed: The regex was not correctly handling inline math
        # Old pattern would miss some cases, let's use a more robust pattern
        text = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', escape_in_inline, text)
        
        return text
    
    def _fix_tables(self, text):
        """Ensure tables have blank lines before and after them."""
        lines = text.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line looks like a table separator (including in blockquotes)
            if re.match(r'^(\s*>)?\s*\|[\s\-:|]+\|\s*$', line):
                # Look back to find the header
                if i > 0 and '|' in lines[i-1]:
                    # Check if we're in a blockquote
                    in_blockquote = line.strip().startswith('>')
                    
                    # Check if there's already a blank line before the table
                    if i > 1 and new_lines and new_lines[-1].strip():
                        if in_blockquote and new_lines[-1].startswith('>'):
                            new_lines.append('>')  # Add blank blockquote line
                        else:
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
        """Fix code blocks that are inside lists by removing leading spaces and adding blank lines."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts with spaces followed by ```
            if re.match(r'^(\s+)```', line):
                indent_match = re.match(r'^(\s+)', line)
                spaces_to_remove = len(indent_match.group(1))
                
                # Check if we're in a list context by looking back
                in_list_context = False
                for j in range(max(0, i-5), i):
                    # Check for ordered or unordered list markers
                    if re.match(r'^\s*[-*+]\s+|^\s*\d+[.)]\s+', lines[j]):
                        in_list_context = True
                        break
                
                if in_list_context:
                    # Add a blank line before the code block if the previous line is not empty
                    if i > 0 and fixed_lines and fixed_lines[-1].strip():
                        fixed_lines.append('')
                    
                    # Remove leading spaces from the opening ```
                    fixed_lines.append(line[spaces_to_remove:] if len(line) >= spaces_to_remove else line)
                    i += 1
                    
                    # Process all lines until closing ```
                    while i < len(lines):
                        current_line = lines[i]
                        
                        # Check for closing ``` (with any indentation)
                        if re.match(r'^\s*```\s*$', current_line):
                            # Remove the same amount of spaces from closing ```
                            if current_line.startswith(' ' * spaces_to_remove):
                                fixed_lines.append(current_line[spaces_to_remove:])
                            else:
                                fixed_lines.append(current_line.lstrip())
                            i += 1
                            break
                        else:
                            # Remove the same amount of leading spaces from content lines
                            # but preserve any additional indentation
                            if len(current_line) >= spaces_to_remove and current_line[:spaces_to_remove] == ' ' * spaces_to_remove:
                                fixed_lines.append(current_line[spaces_to_remove:])
                            else:
                                # If line has fewer spaces or doesn't start with spaces, keep as is
                                fixed_lines.append(current_line)
                            i += 1
                    continue
                
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_lists(self, text):
        """Ensure lists have blank lines before them."""
        lines = text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Check if this line starts a list (-, *, +, or numbered)
            # Also handle bold markers (**) at the beginning
            # Also handle lists in blockquotes (starting with >)
            list_pattern = r'^(\s*>)?\s*(\*\*)?\s*[-*+]\s+|^(\s*>)?\s*(\*\*)?\s*\d+[.)]\s+'
            
            # Special pattern for blockquote lists
            blockquote_list_pattern = r'^>\s*[-*+]\s+|^>\s*\d+[.)]\s+'
            
            if re.match(list_pattern, line):
                # Check if previous line exists and is not empty and not a list item
                if i > 0 and fixed_lines and fixed_lines[-1].strip() and not re.match(list_pattern, fixed_lines[-1]):
                    # Special handling for blockquotes
                    if line.startswith('>') and fixed_lines[-1].startswith('>'):
                        # Insert a blank blockquote line
                        fixed_lines.append('>')
                    else:
                        # Add blank line before the list
                        fixed_lines.append('')
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _convert_bold_lists(self, text):
        """Convert bold numbered lists to proper HTML."""
        lines = text.split('\n')
        fixed_lines = []
        in_bold_list = False
        
        for i, line in enumerate(lines):
            # Pattern to match bold numbered items
            bold_pattern = r'^\s*\*\*(\d+)[.)]\s+(.+?)\*\*\s*$'
            match = re.match(bold_pattern, line)
            
            if match:
                # Found a bold numbered item
                num = match.group(1)
                content = match.group(2)
                
                # If this is the first item or previous line is blank, start new list
                if not in_bold_list or (i > 0 and not lines[i-1].strip()):
                    in_bold_list = True
                
                # Convert to numbered list with bold content
                fixed_lines.append(f'{num}. **{content}**')
            else:
                # Check if we're ending a bold list
                if in_bold_list and line.strip() and not re.match(r'^\s*[-*+]|\s*\d+[.)]', line):
                    in_bold_list = False
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _add_markdown_attribute(self, text):
        """Add markdown="1" attribute to HTML blocks that contain markdown content."""
        # Simple approach: add markdown="1" to all details tags
        text = re.sub(r'<details>', '<details markdown="1">', text)
        # Also handle cases where details already has attributes
        text = re.sub(r'<details\s+([^>]+)(?<!markdown="1")>', r'<details \1 markdown="1">', text)
        
        return text



class MarkdownConverter:
    """Main converter class for Markdown to HTML conversion."""
    
    def __init__(self, cache_dir: Path = None, debug: bool = False):
        self.cache_dir = cache_dir or Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'conversion_cache.json'
        self.cache = self._load_cache()
        self.debug = debug
        
        # Create debug directory if debug mode is enabled
        if self.debug:
            self.debug_dir = Path('debug_markdown')
            self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize markdown with extensions
        self.md = markdown.Markdown(extensions=[
            'extra',  # tables, fenced code blocks, etc.
            'codehilite',  # syntax highlighting
            'toc',  # table of contents
            'sane_lists',  # better list handling
            'md_in_html',  # process markdown inside HTML blocks
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
    
    def _sort_files(self, files: List[Path]) -> List[Path]:
        """Sort files with special rules for chapters and appendices."""
        def sort_key(path: Path):
            name = path.stem.lower()
            
            # Priority order: index, chapters, appendices, others
            if name == 'index':
                return (0, 0, name)
            elif name.startswith('chapter'):
                # Extract chapter number
                try:
                    num = int(name.replace('chapter', ''))
                    return (1, num, name)
                except ValueError:
                    return (1, 999, name)
            elif name.startswith('appendix'):
                # Extract appendix letter/number
                suffix = name.replace('appendix-', '').replace('appendix', '')
                return (2, suffix, name)
            else:
                # Other files come last
                return (3, 0, name)
        
        return sorted(files, key=sort_key)
    
    def _build_navigation(self, current_file: Path, all_files: List[Path]) -> Dict:
        """Build navigation structure."""
        nav = {
            'current': str(current_file),
            'files': [],
            'prev': None,
            'next': None,
            'use_tree': False  # Flag to determine tree vs flat layout
        }
        
        # Sort files for consistent ordering
        sorted_files = self._sort_files(all_files)
        
        # Find the common input directory (parent of all files)
        input_dir = sorted_files[0].parent
        for f in sorted_files:
            while not f.is_relative_to(input_dir):
                input_dir = input_dir.parent
        
        # Check if we should use tree structure
        # Criteria: more than 10 files OR files in multiple directories
        unique_dirs = set()
        for f in sorted_files:
            rel_path = f.relative_to(input_dir)
            if len(rel_path.parts) > 1:  # File is in a subdirectory
                unique_dirs.add(rel_path.parent)
        
        # Use tree if many files or multiple directories
        nav['use_tree'] = len(sorted_files) > 10 or len(unique_dirs) > 0
        
        # Build file list with titles
        for f in sorted_files:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                title = self._extract_title(content)
                rel_path = f.relative_to(input_dir)
                nav['files'].append({
                    'path': str(rel_path),
                    'title': title,
                    'active': f == current_file,
                    'directory': str(rel_path.parent) if len(rel_path.parts) > 1 else None
                })
        
        # Find prev/next
        current_idx = sorted_files.index(current_file)
        if current_idx > 0:
            prev_file = sorted_files[current_idx - 1]
            with open(prev_file, 'r', encoding='utf-8') as f:
                prev_title = self._extract_title(f.read())
            nav['prev'] = {
                'path': str(prev_file.relative_to(input_dir)),
                'title': prev_title
            }
        
        if current_idx < len(sorted_files) - 1:
            next_file = sorted_files[current_idx + 1]
            with open(next_file, 'r', encoding='utf-8') as f:
                next_title = self._extract_title(f.read())
            nav['next'] = {
                'path': str(next_file.relative_to(input_dir)),
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
            
            # Run preprocessor to fix markdown issues
            preprocessor = LaTeXPreprocessor(None)
            preprocessed_lines = preprocessor.run(content.split('\n'))
            preprocessed_content = '\n'.join(preprocessed_lines)
            
            # If debug mode is enabled, save preprocessed markdown
            if self.debug:
                # Save to debug directory
                debug_file = self.debug_dir / f"{input_file.stem}_preprocessed.md"
                debug_file.write_text(preprocessed_content, encoding='utf-8')
                print(f"Debug: Saved preprocessed markdown to {debug_file}")
            
            # Reset markdown instance to clear any state
            self.md.reset()
            # Use preprocessed content for conversion
            content_html = self.md.convert(preprocessed_content)
            
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
    <base href="{asset_prefix}">
    <title>{title}</title>
    <link rel="stylesheet" href="assets/style.css">
    <link rel="stylesheet" href="assets/highlight.css">
    <script src="assets/script.js" defer></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: false,
                packages: {{'[+]': ['noerrors', 'ams']}}
            }},
            options: {{
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }},
            loader: {{
                load: ['[tex]/noerrors', '[tex]/ams']
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
            <div class="sidebar-search">
                <input type="text" id="sidebar-search-input" placeholder="搜索..." autocomplete="off">
                <span class="search-clear" id="search-clear">✕</span>
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
        if not nav.get('use_tree', False):
            # Flat list for few items or single directory
            items = []
            for file_info in nav['files']:
                path = file_info['path'].replace('.md', '.html')
                active = 'active' if file_info['active'] else ''
                items.append(f'<li class="{active}"><a href="{path}">{file_info["title"]}</a></li>')
            
            return f'<ul class="nav-list">{"".join(items)}</ul>'
        else:
            # Tree structure for many items or multiple directories
            return self._generate_tree_nav_html(nav, asset_prefix)
    
    def _generate_tree_nav_html(self, nav: Dict, asset_prefix: str) -> str:
        """Generate tree-structured navigation HTML."""
        from collections import defaultdict
        import os
        
        # Build a proper tree structure
        tree = {}
        
        for file_info in nav['files']:
            path_parts = file_info['path'].split(os.sep)
            current_level = tree
            
            # Build nested structure for directories
            for i, part in enumerate(path_parts[:-1]):
                if part not in current_level:
                    current_level[part] = {'_files': [], '_dirs': {}}
                current_level = current_level[part]['_dirs']
            
            # Add file to its directory
            if len(path_parts) == 1:
                # Root level file
                if '_files' not in tree:
                    tree['_files'] = []
                tree['_files'].append(file_info)
            else:
                # File in subdirectory
                if '_files' not in current_level:
                    current_level['_files'] = []
                current_level['_files'].append(file_info)
        
        def generate_tree_html(node, path_prefix='', level=0):
            """Recursively generate HTML for tree nodes."""
            html = []
            
            # Add files at this level
            if '_files' in node:
                for file_info in node['_files']:
                    file_path = file_info['path'].replace('.md', '.html')
                    active = 'active' if file_info['active'] else ''
                    icon = '📄' if not active else '📝'
                    html.append(f'<li class="nav-file {active}" data-level="{level}">')
                    html.append(f'<span class="file-icon">{icon}</span>')
                    html.append(f'<a href="{file_path}" title="{file_info["title"]}">{file_info["title"]}</a>')
                    html.append('</li>')
            
            # Add subdirectories
            for dir_name in sorted(key for key in node.keys() if key not in ['_files', '_dirs']):
                dir_node = node[dir_name]
                # Check if any file in this directory tree is active
                # Need to check both _files and nested structure
                has_active = False
                if '_files' in dir_node:
                    has_active = any(f.get('active', False) for f in dir_node['_files'])
                if not has_active and '_dirs' in dir_node:
                    has_active = self._has_active_file(dir_node['_dirs'])
                
                expanded = 'expanded' if has_active else ''
                
                html.append(f'<li class="nav-directory {expanded}" data-level="{level}">')
                html.append(f'<div class="nav-directory-toggle">')
                html.append(f'<span class="toggle-icon">▼</span>')
                html.append(f'<span class="folder-icon">📁</span>')
                html.append(f'<span class="directory-name">{dir_name}</span>')
                html.append('</div>')
                html.append('<ul class="nav-subdirectory">')
                
                # Recursively add subdirectory contents
                subdirectory_content = []
                if '_dirs' in dir_node and dir_node['_dirs']:
                    subdirectory_content.extend(generate_tree_html(dir_node['_dirs'], 
                                                  os.path.join(path_prefix, dir_name), 
                                                  level + 1))
                if '_files' in dir_node and dir_node['_files']:
                    subdirectory_content.extend(generate_tree_html({'_files': dir_node['_files']}, 
                                                  os.path.join(path_prefix, dir_name), 
                                                  level + 1))
                
                html.extend(subdirectory_content)
                
                html.append('</ul>')
                html.append('</li>')
            
            return html
        
        html_parts = ['<div class="nav-tree-container"><ul class="nav-list nav-tree" role="tree">']
        html_parts.extend(generate_tree_html(tree))
        html_parts.append('</ul></div>')
        
        return ''.join(html_parts)
    
    def _has_active_file(self, node):
        """Check if any file in the tree node is active."""
        if isinstance(node, dict):
            if '_files' in node:
                for file_info in node['_files']:
                    if file_info.get('active', False):
                        return True
            
            if '_dirs' in node:
                for dir_node in node['_dirs'].values():
                    if self._has_active_file(dir_node):
                        return True
            
            for key, value in node.items():
                if key not in ['_files', '_dirs'] and isinstance(value, dict):
                    if self._has_active_file(value):
                        return True
        
        return False
    
    def _generate_prev_next_html(self, nav: Dict, asset_prefix: str) -> str:
        """Generate previous/next navigation HTML."""
        prev_html = ''
        next_html = ''
        
        if nav['prev']:
            path = nav['prev']['path'].replace('.md', '.html')
            prev_html = f'<a href="{path}" class="nav-link prev">← {nav["prev"]["title"]}</a>'
        
        if nav['next']:
            path = nav['next']['path'].replace('.md', '.html')
            next_html = f'<a href="{path}" class="nav-link next">{nav["next"]["title"]} →</a>'
        
        return f'<nav class="page-nav">{prev_html}{next_html}</nav>'
    
    def convert_directory(self, input_dir: Path, output_dir: Path, use_cache: bool = True, filter_pattern: str = None):
        """Convert all markdown files in a directory."""
        # Find all markdown files
        if filter_pattern:
            md_files = list(input_dir.glob(filter_pattern))
            # Also filter to ensure they are actually .md files
            md_files = [f for f in md_files if f.suffix == '.md' and f.is_file()]
            print(f"Using filter pattern: {filter_pattern}")
        else:
            md_files = list(input_dir.rglob('*.md'))
        
        if not md_files:
            if filter_pattern:
                print(f"No markdown files found in {input_dir} matching pattern '{filter_pattern}'")
            else:
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

/* Search input styles */
.sidebar-search {
    position: relative;
    margin-bottom: 20px;
}

#sidebar-search-input {
    width: 100%;
    padding: 8px 30px 8px 12px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 14px;
    background-color: var(--bg-color);
    color: var(--text-color);
    transition: border-color 0.2s;
}

#sidebar-search-input:focus {
    outline: none;
    border-color: var(--secondary-color);
}

.search-clear {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    cursor: pointer;
    color: #999;
    font-size: 18px;
    display: none;
    user-select: none;
    padding: 4px;
}

.search-clear:hover {
    color: var(--text-color);
}

.no-results {
    padding: 12px;
    text-align: center;
    color: #666;
    font-size: 14px;
    border: 1px dashed var(--border-color);
    border-radius: 4px;
    margin-bottom: 20px;
    background-color: rgba(0, 0, 0, 0.02);
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

/* Modern Tree Navigation */
.nav-tree-container {
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 4px;
}

.nav-tree {
    padding: 0;
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
}

.nav-tree li {
    list-style: none;
    margin: 0;
    padding: 0;
    position: relative;
}

/* File items */
.nav-file {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    margin: 1px 0;
    border-radius: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
}

.nav-file:hover {
    background-color: rgba(52, 152, 219, 0.08);
}

.nav-file.active {
    background-color: var(--secondary-color);
    color: white;
}

.nav-file a {
    flex: 1;
    color: inherit;
    text-decoration: none;
    padding: 0 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
}

.nav-file.active a {
    color: white;
}

/* Directory items */
.nav-directory {
    margin: 2px 0;
}

.nav-directory-toggle {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    margin: 1px 0;
    border-radius: 4px;
    cursor: pointer;
    user-select: none;
    transition: background-color 0.2s;
    font-weight: 500;
}

.nav-directory-toggle:hover {
    background-color: rgba(0, 0, 0, 0.04);
}

/* Icons */
.toggle-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    margin-right: 2px;
    transition: transform 0.2s ease;
    font-size: 10px;
    color: #666;
}

.nav-directory.expanded .toggle-icon {
    transform: rotate(0deg);
}

.nav-directory:not(.expanded) .toggle-icon {
    transform: rotate(-90deg);
}

.folder-icon, .file-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    margin-right: 6px;
    font-size: 14px;
    flex-shrink: 0;
}

.nav-directory.expanded .folder-icon {
    content: "📂";
}

.directory-name {
    flex: 1;
    font-weight: 500;
    color: var(--primary-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Subdirectory */
.nav-subdirectory {
    list-style: none;
    margin: 0;
    padding: 0 0 0 12px;
    overflow: hidden;
    max-height: 0;
    transition: max-height 0.3s ease;
}

.nav-directory.expanded .nav-subdirectory {
    max-height: 9999px;
}

/* Indentation for nested levels */
.nav-tree li[data-level="1"] {
    padding-left: 16px;
}

.nav-tree li[data-level="2"] {
    padding-left: 32px;
}

.nav-tree li[data-level="3"] {
    padding-left: 48px;
}

/* Tree lines (optional, for CHM-like appearance) */
.nav-tree li::before {
    content: "";
    position: absolute;
    left: 8px;
    top: 0;
    bottom: 0;
    width: 1px;
    background: linear-gradient(to bottom, transparent, #ddd 20%, #ddd 80%, transparent);
}

.nav-tree li[data-level="0"]::before {
    display: none;
}

/* Smooth scrollbar */
.nav-tree-container::-webkit-scrollbar {
    width: 6px;
}

.nav-tree-container::-webkit-scrollbar-track {
    background: transparent;
}

.nav-tree-container::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 3px;
}

.nav-tree-container::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
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
        width: 85vw;
        max-width: 350px;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar.active {
        transform: translateX(0);
    }
    
    .sidebar-toggle {
        display: block;
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1001;
        background: var(--bg-color);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
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
    
    /* Larger touch targets for mobile */
    .nav-file, .nav-directory-toggle {
        padding: 8px 12px;
        min-height: 44px;
    }
    
    .toggle-icon, .folder-icon, .file-icon {
        width: 24px;
        height: 24px;
        font-size: 16px;
    }
    
    .nav-tree {
        font-size: 16px;
    }
    
    /* Overlay when sidebar is open */
    .sidebar.active::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.3);
        z-index: -1;
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
    
    // Tree navigation state management
    const TREE_STATE_KEY = 'nav-tree-state';
    
    // Load saved tree state
    function loadTreeState() {
        try {
            const saved = localStorage.getItem(TREE_STATE_KEY);
            return saved ? JSON.parse(saved) : {};
        } catch (e) {
            return {};
        }
    }
    
    // Save tree state
    function saveTreeState() {
        const state = {};
        document.querySelectorAll('.nav-directory').forEach((dir, index) => {
            const dirName = dir.querySelector('.directory-name');
            if (dirName) {
                state[dirName.textContent] = dir.classList.contains('expanded');
            }
        });
        try {
            localStorage.setItem(TREE_STATE_KEY, JSON.stringify(state));
        } catch (e) {
            // Ignore localStorage errors
        }
    }
    
    // Apply saved state
    function applyTreeState() {
        const state = loadTreeState();
        document.querySelectorAll('.nav-directory').forEach(dir => {
            const dirName = dir.querySelector('.directory-name');
            if (dirName && state[dirName.textContent] !== undefined) {
                if (state[dirName.textContent]) {
                    dir.classList.add('expanded');
                } else {
                    dir.classList.remove('expanded');
                }
            }
        });
    }
    
    // Initialize tree state
    applyTreeState();
    
    // Tree navigation toggle functionality
    const navDirectories = document.querySelectorAll('.nav-directory-toggle');
    navDirectories.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const directory = this.parentElement;
            directory.classList.toggle('expanded');
            saveTreeState();
            
            // Smooth animation for mobile
            if (window.innerWidth <= 768) {
                const subdirectory = directory.querySelector('.nav-subdirectory');
                if (subdirectory) {
                    if (directory.classList.contains('expanded')) {
                        subdirectory.style.maxHeight = subdirectory.scrollHeight + 'px';
                    } else {
                        subdirectory.style.maxHeight = '0';
                    }
                }
            }
        });
    });
    
    // Double-click to expand/collapse all children
    navDirectories.forEach(toggle => {
        toggle.addEventListener('dblclick', function(e) {
            e.preventDefault();
            const directory = this.parentElement;
            const isExpanded = directory.classList.contains('expanded');
            const allSubDirs = directory.querySelectorAll('.nav-directory');
            
            allSubDirs.forEach(subDir => {
                if (isExpanded) {
                    subDir.classList.remove('expanded');
                } else {
                    subDir.classList.add('expanded');
                }
            });
            
            saveTreeState();
        });
    });
    
    // Instant search functionality
    const searchInput = document.getElementById('sidebar-search-input');
    const searchClear = document.getElementById('search-clear');
    const navList = document.querySelector('.nav-list');
    const isTreeNav = navList && navList.classList.contains('nav-tree');
    
    function performSearch() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        let visibleCount = 0;
        
        // Show/hide clear button
        searchClear.style.display = searchTerm ? 'block' : 'none';
        
        if (isTreeNav) {
            // Tree navigation search for new structure
            const allFiles = navList.querySelectorAll('.nav-file');
            const directories = navList.querySelectorAll('.nav-directory');
            
            // Search through all file items
            allFiles.forEach(item => {
                const link = item.querySelector('a');
                const text = link ? link.textContent.toLowerCase() : '';
                
                if (!searchTerm || text.includes(searchTerm)) {
                    item.style.display = '';
                    visibleCount++;
                    // Show all parent directories
                    let parent = item.parentElement;
                    while (parent && parent !== navList) {
                        if (parent.classList.contains('nav-directory')) {
                            parent.style.display = '';
                            if (searchTerm) {
                                parent.classList.add('expanded');
                            }
                        }
                        parent = parent.parentElement;
                    }
                } else {
                    item.style.display = 'none';
                }
            });
            
            // Handle directories visibility
            directories.forEach(dir => {
                const hasVisibleFiles = dir.querySelectorAll('.nav-file:not([style*="none"])').length > 0;
                const hasVisibleSubDirs = dir.querySelectorAll('.nav-directory:not([style*="none"])').length > 0;
                
                if (!searchTerm) {
                    dir.style.display = '';
                } else if (!hasVisibleFiles && !hasVisibleSubDirs) {
                    dir.style.display = 'none';
                }
            });
        } else {
            // Flat navigation search
            const navItems = navList ? navList.querySelectorAll('li') : [];
            navItems.forEach(item => {
                const link = item.querySelector('a');
                const text = link ? link.textContent.toLowerCase() : '';
                
                if (!searchTerm || text.includes(searchTerm)) {
                    item.style.display = '';
                    visibleCount++;
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        // Show a message if no results found
        let noResultsMsg = document.getElementById('no-search-results');
        if (searchTerm && visibleCount === 0) {
            if (!noResultsMsg) {
                noResultsMsg = document.createElement('div');
                noResultsMsg.id = 'no-search-results';
                noResultsMsg.className = 'no-results';
                noResultsMsg.textContent = '没有找到匹配的结果';
                navList.parentNode.insertBefore(noResultsMsg, navList);
            }
            noResultsMsg.style.display = 'block';
        } else if (noResultsMsg) {
            noResultsMsg.style.display = 'none';
        }
        
        // Restore original expanded state when search is cleared
        if (!searchTerm && isTreeNav) {
            const directories = navList.querySelectorAll('.nav-directory');
            directories.forEach(dir => {
                // Check if directory contains active item
                const hasActive = dir.querySelector('.nav-subdirectory .active');
                if (hasActive) {
                    dir.classList.add('expanded');
                } else {
                    dir.classList.remove('expanded');
                }
            });
        }
    }
    
    if (searchInput) {
        // Perform search on input
        searchInput.addEventListener('input', performSearch);
        
        // Clear search when clicking X
        searchClear.addEventListener('click', function() {
            searchInput.value = '';
            performSearch();
            searchInput.focus();
        });
        
        // Clear search with Escape key
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                searchInput.value = '';
                performSearch();
            }
        });
    }
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
    parser.add_argument('--debug', action='store_true', help='Save preprocessed markdown files for debugging')
    parser.add_argument('--filter', help='Glob pattern to filter files (e.g., "**/*_zh.md")')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    converter = MarkdownConverter(debug=args.debug)
    
    if args.clear_cache:
        converter.clear_cache()
    
    if input_path.is_file() and input_path.suffix == '.md':
        # Single file conversion
        output_file = output_path / input_path.name.replace('.md', '.html')
        converter.convert_file(input_path, output_file, [input_path])
    elif input_path.is_dir():
        # Directory conversion
        converter.convert_directory(input_path, output_path, filter_pattern=args.filter)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
