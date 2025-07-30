# Markdown to HTML Converter

A Python-based markdown to HTML converter with special support for LaTeX math, tables, code highlighting, and multi-file navigation.

## Features

- 📝 **Markdown Support**: Full CommonMark support with extensions (tables, fenced code blocks, etc.)
- 🔢 **LaTeX Math**: MathJax integration with tolerance for common LaTeX errors
- 🎨 **Syntax Highlighting**: Code blocks with language-specific highlighting via Pygments
- 📱 **Responsive Design**: Mobile-first CSS with collapsible navigation
- 🔗 **Smart Navigation**: Automatic sidebar generation and prev/next links
- 💾 **Caching**: MD5-based content caching for faster repeated conversions
- 📁 **Directory Structure**: Preserves folder hierarchy in output

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd md_to_html

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Conversion

Convert a directory of markdown files:
```bash
python convert.py input_dir/ output_dir/
```

Convert a single file:
```bash
python convert.py input.md output_dir/
```

### Options

Clear cache before conversion:
```bash
python convert.py input_dir/ output_dir/ --clear-cache
```

## Project Structure

```
md_to_html/
├── convert.py              # Main conversion script
├── requirements.txt        # Python dependencies
├── test_converter.py       # Unit tests
├── test_markdown_edge_cases.py  # Extended tests
├── test_browser_simple.py  # Browser integration tests
├── test_data/             # Sample markdown files
│   ├── index.md
│   ├── chapter1.md
│   └── test_features.md
└── README.md
```

## Special Features

### LaTeX Support

The converter handles common LaTeX errors:
- Mismatched delimiters (`$$...$` → `$$...$$`)
- Unescaped underscores in math expressions

### Table Processing

- Automatically adds blank lines before tables when needed
- Preserves column alignment
- Supports inline formatting within table cells

### Code Blocks in Lists

- Proper indentation for code blocks within lists
- Syntax highlighting maintained
- Handles nested list structures

## Testing

Run unit tests:
```bash
python test_converter.py
python test_markdown_edge_cases.py
```

Run browser tests:
```bash
python test_browser_simple.py
```

## Output

The converter generates:
- HTML files with the same structure as input
- `/assets` directory with CSS and JavaScript
- Responsive navigation sidebar
- MathJax-rendered mathematics
- Syntax-highlighted code blocks

## Requirements

- Python 3.6+
- markdown==3.5.1
- Pygments==2.17.2
- requests (for testing)

## License

This project is open source and available under the MIT License.