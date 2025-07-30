#!/usr/bin/env python3
"""
Extended unit tests for markdown edge cases including tables and code blocks.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import re

from convert import MarkdownConverter


class TestTableHandling(unittest.TestCase):
    """Test markdown table processing."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter(cache_dir=self.test_path / 'cache')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_simple_table(self):
        """Test basic markdown table conversion."""
        content = """# Table Test

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
"""
        md_file = self.test_path / "table_test.md"
        md_file.write_text(content)
        output_file = self.test_path / "table_test.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check table structure
        self.assertIn("<table>", html)
        self.assertIn("</table>", html)
        self.assertIn("<thead>", html)
        self.assertIn("<tbody>", html)
        self.assertIn("Column 1", html)
        self.assertIn("Cell 1", html)
    
    def test_table_with_alignment(self):
        """Test table with column alignment."""
        content = """# Aligned Table

| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |
| L2   | C2     | R2    |
"""
        md_file = self.test_path / "aligned_table.md"
        md_file.write_text(content)
        output_file = self.test_path / "aligned_table.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check for table
        self.assertIn("<table>", html)
        # Python-Markdown adds style attributes for alignment
        self.assertIn("Left", html)
        self.assertIn("Center", html)
        self.assertIn("Right", html)
        # Check alignment styles
        self.assertIn('style="text-align: left;"', html)
        self.assertIn('style="text-align: center;"', html)
        self.assertIn('style="text-align: right;"', html)
    
    def test_table_with_inline_formatting(self):
        """Test table with inline markdown formatting."""
        content = """# Formatted Table

| Feature | Description | Status |
|---------|-------------|--------|
| **Bold** | *Italic* text | `code` |
| [Link](test.md) | ~~Strike~~ | $x^2$ |
"""
        md_file = self.test_path / "formatted_table.md"
        md_file.write_text(content)
        output_file = self.test_path / "formatted_table.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check inline formatting
        self.assertIn("Bold", html)
        self.assertIn("Italic", html) 
        self.assertIn("code", html)
        self.assertIn('href="test.html"', html)  # Link should be converted
        self.assertIn("Strike", html)
        self.assertIn("$x^2$", html)  # LaTeX should be preserved


class TestCodeBlockHandling(unittest.TestCase):
    """Test code block processing and syntax highlighting."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter(cache_dir=self.test_path / 'cache')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_fenced_code_block(self):
        """Test fenced code blocks with language specification."""
        content = '''# Code Test

```python
def hello_world():
    print("Hello, World!")
    return 42
```

```javascript
function helloWorld() {
    console.log("Hello, World!");
    return 42;
}
```
'''
        md_file = self.test_path / "code_test.md"
        md_file.write_text(content)
        output_file = self.test_path / "code_test.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check for code highlighting
        self.assertIn('class="codehilite"', html)
        self.assertIn("<pre>", html)
        # Check that code content is preserved (may be in spans)
        self.assertIn("hello_world", html)
        self.assertIn("helloWorld", html)
    
    def test_indented_code_block(self):
        """Test indented code blocks."""
        content = """# Indented Code

Here's some code:

    def example():
        return "indented"
    
    print(example())

End of code.
"""
        md_file = self.test_path / "indented_code.md"
        md_file.write_text(content)
        output_file = self.test_path / "indented_code.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check for code block
        self.assertIn("<pre>", html)
        self.assertIn("<code>", html)
        # Content should be preserved
        self.assertIn("example", html)
        self.assertIn("indented", html)
    
    def test_code_block_with_special_chars(self):
        """Test code blocks containing special characters."""
        content = '''# Special Characters in Code

```python
# Special characters: < > & " '
html = "<div class='test'>&nbsp;</div>"
regex = r"\\d+\\.\\d+"
```

```bash
echo "Hello $USER"
if [ $? -eq 0 ]; then
    echo "Success!"
fi
```
'''
        md_file = self.test_path / "special_code.md"
        md_file.write_text(content)
        output_file = self.test_path / "special_code.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check that special characters are properly escaped
        self.assertIn("&lt;", html)  # < should be escaped
        self.assertIn("&gt;", html)  # > should be escaped
        self.assertIn("&amp;", html)  # & should be escaped
        # Check that code structure is preserved
        self.assertIn("echo", html)
        self.assertIn("Success!", html)
    
    def test_inline_code(self):
        """Test inline code handling."""
        content = """# Inline Code Test

Use `print()` function to output text. The `<div>` tag creates a division.

Complex inline: `arr[i] = x < y ? x : y`
"""
        md_file = self.test_path / "inline_code.md"
        md_file.write_text(content)
        output_file = self.test_path / "inline_code.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check inline code
        self.assertIn("<code>print()</code>", html)
        self.assertIn("<code>&lt;div&gt;</code>", html)  # Special chars escaped
        # Check complex expression
        self.assertTrue("<code>" in html and "?" in html)


class TestMixedContent(unittest.TestCase):
    """Test combinations of different markdown elements."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter(cache_dir=self.test_path / 'cache')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_table_with_code_and_latex(self):
        """Test table containing code and LaTeX."""
        content = """# Complex Table

| Algorithm | Code | Complexity |
|-----------|------|------------|
| Bubble Sort | `for i in range(n): ...` | $O(n^2)$ |
| Quick Sort | `pivot = arr[mid]` | $O(n \\log n)$ |
| Merge Sort | See below | $O(n \\log n)$ |

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    # Implementation
```
"""
        md_file = self.test_path / "complex_table.md"
        md_file.write_text(content)
        output_file = self.test_path / "complex_table.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check all elements are present
        self.assertIn("<table>", html)
        self.assertIn("for i in range(n)", html)
        self.assertIn("$O(n^2)$", html)
        self.assertIn("merge_sort", html)
    
    def test_nested_lists_with_code(self):
        """Test nested lists containing code blocks."""
        content = """# Nested Lists

1. First item
   - Sub-item with `inline code`
   - Another sub-item
     ```python
     # Code in nested list
     print("nested")
     ```
2. Second item
   1. Numbered sub-item
   2. Another numbered item
      - Mixed nesting
"""
        md_file = self.test_path / "nested_lists.md"
        md_file.write_text(content)
        output_file = self.test_path / "nested_lists.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check list structure
        self.assertIn("<ol>", html)
        self.assertIn("<ul>", html)
        self.assertIn("<code>inline code</code>", html)
        self.assertIn('print("nested")', html)
    
    def test_blockquote_with_multiple_elements(self):
        """Test blockquotes containing various elements."""
        content = """# Blockquote Test

> **Note:** This is important!
> 
> Here's a table in a blockquote:
> 
> | A | B |
> |---|---|
> | 1 | 2 |
> 
> And some code:
> ```python
> print("quoted")
> ```
"""
        md_file = self.test_path / "blockquote_test.md"
        md_file.write_text(content)
        output_file = self.test_path / "blockquote_test.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check blockquote structure
        self.assertIn("<blockquote>", html)
        self.assertIn("<strong>Note:</strong>", html)
        # Table might be rendered differently in blockquote
        self.assertIn('print("quoted")', html)


class TestEdgeCasesExtended(unittest.TestCase):
    """Test additional edge cases."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter(cache_dir=self.test_path / 'cache')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_empty_table_cells(self):
        """Test tables with empty cells."""
        content = """# Empty Cells

| A | B | C |
|---|---|---|
| 1 |   | 3 |
|   | 2 |   |
"""
        md_file = self.test_path / "empty_cells.md"
        md_file.write_text(content)
        output_file = self.test_path / "empty_cells.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check table renders with empty cells
        self.assertIn("<table>", html)
        self.assertIn("<td>1</td>", html)
        self.assertIn("<td></td>", html)  # Empty cell
    
    def test_code_block_at_document_boundaries(self):
        """Test code blocks at the start and end of document."""
        content = '''```python
# Code at start
print("start")
```

# Middle Content

Some text here.

```python
# Code at end
print("end")
```'''
        md_file = self.test_path / "boundary_code.md"
        md_file.write_text(content)
        output_file = self.test_path / "boundary_code.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check both code blocks are rendered (with HTML entities)
        self.assertIn('start', html)
        self.assertIn('end', html)
        self.assertIn("<h1", html)  # Middle content header
        self.assertIn('codehilite', html)  # Should have syntax highlighting
    
    def test_latex_in_lists_and_tables(self):
        """Test LaTeX expressions in various contexts."""
        content = """# LaTeX Everywhere

## In Lists
- First item with $x^2 + y^2 = z^2$
- Second item with $$\\int_0^1 x dx = \\frac{1}{2}$$

## In Table
| Formula | Result |
|---------|--------|
| $e^{i\\pi} + 1 = 0$ | Euler's identity |
| $$\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}$$ | Basel problem |
"""
        md_file = self.test_path / "latex_contexts.md"
        md_file.write_text(content)
        output_file = self.test_path / "latex_contexts.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check LaTeX is preserved in all contexts
        self.assertIn("$x^2 + y^2 = z^2$", html)
        self.assertIn("$$\\int_0^1 x dx", html)
        self.assertIn("$e^{i\\pi} + 1 = 0$", html)
        # Underscores should be escaped
        self.assertTrue("\\sum\\_" in html or "\\sum_" in html)
    
    def test_horizontal_rules(self):
        """Test horizontal rules in various contexts."""
        content = """# Horizontal Rules

First section

---

Second section

***

Third section

___

End
"""
        md_file = self.test_path / "hr_test.md"
        md_file.write_text(content)
        output_file = self.test_path / "hr_test.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Count horizontal rules
        hr_count = html.count("<hr")
        self.assertEqual(hr_count, 3)
    
    def test_definition_lists(self):
        """Test definition list syntax (if supported)."""
        content = """# Definitions

Term 1
:   Definition 1

Term 2
:   Definition 2a
:   Definition 2b
"""
        md_file = self.test_path / "def_list.md"
        md_file.write_text(content)
        output_file = self.test_path / "def_list.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check if definition lists are supported
        if "<dl>" in html:
            self.assertIn("<dt>", html)
            self.assertIn("<dd>", html)
    
    def test_html_entities_in_markdown(self):
        """Test HTML entities and special characters."""
        content = """# HTML Entities

Copyright &copy; 2024

Quotes: &ldquo;Hello&rdquo;

Math: &alpha; + &beta; = &gamma;

Literal: &amp;nbsp; and &#8212;
"""
        md_file = self.test_path / "entities.md"
        md_file.write_text(content)
        output_file = self.test_path / "entities.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        html = output_file.read_text()
        
        # Check entities are preserved
        self.assertIn("&copy;", html)
        self.assertIn("&ldquo;", html)
        self.assertIn("&alpha;", html)
        self.assertIn("&amp;nbsp;", html)  # Should be double-escaped


def run_extended_tests():
    """Run extended test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTableHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeBlockHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestMixedContent))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCasesExtended))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("EXTENDED TEST REPORT")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_extended_tests()
    exit(0 if success else 1)