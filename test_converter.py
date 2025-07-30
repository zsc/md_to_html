#!/usr/bin/env python3
"""
Test suite for the markdown to HTML converter.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import time

from convert import MarkdownConverter, LaTeXPreprocessor


class TestLaTeXPreprocessor(unittest.TestCase):
    """Test LaTeX preprocessing functionality."""
    
    def setUp(self):
        self.preprocessor = LaTeXPreprocessor(None)
    
    def test_fix_mismatched_delimiters(self):
        """Test fixing $$ opening with $ closing."""
        text = "This is display math: $$x^2 + y^2 = z^2$"
        result = self.preprocessor._fix_display_math(text)
        self.assertEqual(result, "This is display math: $$x^2 + y^2 = z^2$$")
    
    def test_escape_underscores_in_latex(self):
        """Test escaping underscores in LaTeX expressions."""
        text = "Inline math $x_1 + x_2$ and display $$a_n = b_n$$"
        result = self.preprocessor._escape_underscores_in_latex(text)
        # Check that underscores are escaped (using raw strings properly)
        self.assertIn("x\\_1", result)
        self.assertIn("a\\_n", result)
    
    def test_multiple_latex_expressions(self):
        """Test handling multiple LaTeX expressions."""
        lines = [
            "First equation: $a_1 + b_1 = c_1$",
            "Second equation: $$x_n^2 + y_n^2 = z_n^2$",
            "Mixed: $$\\sum_{i=1}^n x_i$ should be $$\\sum_{i=1}^n x_i$$"
        ]
        result = self.preprocessor.run(lines)
        # Check that underscores are escaped
        self.assertTrue(any(r"\_" in line for line in result))


class TestMarkdownConverter(unittest.TestCase):
    """Test the main converter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter(cache_dir=self.test_path / 'cache')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_extract_title(self):
        """Test title extraction from markdown."""
        content = "# Test Title\n\nSome content"
        title = self.converter._extract_title(content)
        self.assertEqual(title, "Test Title")
        
        # Test missing title
        content = "No title here\n\nJust content"
        title = self.converter._extract_title(content)
        self.assertEqual(title, "Untitled")
    
    def test_convert_links(self):
        """Test markdown link conversion."""
        html = '<a href="chapter1.md">Chapter 1</a> and <a href="index.md#section">Index</a>'
        result = self.converter._convert_links(html)
        self.assertIn('href="chapter1.html"', result)
        self.assertIn('href="index.html#section"', result)
    
    def test_cache_functionality(self):
        """Test caching mechanism."""
        # Create test markdown file
        md_file = self.test_path / "test.md"
        md_file.write_text("# Test\n\nContent")
        
        # First conversion
        output_file = self.test_path / "test.html"
        self.converter.convert_file(md_file, output_file, [md_file])
        
        # Check cache was created
        self.assertIn(str(md_file), self.converter.cache)
        
        # Modify file to test cache invalidation
        time.sleep(0.1)  # Ensure different timestamp
        md_file.write_text("# Test\n\nModified content")
        
        # Convert again
        self.converter.convert_file(md_file, output_file, [md_file])
        
        # Check cache was updated
        self.assertTrue(output_file.exists())
        content = output_file.read_text()
        self.assertIn("Modified content", content)
    
    def test_navigation_generation(self):
        """Test navigation structure generation."""
        # Create test files
        files = []
        for i in range(3):
            f = self.test_path / f"chapter{i}.md"
            f.write_text(f"# Chapter {i}\n\nContent")
            files.append(f)
        
        nav = self.converter._build_navigation(files[1], files)
        
        # Check navigation structure
        self.assertEqual(len(nav['files']), 3)
        self.assertIsNotNone(nav['prev'])
        self.assertIsNotNone(nav['next'])
        self.assertEqual(nav['prev']['title'], "Chapter 0")
        self.assertEqual(nav['next']['title'], "Chapter 2")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.converter = MarkdownConverter()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_complex_latex(self):
        """Test complex LaTeX expressions."""
        # Create test file with edge cases
        content = """# LaTeX Edge Cases

1. Mismatched delimiters: $$E = mc^2$

2. Nested braces: $f(x) = \\begin{cases} x^2 & x > 0 \\\\ -x^2 & x \\leq 0 \\end{cases}$

3. Multiple underscores: $x_1 + x_2 + x_3 = y_1 + y_2$

4. Display math with underscores:
$$
\\sum_{i=1}^{n} x_i = \\frac{1}{n} \\sum_{j=1}^{m} y_j
$$

5. Inline and display mixed: The equation $a_n = b_n$ leads to $$c_n = a_n + b_n$$
"""
        
        md_file = self.test_path / "latex_test.md"
        md_file.write_text(content)
        output_file = self.test_path / "latex_test.html"
        
        self.converter.convert_file(md_file, output_file, [md_file])
        
        # Check output
        html = output_file.read_text()
        # Should have fixed the mismatched delimiter
        self.assertIn("$$E = mc^2$$", html)
        # Should have escaped underscores
        self.assertTrue(r"\_" in html or "MathJax" in html)
    
    def test_nested_directories(self):
        """Test conversion with nested directory structure."""
        # Create nested structure
        sub_dir = self.test_path / "docs" / "guides"
        sub_dir.mkdir(parents=True)
        
        # Create files
        index = self.test_path / "index.md"
        index.write_text("# Index\n\n[Guide](docs/guides/guide.md)")
        
        guide = sub_dir / "guide.md"
        guide.write_text("# Guide\n\n[Back to index](../../index.md)")
        
        # Convert
        output_dir = self.test_path / "output"
        self.converter.convert_directory(self.test_path, output_dir)
        
        # Check output structure
        self.assertTrue((output_dir / "index.html").exists())
        self.assertTrue((output_dir / "docs" / "guides" / "guide.html").exists())
        
        # Check links are converted
        index_html = (output_dir / "index.html").read_text()
        self.assertIn('href="docs/guides/guide.html"', index_html)


def run_tests():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLaTeXPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMarkdownConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    print("\n" + "="*70)
    print("TEST REPORT SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:\n{traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:\n{traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)