#!/usr/bin/env python3
"""
Headless browser tests for the HTML output using Playwright.
"""

import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright, expect
import subprocess
import time


class HeadlessTests:
    """Headless browser tests for converted HTML."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
    
    async def test_page_loads(self, page):
        """Test that pages load successfully."""
        print("Testing page loads...")
        
        # Test index page
        response = await page.goto(f"{self.base_url}/index.html")
        assert response.status == 200, f"Index page failed to load: {response.status}"
        
        # Test chapter page
        response = await page.goto(f"{self.base_url}/chapter1.html")
        assert response.status == 200, f"Chapter page failed to load: {response.status}"
        
        self.test_results.append(("Page loads", "PASSED"))
        print("✓ Pages load successfully")
    
    async def test_navigation_links(self, page):
        """Test navigation between pages."""
        print("\nTesting navigation links...")
        
        # Go to index
        await page.goto(f"{self.base_url}/index.html")
        
        # Click on chapter link in content
        await page.click('a[href="chapter1.html"]')
        await page.wait_for_load_state('networkidle')
        
        # Verify we're on chapter 1
        assert page.url.endswith('chapter1.html'), "Navigation to chapter failed"
        
        # Test sidebar navigation
        await page.goto(f"{self.base_url}/index.html")
        await page.click('.nav-list a[href="chapter1.html"]')
        await page.wait_for_load_state('networkidle')
        assert page.url.endswith('chapter1.html'), "Sidebar navigation failed"
        
        self.test_results.append(("Navigation links", "PASSED"))
        print("✓ Navigation works correctly")
    
    async def test_prev_next_navigation(self, page):
        """Test previous/next navigation buttons."""
        print("\nTesting prev/next navigation...")
        
        # Go to index (should have only next)
        await page.goto(f"{self.base_url}/index.html")
        
        # Check no previous button
        prev_button = await page.query_selector('.nav-link.prev')
        assert prev_button is None, "Index should not have previous button"
        
        # Click next
        await page.click('.nav-link.next')
        await page.wait_for_load_state('networkidle')
        assert page.url.endswith('chapter1.html'), "Next navigation failed"
        
        # Now we should have both prev and next
        prev_button = await page.query_selector('.nav-link.prev')
        assert prev_button is not None, "Chapter should have previous button"
        
        # Click previous
        await page.click('.nav-link.prev')
        await page.wait_for_load_state('networkidle')
        assert page.url.endswith('index.html'), "Previous navigation failed"
        
        self.test_results.append(("Prev/Next navigation", "PASSED"))
        print("✓ Previous/Next navigation works")
    
    async def test_mobile_responsive(self, page):
        """Test mobile responsive design."""
        print("\nTesting mobile responsiveness...")
        
        # Set mobile viewport
        await page.set_viewport_size({"width": 375, "height": 667})
        await page.goto(f"{self.base_url}/index.html")
        
        # Check sidebar is hidden by default
        sidebar = await page.query_selector('#sidebar')
        is_visible = await sidebar.is_visible()
        sidebar_style = await sidebar.get_attribute('style')
        
        # On mobile, sidebar should be transformed off-screen
        # Check if hamburger menu is visible
        hamburger = await page.query_selector('#sidebar-toggle')
        is_hamburger_visible = await hamburger.is_visible()
        assert is_hamburger_visible, "Hamburger menu not visible on mobile"
        
        # Click hamburger to open sidebar
        await page.click('#sidebar-toggle')
        await page.wait_for_timeout(500)  # Wait for animation
        
        # Check sidebar is now visible
        sidebar_classes = await sidebar.get_attribute('class')
        assert 'active' in sidebar_classes, "Sidebar should be active after hamburger click"
        
        self.test_results.append(("Mobile responsive", "PASSED"))
        print("✓ Mobile responsive design works")
    
    async def test_latex_rendering(self, page):
        """Test that LaTeX/MathJax renders correctly."""
        print("\nTesting LaTeX rendering...")
        
        await page.goto(f"{self.base_url}/chapter1.html")
        
        # Wait for MathJax to load
        await page.wait_for_timeout(2000)
        
        # Check if MathJax has processed the page
        mathjax_elements = await page.query_selector_all('.MathJax')
        assert len(mathjax_elements) > 0, "No MathJax elements found"
        
        # Check for display math
        display_math = await page.query_selector('.MathJax_Display')
        assert display_math is not None, "Display math not rendered"
        
        self.test_results.append(("LaTeX rendering", "PASSED"))
        print("✓ LaTeX/MathJax renders correctly")
    
    async def test_code_highlighting(self, page):
        """Test code syntax highlighting."""
        print("\nTesting code highlighting...")
        
        await page.goto(f"{self.base_url}/chapter1.html")
        
        # Look for code blocks with highlighting
        highlighted_code = await page.query_selector('.codehilite')
        if highlighted_code:
            # Check if syntax highlighting classes are present
            code_content = await highlighted_code.inner_html()
            assert 'class=' in code_content, "No syntax highlighting classes found"
            self.test_results.append(("Code highlighting", "PASSED"))
            print("✓ Code syntax highlighting works")
        else:
            self.test_results.append(("Code highlighting", "SKIPPED - No code blocks"))
            print("- Code highlighting skipped (no code blocks found)")
    
    async def test_anchor_links(self, page):
        """Test internal anchor links."""
        print("\nTesting anchor links...")
        
        await page.goto(f"{self.base_url}/chapter1.html")
        
        # Check if there are any heading anchors
        headings = await page.query_selector_all('h2, h3, h4')
        if headings:
            first_heading = headings[0]
            heading_id = await first_heading.get_attribute('id')
            
            if heading_id:
                # Test anchor navigation
                await page.goto(f"{self.base_url}/chapter1.html#{heading_id}")
                await page.wait_for_timeout(500)
                
                # Check if scrolled to element
                is_in_viewport = await first_heading.is_in_viewport()
                assert is_in_viewport, "Anchor link navigation failed"
                
                self.test_results.append(("Anchor links", "PASSED"))
                print("✓ Anchor links work correctly")
            else:
                self.test_results.append(("Anchor links", "SKIPPED - No IDs"))
                print("- Anchor links skipped (no heading IDs)")
        else:
            self.test_results.append(("Anchor links", "SKIPPED - No headings"))
            print("- Anchor links skipped (no headings found)")
    
    async def run_all_tests(self):
        """Run all headless tests."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await self.test_page_loads(page)
                await self.test_navigation_links(page)
                await self.test_prev_next_navigation(page)
                await self.test_mobile_responsive(page)
                await self.test_latex_rendering(page)
                await self.test_code_highlighting(page)
                await self.test_anchor_links(page)
            except Exception as e:
                print(f"\n❌ Test failed with error: {e}")
                self.test_results.append(("Error", f"FAILED: {str(e)}"))
            finally:
                await browser.close()
        
        # Print summary
        print("\n" + "="*70)
        print("HEADLESS TEST REPORT")
        print("="*70)
        
        passed = sum(1 for _, status in self.test_results if status == "PASSED")
        failed = sum(1 for _, status in self.test_results if status.startswith("FAILED"))
        skipped = sum(1 for _, status in self.test_results if status.startswith("SKIPPED"))
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Success rate: {(passed / len(self.test_results) * 100):.1f}%")
        
        print("\nDetailed results:")
        for test_name, status in self.test_results:
            symbol = "✓" if status == "PASSED" else "❌" if status.startswith("FAILED") else "-"
            print(f"{symbol} {test_name}: {status}")
        
        return failed == 0


async def main():
    """Main function to run headless tests."""
    # First ensure the server is running
    print("Starting local server...")
    server = subprocess.Popen(
        ["python", "-m", "http.server", "8000"],
        cwd="output",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Give server time to start
    time.sleep(2)
    
    try:
        # Install playwright browsers if needed
        subprocess.run(["playwright", "install", "chromium"], capture_output=True)
        
        # Run tests
        tester = HeadlessTests()
        success = await tester.run_all_tests()
        
        return 0 if success else 1
    finally:
        # Clean up server
        server.terminate()
        server.wait()


if __name__ == '__main__':
    exit_code = asyncio.run(main())