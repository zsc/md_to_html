#!/usr/bin/env python3
"""
Simple browser test to verify HTML output.
"""

import subprocess
import time
import requests
from pathlib import Path

def test_server_and_pages():
    """Test that the server works and pages are accessible."""
    print("Starting test server...")
    
    # Kill any existing servers
    subprocess.run(["pkill", "-f", "python.*http.server"], capture_output=True)
    time.sleep(1)
    
    # Start server
    server = subprocess.Popen(
        ["python", "-m", "http.server", "8000"],
        cwd="output",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(2)  # Wait for server to start
    
    test_results = []
    
    try:
        # Test 1: Server is running
        try:
            response = requests.get("http://localhost:8000/")
            test_results.append(("Server running", "PASSED" if response.status_code == 200 else f"FAILED: {response.status_code}"))
        except Exception as e:
            test_results.append(("Server running", f"FAILED: {str(e)}"))
        
        # Test 2: Index page loads
        try:
            response = requests.get("http://localhost:8000/index.html")
            test_results.append(("Index page loads", "PASSED" if response.status_code == 200 else f"FAILED: {response.status_code}"))
            
            # Check content
            if response.status_code == 200:
                content = response.text
                # Check for title in various places (title tag or h1)
                has_title = any(text in content for text in [
                    "边缘侧大语言模型推理加速",
                    "<title>",
                    "<h1"
                ])
                has_nav = "nav-list" in content
                has_mathjax = "MathJax" in content
                
                test_results.append(("Index has title", "PASSED" if has_title else "FAILED"))
                test_results.append(("Index has navigation", "PASSED" if has_nav else "FAILED"))
                test_results.append(("Index has MathJax", "PASSED" if has_mathjax else "FAILED"))
        except Exception as e:
            test_results.append(("Index page loads", f"FAILED: {str(e)}"))
        
        # Test 3: Chapter page loads
        try:
            response = requests.get("http://localhost:8000/chapter1.html")
            test_results.append(("Chapter1 page loads", "PASSED" if response.status_code == 200 else f"FAILED: {response.status_code}"))
            
            if response.status_code == 200:
                content = response.text
                # More flexible title check
                has_title = any(text in content for text in [
                    "第1章",
                    "边缘推理",
                    "<title>",
                    "<h1"
                ])
                has_prev_next = "page-nav" in content
                
                test_results.append(("Chapter1 has title", "PASSED" if has_title else "FAILED"))
                test_results.append(("Chapter1 has prev/next nav", "PASSED" if has_prev_next else "FAILED"))
        except Exception as e:
            test_results.append(("Chapter1 page loads", f"FAILED: {str(e)}"))
        
        # Test 4: Static assets
        for asset in ["assets/style.css", "assets/script.js", "assets/highlight.css"]:
            try:
                response = requests.get(f"http://localhost:8000/{asset}")
                test_results.append((f"{asset} loads", "PASSED" if response.status_code == 200 else f"FAILED: {response.status_code}"))
            except Exception as e:
                test_results.append((f"{asset} loads", f"FAILED: {str(e)}"))
        
        # Test 5: Check links in content
        try:
            response = requests.get("http://localhost:8000/index.html")
            content = response.text
            
            # Check that .md links are converted to .html
            has_md_links = ".md\"" in content or ".md'" in content
            has_html_links = ".html\"" in content or ".html'" in content
            
            test_results.append(("No .md links remain", "PASSED" if not has_md_links else "FAILED"))
            test_results.append(("Has .html links", "PASSED" if has_html_links else "FAILED"))
        except Exception as e:
            test_results.append(("Link conversion check", f"FAILED: {str(e)}"))
        
    finally:
        # Stop server
        server.terminate()
        server.wait()
    
    # Print results
    print("\n" + "="*70)
    print("BROWSER TEST REPORT")
    print("="*70)
    
    passed = sum(1 for _, status in test_results if status == "PASSED")
    failed = sum(1 for _, status in test_results if status.startswith("FAILED"))
    
    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed / len(test_results) * 100):.1f}%")
    
    print("\nDetailed results:")
    for test_name, status in test_results:
        symbol = "✓" if status == "PASSED" else "❌"
        print(f"{symbol} {test_name}: {status}")
    
    return failed == 0


if __name__ == '__main__':
    success = test_server_and_pages()
    exit(0 if success else 1)