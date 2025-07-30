# Test Features Document

This document tests various markdown features including tables and code blocks.

## Tables

### Basic Table

| Feature | Description | Status |
|---------|-------------|--------|
| Tables | Markdown tables with headers | ✓ |
| Code | Syntax highlighted code blocks | ✓ |
| LaTeX | Mathematical expressions | ✓ |

### Aligned Table

| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left 1 | Center 1 | Right 1 |
| Left 2 | Center 2 | Right 2 |
| Left 3 | Center 3 | Right 3 |

### Complex Table with Formatting

| Algorithm | Time Complexity | Space Complexity | Code Example |
|-----------|-----------------|------------------|--------------|
| **Bubble Sort** | $O(n^2)$ | $O(1)$ | `for i in range(n):` |
| **Quick Sort** | $O(n \log n)$ avg | $O(\log n)$ | `pivot = arr[mid]` |
| **Merge Sort** | $O(n \log n)$ | $O(n)$ | See code below |

## Code Blocks

### Python Example

```python
def quicksort(arr):
    """Quick sort implementation with pivot selection."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Test the function
numbers = [3, 6, 8, 10, 1, 2, 1]
sorted_numbers = quicksort(numbers)
print(f"Sorted: {sorted_numbers}")
```

### JavaScript Example

```javascript
// Fibonacci sequence generator
function* fibonacci() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

// Usage
const fib = fibonacci();
for (let i = 0; i < 10; i++) {
    console.log(fib.next().value);
}
```

### Bash Script

```bash
#!/bin/bash

# System monitoring script
echo "System Information:"
echo "=================="

# CPU usage
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | \
    awk '{print "User: " $2 "% System: " $4 "%"}'

# Memory usage
echo -e "\nMemory Usage:"
free -h | grep "^Mem" | \
    awk '{print "Total: " $2 " Used: " $3 " Free: " $4}'

# Disk usage
echo -e "\nDisk Usage:"
df -h | grep "^/dev" | \
    awk '{print $1 ": " $5 " used"}'
```

### Code with Special Characters

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Special Characters: < > & " '</title>
    <style>
        .highlight { background: #ff0; }
        code { font-family: 'Courier New', monospace; }
    </style>
</head>
<body>
    <p>HTML entities: &lt; &gt; &amp; &quot; &apos;</p>
    <pre><code>if (x < 10 && y > 5) { return true; }</code></pre>
</body>
</html>
```

## Inline Code Examples

Here are some inline code examples:

- Use `print()` in Python or `console.log()` in JavaScript
- HTML tags like `<div>` and `<span>` structure content
- Regular expressions: `^[a-zA-Z0-9_]+$` matches identifiers
- Shell commands: `ls -la | grep ".txt"`
- Math in code: `result = (x < y) ? x : y`

## LaTeX in Various Contexts

### In Lists

1. Pythagorean theorem: $a^2 + b^2 = c^2$
2. Euler's identity: $e^{i\pi} + 1 = 0$
3. Summation: $$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

### In Tables

| Formula | Name | Value |
|---------|------|-------|
| $E = mc^2$ | Mass-energy equivalence | Energy equals mass times speed of light squared |
| $F = ma$ | Newton's second law | Force equals mass times acceleration |
| $\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$ | Gauss's law | Electric field divergence |

### Complex LaTeX

The quadratic formula:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

Matrix multiplication:
$$
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22}
\end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22}
\end{bmatrix}
$$

## Mixed Content in Lists

1. **First item** with `inline code`
   - Sub-item with *emphasis*
   - Another sub-item with a [link](index.md)
   
   ```python
   # Code block in list
   def nested_example():
       return "This is nested"
   ```

2. **Second item** with $LaTeX: x^2$
   - Table in list:
   
   | A | B |
   |---|---|
   | 1 | 2 |
   | 3 | 4 |

3. **Third item** with ~~strikethrough~~
   > Blockquote in list
   > Multiple lines

## Edge Cases

### Empty Table Cells

| Col1 | Col2 | Col3 |
|------|------|------|
| A    |      | C    |
|      | B    |      |
| X    | Y    | Z    |

### Very Long Code Lines

```python
# This is a very long line that might cause horizontal scrolling in the code block
very_long_variable_name_that_exceeds_normal_conventions = {"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4", "key5": "value5"}
```

### Unicode in Code

```python
# Unicode support test
def 你好世界():
    emoji = "😀 🎉 🐍"
    return f"Hello in Chinese: 你好, Emoji: {emoji}"

# Mathematical symbols
α = 0.5
β = 0.3
γ = α + β
print(f"α + β = γ → {α} + {β} = {γ}")
```

---

End of test document.