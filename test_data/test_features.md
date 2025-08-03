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

### LaTeX Edge Cases and Special Tests

#### Test 1: Mismatched Delimiters (opening $$ but closing with single $)
This is a display equation that starts with double dollar:
$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$

#### Test 2: Underscores Without Escaping
Inline math with underscores: $x_1 + x_2 = y_total$ and $a_b_c + d_e_f = result_final$

Display math with multiple underscores:
$$\sum_{i=1}^{n} x_i \cdot weight_i = output_layer$$

#### Test 3: Mixed Delimiters in Text
The function $f(x) = x^2$ can be written as $$f(x) = x \cdot x$$ or even $f(x) = x \times x$.

#### Test 4: Complex Expressions with Special Characters
$$\mathcal{L}(\theta) = -\sum_{i=1}^{N} \log p_\theta(x_i) + \lambda \|\theta\|_2^2$$

#### Test 5: Nested Structures
$$\frac{d}{dx}\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\right]$$

#### Test 6: Greek Letters and Symbols
Inline: $\alpha$, $\beta$, $\gamma$, $\delta$, $\epsilon$, $\zeta$, $\eta$, $\theta$

Display:
$$\Gamma(\alpha) = \int_0^\infty t^{\alpha-1}e^{-t}dt$$

#### Test 7: Subscripts and Superscripts Mix
$x_1^2 + x_2^2 = r^2$ and $e^{i\theta} = \cos\theta + i\sin\theta$

#### Test 8: Multi-line Equations
$$
\begin{align}
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{H} &= \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t} \\
\nabla \cdot \mathbf{D} &= \rho \\
\nabla \cdot \mathbf{B} &= 0
\end{align}
$$

#### Test 9: Text in Math Mode
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

#### Test 10: Special Functions
$$\lim_{x \to \infty} \frac{\sin x}{x} = 0$$
$$\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdot \ldots \cdot x_n$$

#### Test 11: Fractions and Roots
Complex fraction: $\frac{\frac{a}{b}}{\frac{c}{d}} = \frac{ad}{bc}$

Nested roots: $\sqrt{\sqrt{x} + \sqrt{y}}$

#### Test 12: Brackets and Delimiters
$$\left\{ x \in \mathbb{R} : \left| x - \mu \right| < 3\sigma \right\}$$

#### Test 13: LaTeX in Code Comments
```python
# Calculate the loss: $L = -\sum_i y_i \log(\hat{y}_i)$
def cross_entropy_loss(y_true, y_pred):
    # Implements $L = -\sum_i y_i \log(\hat{y}_i)$
    return -np.sum(y_true * np.log(y_pred))
```

#### Test 14: Adjacent Math Expressions
Here are two formulas: $a^2 + b^2$$c^2 + d^2$ should be separate.

#### Test 15: Math with Markdown Formatting
**Bold math**: $\mathbf{v} = (v_1, v_2, v_3)$
*Italic math*: $\mathit{f}(x) = x^2$
***Bold italic***: $\boldsymbol{\alpha}$

### Test 16: escaped underscore
$$q(\mathbf{x}\_t | \mathbf{x}\_{t-1}) = \mathcal{N}(\mathbf{x}\_t; \sqrt{1-\beta\_t}\mathbf{x}\_{t-1}, \beta\_t\mathbf{I})$$

### Test 17: multiple inline $
让我们仔细解析这个公式的含义。这个条件分布告诉我们，给定第 $t-1$ 步的状态 $\mathbf{x}\_{t-1}$ ，第 $t$ 步的状态 $\mathbf{x}\_t$ 是如何生成的：

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
