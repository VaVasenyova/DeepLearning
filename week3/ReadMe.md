# The Gradient Puzzle — Summary

## 🎯 Assignment Essence

**Goal:** Transform 16×16 random noise into a smooth gradient **without target labels**.

**Main Constraint:**
- ❌ Cannot create new colors
- ✅ Can only **rearrange** existing pixels
- 📊 Input Histogram ≈ Output Histogram

**Analogy:** Sliding puzzle — you move the tiles, you don't repaint them.

---

## 🏗️ Difference Between Architectures

| Architecture | Hidden Layer | Principle | Result |
|-------------|-------------|---------|-----------|
| **Compression** | 64 neurons | Compression (Many→Few) | Loses information, difficult |
| **Transformation** | 256 neurons | 1:1 (Same→Same) | Good balance |
| **Expansion** ✅ | 512 neurons | Expansion (Few→Many) | **Better** — more freedom |

---

## ⚙️ Why the Solution Works

### Loss Function

| Component | Weight | Effect |
|-----------|-----|--------|
| `MSE` | 0.01 | Minimal — doesn't block movement |
| `Smoothness` | 0.5 | Removes noise → smooth transitions |
| `Direction` | 0.3 | Darker on left, brighter on right |

**Result:** The model is forced to **rearrange** pixels to minimize loss.

---

## 🔄 What Changed

| Aspect | Before | After |
|--------|------|-------|
| **Architectures** | Only Compression | ✅ All three implemented |
| **Loss** | Only MSE (copying) | ✅ Smoothness + Direction |
| **Optimizer** | One for all | ✅ Separate (baseline + student) |
| **Result** | Noise / copy | ✅ **Smooth gradient** |

---

## 📌 Key Takeaway

> **"Sorted MSE liberates pixels from their positions, while Smoothness/Direction guides them to their new homes."**

**Expansion** gave the best result because 512 neurons (2× the input size) gave the model enough capacity to find the optimal solution.
