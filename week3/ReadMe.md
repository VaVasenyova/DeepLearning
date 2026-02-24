# README.md Description / Описание для README.md

## 🇷🇺 Русский

### **Цель проекта (The Gradient Puzzle)**
Трансформировать входное изображение случайного шума (16x16) в структурированный плавный градиент **без использования целевых меток** (unsupervised learning). Модель должна научиться "переставлять" пиксели, а не копировать их.

### **Основные требования**
1.  **Архитектура:** Использование проекций нейросети (Сжатие, Расширение или Трансформация) вместо простого копирования.
2.  **Ограничение распределения:** Запрещено создавать новые цвета. Гистограмма выходного изображения должна соответствовать входному (Input Histogram ≈ Output Histogram).
3.  **Кастомная функция потерь:** Нельзя использовать стандартный MSE (приводит к Identity Mapping). Необходимо задать "намерение" модели через геометрию (гладкость + направление).

### **Реализация**
*   **Стек:** TensorFlow.js (работает прямо в браузере).
*   **Архитектуры:** Реализованы 3 режима: `Compression` (сжатие), `Expansion` (расширение), `Transformation` (трансформация).
*   **Функция потерь (Loss):** Комбинация трех компонентов:
    1.  `Distribution Match`: Совпадение статистик (среднее и дисперсия) для сохранения палитры.
    2.  `Smoothness`: Штраф за резкие перепады между соседними пикселями (Total Variation).
    3.  `Direction`: Принудительное формирование градиента слева направо.
*   **Интерфейс:** Интерактивное обучение, сравнение базовой модели (MSE) и студента (Custom Loss), визуализация в реальном времени.

---

## 🇬🇧 English

### **Project Objective (The Gradient Puzzle)**
Transform an input image of random noise (16x16) into a structured smooth gradient **without using target labels** (unsupervised learning). The model must learn to "rearrange" pixels rather than copy them.

### **Key Requirements**
1.  **Architecture:** Use Neural Network Projections (Compression, Expansion, or Transformation) instead of simple copying.
2.  **Distribution Constraint:** Cannot create new colors. The output histogram must match the input (Input Histogram ≈ Output Histogram).
3.  **Custom Loss Function:** Standard MSE is forbidden (leads to Identity Mapping). Must define "intent" through geometry (smoothness + direction).

### **Implementation Details**
*   **Stack:** TensorFlow.js (runs directly in the browser).
*   **Architectures:** 3 modes implemented: `Compression`, `Expansion`, `Transformation`.
*   **Loss Function:** A combination of three components:
    1.  `Distribution Match`: Matches statistics (mean & variance) to conserve the color palette.
    2.  `Smoothness`: Penalizes sharp differences between adjacent pixels (Total Variation).
    3.  `Direction`: Enforces a left-to-right gradient pattern.
*   **UI:** Interactive training, comparison between Baseline model (MSE) and Student (Custom Loss), real-time visualization.
