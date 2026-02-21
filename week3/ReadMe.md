# Neural Network Design: The Gradient Puzzle

## Описание
Проект по обучению нейронной сети преобразовывать случайный шум в структурированный градиент без использования целевых меток (unsupervised learning).

## Цель
Разработать архитектуру автоэнкодера и функцию потерь, которые перераспределят пиксели случайного шума 16×16 в плавный направленный градиент.

## Ключевое ограничение
**Нельзя создавать новые цвета** — можно только перемещать существующие пиксели (аналогия: скользящий пазл). Гистограмма входа должна совпадать с гистограммой выхода.

## Типы архитектур

1. **Compression** (Сжатие): 256 → 64 → 256  
   *Many → Few*: Выделение главных признаков

2. **Transformation** (Трансформация): 256 → 256 → 256  
   *Same → Same*: Переориентация без потери информации

3. **Expansion** (Расширение): 256 → 512 → 256  
   *Few → Many*: Создание богатого представления

## Функция потерь

### ✅ Level 1 (Ловушка)
**Pixel-wise MSE** — приводит к копированию входа (Identity Mapping)

### ✅ Level 2 (Распределение)
**Distribution Match** (Moment Matching) — сохраняет статистику цветов (среднее и дисперсию)

### ✅ Level 3 (Геометрия)
- **Smoothness Loss** (Total Variation) — убирает шум, делает переходы плавными
- **Direction Loss** — создаёт градиент яркости слева направо

**Итоговая функция:**  
`Loss = Distribution + Smoothness + Direction`

## Запуск
Откройте `index.html` в браузере.

## Использование
1. Выберите архитектуру
2. Нажмите **Auto Train (Start)**
3. Наблюдайте, как шум превращается в градиент (100-300 шагов)

## Технические детали
- **Фреймворк:** TensorFlow.js
- **Оптимизатор:** Adam (learning rate = 0.02)
- **Вход:** 16×16 случайный шум

## Важное замечание
Sorted MSE из презентации заменён на Moment Matching, так как `tf.topk` в TensorFlow.js не поддерживает вычисление градиентов.

---

# Neural Network Design: The Gradient Puzzle

## Description
A project on training a neural network to transform random noise into a structured gradient without using target labels (unsupervised learning).

## Objective
Develop an autoencoder architecture and loss function that redistributes pixels of 16×16 random noise into a smooth directional gradient.

## Key Constraint
**Cannot create new colors** — can only rearrange existing pixels (analogy: sliding puzzle). The input histogram must match the output histogram.

## Architecture Types

1. **Compression**: 256 → 64 → 256  
   *Many → Few*: Extracting essential features

2. **Transformation**: 256 → 256 → 256  
   *Same → Same*: Re-orientation without information loss

3. **Expansion**: 256 → 512 → 256  
   *Few → Many*: Creating rich representation

## Loss Function

### ✅ Level 1 (The Trap)
**Pixel-wise MSE** — leads to copying input (Identity Mapping)

### ✅ Level 2 (Distribution)
**Distribution Match** (Moment Matching) — preserves color statistics (mean and variance)

### ✅ Level 3 (Geometry)
- **Smoothness Loss** (Total Variation) — removes noise, makes transitions smooth
- **Direction Loss** — creates brightness gradient from left to right

**Combined Loss:**  
`Loss = Distribution + Smoothness + Direction`

## Running
Open `index.html` in a browser.

## Usage
1. Select architecture
2. Click **Auto Train (Start)**
3. Watch noise transform into gradient (100-300 steps)

## Technical Details
- **Framework:** TensorFlow.js
- **Optimizer:** Adam (learning rate = 0.02)
- **Input:** 16×16 random noise

## Important Note
Sorted MSE from the presentation is replaced with Moment Matching because `tf.topk` in TensorFlow.js does not support gradient computation.
