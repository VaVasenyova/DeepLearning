## How the Sigmoid Gate Works (Short Version)

### 🔑 Core Idea
A **learnable gate layer** sits right after input features. During training, it learns a weight (0–1) for each feature:

- **≈1.0** → Feature is critical ("gate open")
- **≈0.0** → Feature is ignored ("gate closed")

These weights multiply the features before they reach the rest of the network.

### 🌟 Two Views of Importance

1. **Global**: Average gate weights across all passengers  
   → Shows which features matter *overall* (e.g., "Sex_female" usually highest)

2. **Per Passenger**: Gate weights for one person  
   → Explains *why* that passenger was predicted to survive/die  
   → Example: For a 1st-class woman, the model heavily used `Sex_female` (0.92) and `Pclass_1` (0.87)

### ✅ Benefits
- **Interpretable**: No black box—see exactly what drives predictions
- **Accurate**: <2% accuracy drop vs. standard model
- **Visual**: Green→yellow gradient shows importance strength
- **Actionable**: Spot data issues (e.g., model relying on passenger ID = leakage)

### 💡 In Practice
After training, Section 7 appears showing:
- A bar chart of global feature importance
- An interactive tool to analyze any passenger's prediction drivers

You get **both** accurate predictions **and** clear explanations—no tradeoff.
