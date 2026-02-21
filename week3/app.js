/**
Neural Network Design: The Gradient Puzzle
Objective:
Modify the Student Model architecture and loss function to transform
random noise input into a smooth, directional gradient output.
*/

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  // Model definition shape (no batch dim)
  inputShapeModel: [16, 16, 1],
  // Data tensor shape (includes batch dim)
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.02, // Reduced for stability
  autoTrainSpeed: 50,
  // [FIX] Balanced loss weights to prevent saturation (black/white)
  lossWeights: {
    distribution: 10.0, // High weight to conserve colors
    smoothness: 0.5,    // Moderate weight for smoothness
    direction: 2.0,     // Moderate weight for gradient direction
  },
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  // [FIX] Separate optimizers to prevent shape conflicts
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE for baseline
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// [FIX] Distribution Constraint - Match Mean and Variance
// This ensures "Input Histogram ~ Output Histogram" without fixing positions
function distributionMatch(yTrue, yPred) {
  return tf.tidy(() => {
    // Match Mean (overall brightness)
    const meanTrue = tf.mean(yTrue);
    const meanPred = tf.mean(yPred);
    const lossMean = tf.square(meanTrue.sub(meanPred));

    // Match Variance (color diversity/contrast)
    const momentsTrue = tf.moments(yTrue);
    const momentsPred = tf.moments(yPred);
    const lossVar = tf.square(momentsTrue.variance.sub(momentsPred.variance));

    return lossMean.add(lossVar);
  });
}

// Smoothness (Total Variation Loss)
// Penalizes differences between adjacent pixels
function smoothness(yPred) {
  return tf.tidy(() => {
    // X direction differences
    const diffX = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

    // Y direction differences
    const diffY = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

// Directionality (Gradient Left to Right)
// [FIX] Adjusted mask to prevent extreme saturation
function directionX(yPred) {
  return tf.tidy(() => {
    const width = 16;
    // Create mask from -0.5 to 0.5 (gentler gradient pressure)
    const mask = tf.linspace(-0.5, 0.5, width).reshape([1, 1, width, 1]);
    // Maximize correlation: minimize negative correlation
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// [FIX] Student Model - All architectures implemented
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Compression: 256 -> 64 -> 256 (Many -> Few)
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // [FIXED] Transformation: 256 -> 256 -> 256 (Same -> Same)
    // Best for this task - no information loss during rearrangement
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // [FIXED] Expansion: 256 -> 512 -> 256 (Few -> Many)
    // Overcomplete representation
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// [FIX] Student Loss - Balanced components to prevent black/white saturation
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // CRITICAL: Do NOT use pixel-wise MSE(yTrue, yPred)
    // This causes Identity Mapping (Level 1 Trap)

    // Level 2: Distribution Constraint
    // "Conserve the inventory of colors" - Input Histogram ~ Output Histogram
    const lossDist = distributionMatch(yTrue, yPred).mul(CONFIG.lossWeights.distribution);

    // Level 3: Smoothness
    // "Be smooth locally" - Remove jagged noise
    const lossSmooth = smoothness(yPred).mul(CONFIG.lossWeights.smoothness);

    // Level 3: Direction
    // "Be bright on the right" - Create gradient pattern
    const lossDir = directionX(yPred).mul(CONFIG.lossWeights.direction);

    // Total Loss
    // "Distribution matching conserves colors, Smoothness/Direction guides them"
    const totalLoss = lossDist.add(lossSmooth).add(lossDir);

    return totalLoss;
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());
    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());
      state.studentOptimizer.applyGradients(grads);
      return value.dataSync()[0];
    });

    log(
      `Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`,
    );
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Recommended: Transformation architecture.");
  log("Loss: Distribution + Smoothness + Direction = Gradient!");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    archType = null;
  }

  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  // Dispose old resources
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  // [FIX] Dispose BOTH optimizers
  if (state.baselineOptimizer) {
    state.baselineOptimizer.dispose();
    state.baselineOptimizer = null;
  }
  if (state.studentOptimizer) {
    state.studentOptimizer.dispose();
    state.studentOptimizer = null;
  }

  // Create new models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error: ${e.message}`, true);
    state.studentModel = createBaselineModel();
  }

  // [FIX] Create SEPARATE optimizers
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  log(`Reset. Architecture: ${archType}`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline"),
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start
init();
