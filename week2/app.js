// ========================================
// TITANIC SURVIVAL CLASSIFIER - TensorFlow.js
// WITH SIGMOID GATE FOR FEATURE IMPORTANCE
// ========================================

// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let gateModel = null; // Sub-model for extracting gate activations
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let featureNames = [];

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// ========================================
// DATA LOADING FUNCTIONS
// ========================================

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = '<p>Loading data...</p>';
    statusDiv.className = 'status';

    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);

        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);

        statusDiv.innerHTML = `<p class="success">✓ Data loaded successfully!</p>
            <p>Training samples: ${trainData.length}</p>
            <p>Test samples: ${testData.length}</p>
            <p>Features: ${Object.keys(trainData[0]).join(', ')}</p>`;

        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `<p class="error">Error loading data: ${error.message}</p>`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];

    // Parse header
    const headers = parseCSVLine(lines[0]);

    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};

        headers.forEach((header, i) => {
            let value = values[i] !== undefined ? values[i] : null;

            // Trim quotes and whitespace
            if (typeof value === 'string') {
                value = value.trim().replace(/^"+|"+$/g, '');
            }

            // Convert to number if possible
            if (value !== null && value !== '' && !isNaN(Number(value))) {
                obj[header] = Number(value);
            } else {
                obj[header] = value === '' ? null : value;
            }
        });

        return obj;
    });
}

// Properly parse a single CSV line handling quoted fields
function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    let i = 0;

    while (i < line.length) {
        const char = line[i];

        if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
                // Escaped quote
                current += '"';
                i += 2;
                continue;
            }
            inQuotes = !inQuotes;
            i++;
        } else if (char === ',' && !inQuotes) {
            values.push(current);
            current = '';
            i++;
        } else {
            current += char;
            i++;
        }
    }

    values.push(current);
    return values;
}

// ========================================
// DATA INSPECTION FUNCTIONS
// ========================================

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }

    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';

    const shapeInfo = `Dataset shape: ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    // Calculate missing values
    let missingInfo = '<h4>Missing Values:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(1);
        if (missingCount > 0) {
            missingInfo += `<li>${feature}: ${missingCount} (${missingPercent}%)</li>`;
        }
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;

    // Create visualizations
    createVisualizations();

    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');

    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null && value !== undefined ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

// Create visualizations using tfjs-vis
// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) {
                survivalBySex[row.Sex].survived++;
            }
        }
    });

    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex: sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
        { 
            xLabel: 'Sex', 
            yLabel: 'Survival Rate (%)',
            height: 300
        }
    );

    console.log('✓ Chart created: Survival Rate by Sex');
    console.log('  Data points:', sexData.length);
    console.log('  Values:', sexData.map(d => `${d.sex}: ${d.survivalRate.toFixed(2)}%`).join(', '));

    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Survived !== undefined) {
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++;
            if (row.Survived === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });

    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
        { 
            xLabel: 'Passenger Class', 
            yLabel: 'Survival Rate (%)',
            height: 300
        }
    );

    console.log('✓ Chart created: Survival Rate by Passenger Class');
    console.log('  Data points:', pclassData.length);
    console.log('  Values:', pclassData.map(d => `${d.pclass}: ${d.survivalRate.toFixed(2)}%`).join(', '));

    chartsDiv.innerHTML += '<p>Interactive charts are displayed in the tfjs-vis visor (bottom-right corner).</p>';
    console.log('✓ Data visualizations section updated in DOM');
}

// Display global feature importance with horizontal bar chart
function displayGlobalFeatureImportance(importanceArray) {
    const importanceDiv = document.getElementById('global-importance');
    if (!importanceDiv) return;

    // Normalize importance for color gradient (0 = yellow, 1 = green)
    const maxImportance = Math.max(...importanceArray.map(d => d.importance));
    
    // Create colored bars with gradient for tfvis
    const barData = importanceArray.map(d => {
        const normalized = d.importance / maxImportance;
        // Green (0, 128, 0) to Yellow (255, 255, 0) gradient
        const r = Math.floor(255 * normalized);
        const g = Math.floor(128 + 127 * normalized);
        const b = 0;
        const color = `rgb(${r}, ${g}, ${b})`;
        
        return {
            x: d.feature,
            y: d.importance,
            color: color
        };
    });

    // Render with tfvis
    tfvis.render.barchart(
        { name: 'Global Feature Importance', tab: 'Feature Importance' },
        barData,
        {
            xLabel: 'Feature',
            yLabel: 'Mean Gate Activation (0-1)',
            title: 'Which Features Matter Most?',
            height: 400,
            fontSize: 10
        }
    );

    console.log('✓ Chart created: Global Feature Importance');
    console.log('  Total features:', importanceArray.length);
    console.log('  Top 3 features:', importanceArray.slice(0, 3).map(f => `${f.feature}: ${f.importance.toFixed(3)}`).join(', '));
    console.log('  Feature range: min=' + Math.min(...importanceArray.map(f => f.importance)).toFixed(3) + 
                ', max=' + Math.max(...importanceArray.map(f => f.importance)).toFixed(3));

    // Also display as HTML table for accessibility
    const tableHTML = `
        <h3>Global Feature Importance</h3>
        <p><strong>Interpretation:</strong> Gate activation close to 1 means the feature strongly influences predictions. 
        Close to 0 means the feature is largely ignored by the model.</p>
        <table class="importance-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance Score</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
                ${importanceArray.map((item, idx) => {
                    const normalized = item.importance;
                    const r = Math.floor(255 * normalized);
                    const g = Math.floor(128 + 127 * normalized);
                    const b = 0;
                    const color = `rgb(${r}, ${g}, ${b})`;
                    
                    const impact = item.importance > 0.7 ? 'High' : 
                                   item.importance > 0.4 ? 'Medium' : 'Low';
                    
                    return `
                        <tr>
                            <td>${idx + 1}</td>
                            <td><strong>${item.feature}</strong></td>
                            <td style="background-color: ${color}; color: white; font-weight: bold;">
                                ${item.importance.toFixed(3)}
                            </td>
                            <td>${impact}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
        <p><strong>Note:</strong> The sigmoid gate learns to "turn on" (≈1) important features and "turn off" (≈0) irrelevant ones.</p>
    `;

    importanceDiv.innerHTML = tableHTML;
    console.log('✓ Global feature importance table rendered in DOM');
}

// ========================================
// PREPROCESSING FUNCTIONS
// ========================================

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }

    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = '<p>Preprocessing data...</p>';
    outputDiv.className = 'status';

    try {
        // Calculate imputation values from training data
        const ageValues = trainData.map(row => row.Age).filter(age => age !== null && !isNaN(age));
        const ageMedian = calculateMedian(ageValues);

        const fareValues = trainData.map(row => row.Fare).filter(fare => fare !== null && !isNaN(fare));
        const fareMedian = calculateMedian(fareValues);

        const embarkedValues = trainData.map(row => row.Embarked).filter(e => e !== null);
        const embarkedMode = calculateMode(embarkedValues);

        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };

        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });

        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };

        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

        outputDiv.innerHTML = `<p class="success">✓ Preprocessing completed!</p>
            <p><strong>Training data:</strong></p>
            <p>• Features shape: ${preprocessedTrainData.features.shape}</p>
            <p>• Labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p><strong>Test data:</strong></p>
            <p>• Features: ${preprocessedTestData.features.length} samples × ${preprocessedTestData.features[0].length} features</p>
            <p>• Passenger IDs: ${preprocessedTestData.passengerIds.length}</p>`;

        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `<p class="error">Error during preprocessing: ${error.message}</p>`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null && !isNaN(row.Age) ? row.Age : ageMedian;
    const fare = row.Fare !== null && !isNaN(row.Fare) ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;

    // Calculate statistics for normalization
    const ageValues = trainData.map(r => r.Age).filter(a => a !== null && !isNaN(a));
    const ageMean = ageValues.reduce((sum, val) => sum + val, 0) / ageValues.length;
    const ageStd = Math.sqrt(ageValues.reduce((sum, val) => sum + Math.pow(val - ageMean, 2), 0) / ageValues.length) || 1;

    const fareValues = trainData.map(r => r.Fare).filter(f => f !== null && !isNaN(f));
    const fareMean = fareValues.reduce((sum, val) => sum + val, 0) / fareValues.length;
    const fareStd = Math.sqrt(fareValues.reduce((sum, val) => sum + Math.pow(val - fareMean, 2), 0) / fareValues.length) || 1;

    // Standardize numerical features
    const standardizedAge = (age - ageMean) / ageStd;
    const standardizedFare = (fare - fareMean) / fareStd;

    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);

    // Build feature vector
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];

    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);

    // Add optional family features
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }

    return features;
}

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[half - 1] + sorted[half]) / 2;
    }
    return sorted[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (values.length === 0) return null;
    const frequency = {};
    let maxCount = 0;
    let mode = null;

    values.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });

    return mode;
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    return categories.map(category => category === value ? 1 : 0);
}

// ========================================
// MODEL CREATION WITH SIGMOID GATE
// ========================================

// Create the model with sigmoid gating layer
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    const inputShape = preprocessedTrainData.features.shape[1];

    // Store feature names for reference
    featureNames = [
        'Age', 'Fare', 'SibSp', 'Parch',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_male', 'Sex_female',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ];

    if (document.getElementById('add-family-features').checked) {
        featureNames.push('FamilySize', 'IsAlone');
    }

    // Create model using Functional API with sigmoid gate
    const input = tf.input({ shape: [inputShape] });

    // SIGMOID GATING LAYER: Learnable weights for each feature (0-1 range)
    // L1 regularization encourages sparsity - gates will be close to 0 or 1
    const gate = tf.layers.dense({
        units: inputShape,
        activation: 'sigmoid',
        kernel_regularizer: tf.regularizers.l1({ l1: 0.01 }), // Encourage sparse selection
        name: 'feature_gate'
    }).apply(input);

    // Multiply gate weights element-wise with input features
    const gatedFeatures = tf.layers.multiply().apply([input, gate]);

    // Hidden layer
    const hidden = tf.layers.dense({
        units: 16,
        activation: 'relu'
    }).apply(gatedFeatures);

    // Output layer
    const output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }).apply(hidden);

    // Create model
    model = tf.model({ inputs: input, outputs: output });

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary (with Sigmoid Gate)</h3>';

    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li><strong>Layer ${i + 1}:</strong> ${layer.name} (${layer.getClassName()}) - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p><strong>Total parameters:</strong> ${model.countParams()}</p>`;
    summaryText += `<p><strong>Input shape:</strong> [${inputShape}]</p>`;
    summaryText += `<p><strong>Output:</strong> Binary classification (survival probability)</p>`;
    summaryText += `<p><strong>Sigmoid Gate:</strong> Learnable importance weights (0-1) for each feature with L1 regularization</p>`;
    summaryText += `<p><strong>Feature Names:</strong> ${featureNames.join(', ')}</p>`;
    summaryDiv.innerHTML += summaryText;

    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// ========================================
// TRAINING FUNCTIONS
// ========================================

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }

    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = '<p>Training model... (this may take 30-60 seconds)</p>';
    statusDiv.className = 'status';

    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);

        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);

        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);

        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;

        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                {
                    callbacks: ['onEpochEnd'],
                    height: 400
                }
            )
        });

        statusDiv.innerHTML += '<p class="success">✓ Training completed!</p>';
        statusDiv.innerHTML += `<p><strong>Final training accuracy:</strong> ${(trainingHistory.history.acc[trainingHistory.history.acc.length - 1] * 100).toFixed(2)}%</p>`;
        statusDiv.innerHTML += `<p><strong>Final validation accuracy:</strong> ${(trainingHistory.history.val_acc[trainingHistory.history.val_acc.length - 1] * 100).toFixed(2)}%</p>`;

        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);

        // Analyze feature importance
        statusDiv.innerHTML += '<p>Analyzing feature importance...</p>';
        await analyzeFeatureImportance();

        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);

        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;

        // Calculate initial metrics
        await updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `<p class="error">Error during training: ${error.message}</p>`;
        console.error(error);
    }
}

// ========================================
// FEATURE IMPORTANCE ANALYSIS
// ========================================

// Analyze and visualize feature importance using the sigmoid gate
async function analyzeFeatureImportance() {
    if (!model || !preprocessedTrainData) {
        console.warn('Model or data not available for feature importance analysis');
        return;
    }

    try {
        // Extract gate layer
        const gateLayer = model.getLayer('feature_gate');
        if (!gateLayer) {
            console.warn('Gate layer not found');
            return;
        }

        // Create sub-model to get gate activations
        gateModel = tf.model({
            inputs: model.inputs[0],
            outputs: gateLayer.output
        });

        // Get gate activations for all training samples
        const gateActivations = gateModel.predict(preprocessedTrainData.features);
        const gateValues = await gateActivations.array();

        // Calculate GLOBAL feature importance (mean gate activation across all samples)
        const globalImportance = [];
        for (let i = 0; i < featureNames.length; i++) {
            const meanGate = gateValues.reduce((sum, sample) => sum + sample[i], 0) / gateValues.length;
            globalImportance.push({
                feature: featureNames[i],
                importance: meanGate,
                index: i
            });
        }

        // Sort by importance (descending)
        globalImportance.sort((a, b) => b.importance - a.importance);

        // Display global importance
        displayGlobalFeatureImportance(globalImportance);

        // Show the feature importance section
        document.getElementById('feature-importance').style.display = 'block';

        // Set max passenger index
        const passengerSelect = document.getElementById('passenger-select');
        if (passengerSelect) {
            passengerSelect.max = preprocessedTrainData.features.shape[0] - 1;
        }

        console.log('Feature importance analysis completed');
    } catch (error) {
        console.error('Error in feature importance analysis:', error);
    }
}

// Display global feature importance with horizontal bar chart
function displayGlobalFeatureImportance(importanceArray) {
    const importanceDiv = document.getElementById('global-importance');
    if (!importanceDiv) return;

    // Normalize importance for color gradient (0 = yellow, 1 = green)
    const maxImportance = Math.max(...importanceArray.map(d => d.importance));

    // Create colored bars with gradient for tfvis
    const barData = importanceArray.map(d => {
        const normalized = d.importance / maxImportance;
        // Green (0, 128, 0) to Yellow (255, 255, 0) gradient
        const r = Math.floor(255 * normalized);
        const g = Math.floor(128 + 127 * normalized);
        const b = 0;
        const color = `rgb(${r}, ${g}, ${b})`;

        return {
            x: d.feature,
            y: d.importance,
            color: color
        };
    });

    // Render with tfvis
    tfvis.render.barchart(
        { name: 'Global Feature Importance', tab: 'Feature Importance' },
        barData,
        {
            xLabel: 'Feature',
            yLabel: 'Mean Gate Activation (0-1)',
            title: 'Which Features Matter Most?',
            height: 400,
            fontSize: 10
        }
    );

    // Also display as HTML table for accessibility
    const tableHTML = `
        <h3>Global Feature Importance</h3>
        <p><strong>Interpretation:</strong> Gate activation close to 1 means the feature strongly influences predictions. 
        Close to 0 means the feature is largely ignored by the model.</p>
        <table class="importance-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance Score</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
                ${importanceArray.map((item, idx) => {
        const normalized = item.importance;
        const r = Math.floor(255 * normalized);
        const g = Math.floor(128 + 127 * normalized);
        const b = 0;
        const color = `rgb(${r}, ${g}, ${b})`;

        const impact = item.importance > 0.7 ? 'High' :
            item.importance > 0.4 ? 'Medium' : 'Low';

        return `
                        <tr>
                            <td>${idx + 1}</td>
                            <td><strong>${item.feature}</strong></td>
                            <td style="background-color: ${color}; color: white; font-weight: bold;">
                                ${item.importance.toFixed(3)}
                            </td>
                            <td>${impact}</td>
                        </tr>
                    `;
    }).join('')}
            </tbody>
        </table>
        <p><strong>Note:</strong> The sigmoid gate learns to "turn on" (≈1) important features and "turn off" (≈0) irrelevant ones.</p>
    `;

    importanceDiv.innerHTML = tableHTML;
}

// Analyze feature importance for selected passenger
async function analyzeSelectedPassenger() {
    const passengerIdx = parseInt(document.getElementById('passenger-select').value);
    if (isNaN(passengerIdx) || passengerIdx < 0 || !gateModel) {
        alert('Please enter a valid passenger index.');
        return;
    }
    await analyzeSinglePassenger(gateModel, passengerIdx);
}

// Analyze random passenger
async function analyzeRandomPassenger() {
    if (!preprocessedTrainData || !gateModel) {
        alert('Model not ready for analysis.');
        return;
    }
    const maxIdx = preprocessedTrainData.features.shape[0] - 1;
    const randomIdx = Math.floor(Math.random() * (maxIdx + 1));
    document.getElementById('passenger-select').value = randomIdx;
    await analyzeSinglePassenger(gateModel, randomIdx);
}

// Analyze feature importance for a single passenger
async function analyzeSinglePassenger(gateModel, passengerIdx) {
    if (!preprocessedTrainData || passengerIdx >= preprocessedTrainData.features.shape[0]) {
        alert('Invalid passenger index');
        return;
    }

    const analysisDiv = document.getElementById('passenger-analysis');
    analysisDiv.innerHTML = '<p>Analyzing passenger...</p>';

    try {
        // Get single passenger features
        const passengerFeatures = preprocessedTrainData.features.slice([passengerIdx, 0], [1, -1]);

        // Get original data row for context
        const originalRow = trainData[passengerIdx];

        // Get gate activations for this passenger
        const gateOutput = gateModel.predict(passengerFeatures);
        const gateValues = await gateOutput.array();

        // Get model prediction
        const prediction = model.predict(passengerFeatures);
        const predValue = await prediction.array();

        // Create importance array for this passenger
        const passengerImportance = featureNames.map((feature, idx) => ({
            feature: feature,
            importance: gateValues[0][idx],
            rawValue: passengerFeatures.dataSync()[idx]
        }));

        // Sort by importance
        passengerImportance.sort((a, b) => b.importance - a.importance);

        // Display analysis
        const survivalStatus = originalRow.Survived === 1 ? '✓ Survived' : '✗ Did Not Survive';
        const predictedStatus = predValue[0][0] >= 0.5 ? 'Predicted: Survive' : 'Predicted: Not Survive';
        const confidence = Math.max(predValue[0][0], 1 - predValue[0][0]) * 100;

        let analysisHTML = `
            <div style="border: 2px solid #333; padding: 20px; border-radius: 10px; background: #f9f9f9;">
                <h4>Passenger Analysis (Index: ${passengerIdx})</h4>
                <p><strong>Actual:</strong> ${survivalStatus} | <strong>${predictedStatus}</strong> (${confidence.toFixed(1)}% confidence)</p>
                <p><strong>Passenger Context:</strong> ${originalRow.Sex || 'N/A'}, Class ${originalRow.Pclass || 'N/A'}, Age ${originalRow.Age || 'N/A'}</p>
                
                <h5>Top Influential Features for This Prediction:</h5>
                <table class="importance-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature</th>
                            <th>Gate Activation</th>
                            <th>Passenger Value</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        passengerImportance.slice(0, 6).forEach((item, idx) => {
            const normalized = item.importance;
            const r = Math.floor(255 * normalized);
            const g = Math.floor(128 + 127 * normalized);
            const b = 0;
            const color = `rgb(${r}, ${g}, ${b})`;

            let interpretation = '';
            if (item.importance > 0.8) {
                interpretation = '⭐ Critical factor';
            } else if (item.importance > 0.6) {
                interpretation = '✓ Strong influence';
            } else if (item.importance > 0.4) {
                interpretation = '→ Moderate influence';
            } else {
                interpretation = '○ Weak influence';
            }

            analysisHTML += `
                <tr>
                    <td>${idx + 1}</td>
                    <td><strong>${item.feature}</strong></td>
                    <td style="background-color: ${color}; color: white; font-weight: bold;">
                        ${item.importance.toFixed(3)}
                    </td>
                    <td>${item.rawValue.toFixed(3)}</td>
                    <td>${interpretation}</td>
                </tr>
            `;
        });

        analysisHTML += `
                    </tbody>
                </table>
                <p style="font-size: 0.9em; margin-top: 15px;">
                    <strong>How to read this:</strong> High gate activation (dark green) means this feature strongly influenced the prediction for this specific passenger. 
                    For example, if "Sex_female" has high activation for a female passenger who survived, the model heavily relied on gender for this prediction.
                </p>
            </div>
        `;

        analysisDiv.innerHTML = analysisHTML;
    } catch (error) {
        analysisDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        console.error(error);
    }
}

// ========================================
// EVALUATION FUNCTIONS
// ========================================

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;

    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    // Calculate confusion matrix
    const predVals = await validationPredictions.array();
    const trueVals = await validationLabels.array();

    let tp = 0, tn = 0, fp = 0, fn = 0;

    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i][0] >= threshold ? 1 : 0;
        const actual = trueVals[i];

        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }

    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr>
                <th></th>
                <th><strong>Predicted Positive</strong></th>
                <th><strong>Predicted Negative</strong></th>
            </tr>
            <tr>
                <th><strong>Actual Positive</strong></th>
                <td style="background-color: #d4edda; color: #155724;">${tp}</td>
                <td style="background-color: #f8d7da; color: #721c24;">${fn}</td>
            </tr>
            <tr>
                <th><strong>Actual Negative</strong></th>
                <td style="background-color: #f8d7da; color: #721c24;">${fp}</td>
                <td style="background-color: #d4edda; color: #155724;">${tn}</td>
            </tr>
        </table>
        <p><strong>TP:</strong> ${tp} | <strong>TN:</strong> ${tn} | <strong>FP:</strong> ${fp} | <strong>FN:</strong> ${fn}</p>
    `;

    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;

    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
        <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
    `;

    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals.map(p => p[0]));
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocData = [];

    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;

        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];

            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }

        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;

        rocData.push({ threshold, fpr, tpr });
    });

    // Calculate AUC (trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i - 1].fpr) * (rocData[i].tpr + rocData[i - 1].tpr) / 2;
    }
    auc = Math.abs(auc);

    // Plot ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            title: `ROC Curve (AUC: ${auc.toFixed(4)})`,
            width: 400,
            height: 400
        }
    );

    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
}

// ========================================
// PREDICTION FUNCTIONS
// ========================================

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = '<p>Making predictions on test data...</p>';
    outputDiv.className = 'status';

    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);

        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = await testPredictions.array();

        // Create prediction results
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i][0] >= 0.5 ? 1 : 0,
            Probability: predValues[i][0]
        }));

        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Samples)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));

        outputDiv.innerHTML += `<p class="success">✓ Predictions completed!</p>`;
        outputDiv.innerHTML += `<p><strong>Total predictions:</strong> ${results.length}</p>`;
        outputDiv.innerHTML += `<p><strong>Predicted survivors:</strong> ${results.filter(r => r.Survived === 1).length}</p>`;
        outputDiv.innerHTML += `<p><strong>Predicted non-survivors:</strong> ${results.filter(r => r.Survived === 0).length}</p>`;

        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `<p class="error">Error during prediction: ${error.message}</p>`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');

    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.PassengerId}</td>
            <td style="background-color: ${row.Survived === 1 ? '#d4edda' : '#f8d7da'};">
                ${row.Survived === 1 ? '✓ Yes' : '✗ No'}
            </td>
            <td>${row.Probability.toFixed(4)}</td>
        `;
        table.appendChild(tr);
    });

    return table;
}

// ========================================
// EXPORT FUNCTIONS
// ========================================

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }

    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = '<p>Exporting results...</p>';
    statusDiv.className = 'status';

    try {
        // Get predictions
        const predValues = await testPredictions.array();

        // Create submission CSV (PassengerId, Survived) - Kaggle format
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i][0] >= 0.5 ? 1 : 0}\n`;
        });

        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predValues[i][0].toFixed(6)}\n`;
        });

        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'titanic_submission.csv';

        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'titanic_probabilities.csv';

        // Trigger downloads
        submissionLink.click();
        probabilitiesLink.click();

        // Clean up
        setTimeout(() => {
            URL.revokeObjectURL(submissionLink.href);
            URL.revokeObjectURL(probabilitiesLink.href);
        }, 100);

        statusDiv.innerHTML = `<p class="success">✓ Export completed!</p>
            <p><strong>Downloaded files:</strong></p>
            <ul>
                <li><code>titanic_submission.csv</code> - Kaggle submission format (PassengerId, Survived)</li>
                <li><code>titanic_probabilities.csv</code> - Prediction probabilities</li>
            </ul>
            <p><strong>Next steps:</strong> Upload <code>titanic_submission.csv</code> to Kaggle to see your score!</p>`;

    } catch (error) {
        statusDiv.innerHTML = `<p class="error">Error during export: ${error.message}</p>`;
        console.error(error);
    }
}

// ========================================
// INITIALIZATION
// ========================================

// Initialize button states when page loads
window.addEventListener('load', () => {
    console.log('Titanic Survival Classifier with Sigmoid Gate initialized');

    // All buttons start disabled except load-data-btn
    const buttonsToDisable = [
        'inspect-btn',
        'preprocess-btn',
        'create-model-btn',
        'train-btn',
        'predict-btn',
        'export-btn'
    ];

    buttonsToDisable.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = true;
    });

    // Threshold slider starts disabled
    const slider = document.getElementById('threshold-slider');
    if (slider) slider.disabled = true;
});
