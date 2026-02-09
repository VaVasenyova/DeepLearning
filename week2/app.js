// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    statusDiv.className = 'status';

    try {
        // Load training data using PapaParse
        const trainText = await readFile(trainFile);
        trainData = await parseCSVWithPapaParse(trainText);

        // Load test data using PapaParse
        const testText = await readFile(testFile);
        testData = await parseCSVWithPapaParse(testText);

        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;

        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        statusDiv.className = 'status error';
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

// Parse CSV text to array of objects using PapaParse
function parseCSVWithPapaParse(csvText) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            quotes: true,
            complete: function (results) {
                if (results.errors.length > 0) {
                    console.warn('CSV parsing warnings:', results.errors);
                }
                resolve(results.data);
            },
            error: function (error) {
                reject(error);
            }
        });
    });
}

// Inspect the loaded data
async function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }

    // Wait for PapaParse to finish if needed
    if (trainData instanceof Promise) {
        trainData = await trainData;
    }
    if (testData instanceof Promise) {
        testData = await testData;
    }

    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>'; // ИСПРАВЛЕНО: было "inne rHTML"

    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length; // ИСПРАВЛЕНО: было "= >"
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`; // ИСПРАВЛЕНО: было "lengt h"

    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => { // ИСПРАВЛЕНО: было "= >"
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length; // ИСПРАВЛЕНО: было "= >"
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
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
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => { // ИСПРАВЛЕНО: было "= >"
        if (row.Sex && row.Survived !== undefined) { // ИСПРАВЛЕНО: было "  & &"
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++; // ИСПРАВЛЕНО: было "tota l++"
            if (row.Survived === 1) {
                survivalBySex[row.Sex].survived++;
            }
        }
    });

    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({ // ИСПРАВЛЕНО: было "= >"
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })), // ИСПРАВЛЕНО: было "= >"
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)', height: 300 }
    );

    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => { // ИСПРАВЛЕНО: было "= >"
        if (row.Pclass !== undefined && row.Survived !== undefined) { // ИСПРАВЛЕНО: было "  & &"
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++; // ИСПРАВЛЕНО: было "survivalByPclas s"
            if (row.Survived === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });

    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({ // ИСПРАВЛЕНО: было "= >" и "survivalByPclas s"
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts' }, // ИСПРАВЛЕНО: было "Charts "
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })), // ИСПРАВЛЕНО: было "= >"
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)', height: 300 }
    );

    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Preprocess the data
async function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }

    // Wait for PapaParse to finish if needed
    if (trainData instanceof Promise) {
        trainData = await trainData;
    }
    if (testData instanceof Promise) {
        testData = await testData;
    }

    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    outputDiv.className = 'status';

    try {
        // Calculate imputation values from training data
        const ageValues = trainData.map(row => row.Age).filter(age => age !== null && age !== undefined); // ИСПРАВЛЕНО: было "= >"
        const ageMedian = calculateMedian(ageValues);

        const fareValues = trainData.map(row => row.Fare).filter(fare => fare !== null && fare !== undefined); // ИСПРАВЛЕНО: было "= >"
        const fareMedian = calculateMedian(fareValues);

        const embarkedValues = trainData.map(row => row.Embarked).filter(e => e !== null && e !== undefined); // ИСПРАВЛЕНО: было "= >"
        const embarkedMode = calculateMode(embarkedValues);

        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };

        trainData.forEach(row => { // ИСПРАВЛЕНО: было "= >"
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]); // ИСПРАВЛЕНО: было "labels. push"
        });

        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };

        testData.forEach(row => { // ИСПРАВЛЕНО: было "= >" и "row  = >"
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]); // ИСПРАВЛЕНО: было "passenger Ids"
        });

        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels); // ИСПРАВЛЕНО: было "labels  ="

        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
        `;
        outputDiv.className = 'status';

        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        outputDiv.className = 'status error';
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null && row.Age !== undefined ? row.Age : ageMedian;
    const fare = row.Fare !== null && row.Fare !== undefined ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null && row.Embarked !== undefined ? row.Embarked : embarkedMode;

    // Calculate std dev for standardization
    const ageValues = trainData.map(r => r.Age).filter(a => a !== null && a !== undefined);
    const ageStd = calculateStdDev(ageValues) || 1;

    const fareValues = trainData.map(r => r.Fare).filter(f => f !== null && f !== undefined);
    const fareStd = calculateStdDev(fareValues) || 1;

    // Standardize numerical features
    const standardizedAge = (age - ageMedian) / ageStd;
    const standardizedFare = (fare - fareMedian) / fareStd;

    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass values: 1, 2, 3
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);

    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];

    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);

    // Add optional family features if enabled
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

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    const inputShape = preprocessedTrainData.features.shape[1];

    // Create a sequential model
    model = tf.sequential();

    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu', // ИСПРАВЛЕНО: было "act ivation"
        inputShape: [inputShape]
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    // Compile the model
    model.compile({
        optimizer: 'adam', // ИСПРАВЛЕНО: было "a dam"
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';

    // Simple summary since tfjs doesn't have a built-in summary function for the browser
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => { // ИСПРАВЛЕНО: было "= >"
        summaryText += `<li>Layer ${i + 1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;

    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }

    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    statusDiv.className = 'status';

    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);

        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex); // ИСПРАВЛЕНО: было "trainL abels"

        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex); // ИСПРАВЛЕНО: было "trainDat a.labels"

        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;

        // Train the model - ИСПРАВЛЕНО: убран конфликтующий callbacks
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                }
            }
        });

        statusDiv.innerHTML += '<p>Training completed!</p>';
        statusDiv.className = 'status';

        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);

        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);

        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;

        // Calculate initial metrics
        await updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        statusDiv.className = 'status error';
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;

    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    // Calculate confusion matrix - ИСПРАВЛЕНО: используем await .array()
    const predVals = await validationPredictions.array();
    const trueVals = await validationLabels.array();

    let tp = 0, tn = 0, fp = 0, fn = 0;

    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];

        if (prediction === 1 && actual === 1) tp++; // ИСПРАВЛЕНО: было "  & &"
        else if (prediction === 0 && actual === 0) tn++; // ИСПРАВЛЕНО: было "  & &"
        else if (prediction === 1 && actual === 0) fp++; // ИСПРАВЛЕНО: было "  & &"
        else if (prediction === 0 && actual === 1) fn++;
    }

    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <h3>Confusion Matrix</h3>
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0; // ИСПРАВЛЕНО: было "c onst"

    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics'); // ИСПРАВЛЕНО: было "metricsDiv.innerHTML  ="
    metricsDiv.innerHTML = `
        <h3>Performance Metrics</h3>
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;

    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals);
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

        rocData.push({ x: fpr, y: tpr }); // ИСПРАВЛЕНО: правильный формат для tfvis
    });

    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].x - rocData[i - 1].x) * (rocData[i].y + rocData[i - 1].y) / 2;
    }
    auc = Math.abs(auc); // Ensure positive

    // Plot ROC curve - ИСПРАВЛЕНО: правильный формат данных
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: [{ label: 'ROC Curve', values: rocData }] },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            height: 300,
            width: 400
        }
    );

    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    outputDiv.className = 'status';

    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);

        // Make predictions
        testPredictions = model.predict(testFeatures);

        // ИСПРАВЛЕНО: используем await .array() и извлекаем значения из вложенных массивов
        const predValues = await testPredictions.array();

        // Создание результатов с правильным извлечением значений
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i][0] >= 0.5 ? 1 : 0,  // predValues[i][0] вместо predValues[i]
            Probability: predValues[i][0]                // predValues[i][0] вместо predValues[i]
        }));

        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.className = 'status';

        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;

        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        outputDiv.className = 'status error';
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
        ['PassengerId', 'Survived', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            td.textContent = key === 'Probability' ? row[key].toFixed(4) : row[key];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }

    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    statusDiv.className = 'status';

    try {
        // Get predictions
        // ИСПРАВЛЕНО: используем await .array() и извлекаем значения
        const predValues = await testPredictions.array();

        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            // ИСПРАВЛЕНО: используем predValues[i][0]
            submissionCSV += `${id},${predValues[i][0] >= 0.5 ? 1 : 0}\n`;
        });

        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            // ИСПРАВЛЕНО: используем predValues[i][0]
            probabilitiesCSV += `${id},${predValues[i][0].toFixed(6)}\n`;
        });

        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'submission.csv';

        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'probabilities.csv';

        // Trigger downloads
        submissionLink.click();
        probabilitiesLink.click();

        // Save model
        await model.save('downloads://titanic-tfjs-model');

        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: submission.csv (Kaggle submission format)</p>
            <p>Downloaded: probabilities.csv (Prediction probabilities)</p>
            <p>Model saved to browser downloads</p>
        `;
        statusDiv.className = 'status';
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        statusDiv.className = 'status error';
        console.error(error);
    }
}