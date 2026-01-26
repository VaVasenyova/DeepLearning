// Global variables
let mergedData = null;
let trainData = null;
let testData = null;
let summaryReport = "";

// Utility: download text/blob
function downloadBlob(content, filename, contentType = "text/plain") {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Parse CSV files
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (error) => reject(error)
        });
    });
}

// Add source column
function addSourceColumn(data, source) {
    return data.map(row => ({ ...row, source }));
}

// Format percentage
function pct(value, total) {
    return ((value / total) * 100).toFixed(2) + "%";
}

// Main load handler
document.getElementById("loadDataBtn").addEventListener("click", async () => {
    const trainFile = document.getElementById("trainFile").files[0];
    const testFile = document.getElementById("testFile").files[0];

    if (!trainFile || !testFile) {
        alert("Please upload both train.csv and test.csv");
        return;
    }

    try {
        trainData = await parseCSV(trainFile);
        testData = await parseCSV(testFile);

        const trainWithSource = addSourceColumn(trainData, "train");
        const testWithSource = addSourceColumn(testData, "test");

        mergedData = [...trainWithSource, ...testWithSource];

        summaryReport = "";
        document.getElementById("overviewContent").innerHTML = "";
        document.getElementById("missingContent").innerHTML = "";
        document.getElementById("statsContent").innerHTML = "";
        document.getElementById("vizContent").innerHTML = "";
        document.getElementById("exportStatus").innerHTML = "<p style='color:#a0e75a'>✅ Data loaded successfully!</p>";
    } catch (err) {
        console.error(err);
        document.getElementById("exportStatus").innerHTML = `<p style='color:red'>❌ Error loading data: ${err.message}</p>`;
    }
});

// Show Overview
document.getElementById("showOverviewBtn").addEventListener("click", () => {
    if (!mergedData) {
        alert("Please load data first.");
        return;
    }

    const rows = mergedData.length;
    const cols = Object.keys(mergedData[0]).length;
    const headers = Object.keys(mergedData[0]);

    let html = `<p><strong>Shape:</strong> ${rows} rows × ${cols} columns</p>`;
    html += `<p><strong>Columns:</strong> ${headers.join(", ")}</p>`;
    html += "<h3>First 5 Rows:</h3><table><thead><tr>";
    headers.forEach(h => html += `<th>${h}</th>`);
    html += "</tr></thead><tbody>";
    mergedData.slice(0, 5).forEach(row => {
        html += "<tr>";
        headers.forEach(h => html += `<td>${row[h] === "" || row[h] == null ? "∅" : row[h]}</td>`);
        html += "</tr>";
    });
    html += "</tbody></table>";

    document.getElementById("overviewContent").innerHTML = html;
    summaryReport += `\n=== DATASET OVERVIEW ===\nRows: ${rows}, Columns: ${cols}\nColumns: ${headers.join(", ")}\n`;
});

// Missing Values
document.getElementById("showMissingBtn").addEventListener("click", () => {
    if (!mergedData) {
        alert("Please load data first.");
        return;
    }

    const headers = Object.keys(mergedData[0]);
    const total = mergedData.length;
    const missing = {};

    headers.forEach(col => {
        const missingCount = mergedData.filter(row => !row[col] || row[col] === "").length;
        missing[col] = {
            count: missingCount,
            pct: ((missingCount / total) * 100).toFixed(2)
        };
    });

    let html = "<table><thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr></thead><tbody>";
    Object.entries(missing).forEach(([col, stats]) => {
        html += `<tr><td>${col}</td><td>${stats.count}</td><td>${stats.pct}%</td></tr>`;
    });
    html += "</tbody></table>";

    document.getElementById("missingContent").innerHTML = html;

    summaryReport += "\n=== MISSING VALUES ===\n";
    Object.entries(missing).forEach(([col, stats]) => {
        summaryReport += `${col}: ${stats.count} (${stats.pct}%)\n`;
    });
});

// Stats Summary (Train Only)
document.getElementById("showStatsBtn").addEventListener("click", () => {
    if (!trainData) {
        alert("Please load data first.");
        return;
    }

    const numericCols = ["Age", "Fare", "SibSp", "Parch"];
    const catCols = ["Pclass", "Sex", "Embarked"];

    // Numeric stats
    let html = "<h3>Numeric Features (Train Set)</h3><table><thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead><tbody>";
    let reportText = "\n=== NUMERIC SUMMARY (TRAIN) ===\n";

    numericCols.forEach(col => {
        const values = trainData
            .map(r => parseFloat(r[col]))
            .filter(v => !isNaN(v));
        if (values.length === 0) return;

        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const median = values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
        const std = Math.sqrt(values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length);
        const min = Math.min(...values);
        const max = Math.max(...values);

        html += `<tr><td>${col}</td><td>${mean.toFixed(2)}</td><td>${median.toFixed(2)}</td><td>${std.toFixed(2)}</td><td>${min}</td><td>${max}</td></tr>`;
        reportText += `${col}: Mean=${mean.toFixed(2)}, Median=${median.toFixed(2)}, Std=${std.toFixed(2)}, Min=${min}, Max=${max}\n`;
    });
    html += "</tbody></table>";

    // Categorical counts + survival rates
    const totalTrain = trainData.length;
    const survived = trainData.filter(r => r.Survived === "1").length;

    // Survival rate overall
    html += `<p><strong>Overall Survival Rate:</strong> ${survived} / ${totalTrain} = ${pct(survived, totalTrain)}</p>`;
    reportText += `\nOverall Survival: ${survived}/${totalTrain} (${pct(survived, totalTrain)})\n`;

    // By Pclass, Sex, Embarked
    ["Pclass", "Sex", "Embarked"].forEach(groupCol => {
        const groups = {};
        trainData.forEach(row => {
            const key = row[groupCol] || "Unknown";
            if (!groups[key]) groups[key] = { total: 0, survived: 0 };
            groups[key].total++;
            if (row.Survived === "1") groups[key].survived++;
        });

        html += `<h4>Survival by ${groupCol}</h4><table><thead><tr><th>${groupCol}</th><th>Total</th><th>Survived</th><th>Rate</th></tr></thead><tbody>`;
        reportText += `\n=== SURVIVAL BY ${groupCol.toUpperCase()} ===\n`;
        Object.entries(groups).forEach(([key, stats]) => {
            const rate = pct(stats.survived, stats.total);
            html += `<tr><td>${key}</td><td>${stats.total}</td><td>${stats.survived}</td><td>${rate}</td></tr>`;
            reportText += `${key}: ${stats.survived}/${stats.total} (${rate})\n`;
        });
        html += "</tbody></table>";
    });

    document.getElementById("statsContent").innerHTML = html;
    summaryReport += reportText;
});

// Visualizations
document.getElementById("showVizBtn").addEventListener("click", () => {
    if (!mergedData || !trainData) {
        alert("Please load data first.");
        return;
    }

    const vizDiv = document.getElementById("vizContent");
    vizDiv.innerHTML = `
    <div class="flex-row">
      <div class="flex-chart"><div id="ageHist" class="chart-container"></div></div>
      <div class="flex-chart"><div id="fareBox" class="chart-container"></div></div>
    </div>
    <div class="flex-row">
      <div class="flex-chart"><div id="survivalPie" class="chart-container"></div></div>
      <div class="flex-chart"><div id="corrHeatmap" class="chart-container"></div></div>
    </div>
  `;

    // Age Histogram
    const ageData = trainData
        .map(r => parseFloat(r.Age))
        .filter(v => !isNaN(v));
    Plotly.newPlot("ageHist", [{
        x: ageData,
        type: "histogram",
        marker: { color: "#4CAF50" }
    }], {
        title: "Age Distribution (Train)",
        plot_bgcolor: "white",
        paper_bgcolor: "white"
    });

    // Fare Boxplot
    const fareData = trainData
        .map(r => parseFloat(r.Fare))
        .filter(v => !isNaN(v));
    Plotly.newPlot("fareBox", [{
        y: fareData,
        type: "box",
        boxpoints: "outliers",
        marker: { color: "#FFC107" }
    }], {
        title: "Fare Distribution (Train)",
        plot_bgcolor: "white",
        paper_bgcolor: "white"
    });

    // Survival Pie
    const survivedCount = trainData.filter(r => r.Survived === "1").length;
    const diedCount = trainData.length - survivedCount;
    Plotly.newPlot("survivalPie", [{
        labels: ["Survived", "Died"],
        values: [survivedCount, diedCount],
        type: "pie",
        marker: { colors: ["#4CAF50", "#F44336"] }
    }], {
        title: "Survival Distribution (Train)",
        plot_bgcolor: "white",
        paper_bgcolor: "white"
    });

    // Correlation Heatmap (numeric cols only)
    const numCols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"];
    const matrix = [];
    numCols.forEach(() => matrix.push([]));

    // Compute correlations
    numCols.forEach((col1, i) => {
        numCols.forEach((col2, j) => {
            const x = trainData.map(r => parseFloat(r[col1])).filter(v => !isNaN(v));
            const y = trainData.map(r => parseFloat(r[col2])).filter(v => !isNaN(v));
            // Align by index (only where both exist)
            const pairs = trainData
                .map(r => [parseFloat(r[col1]), parseFloat(r[col2])])
                .filter(([a, b]) => !isNaN(a) && !isNaN(b));
            if (pairs.length === 0) {
                matrix[i][j] = 0;
                return;
            }
            const xVals = pairs.map(p => p[0]);
            const yVals = pairs.map(p => p[1]);
            const n = xVals.length;
            const sumX = xVals.reduce((a, b) => a + b, 0);
            const sumY = yVals.reduce((a, b) => a + b, 0);
            const sumXY = xVals.reduce((acc, xv, idx) => acc + xv * yVals[idx], 0);
            const sumX2 = xVals.reduce((acc, xv) => acc + xv * xv, 0);
            const sumY2 = yVals.reduce((acc, yv) => acc + yv * yv, 0);
            const corr = (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY) || 1);
            matrix[i][j] = isNaN(corr) ? 0 : corr;
        });
    });

    Plotly.newPlot("corrHeatmap", [{
        z: matrix,
        x: numCols,
        y: numCols,
        type: "heatmap",
        colorscale: [
            [0, "green"],
            [0.5, "yellow"],
            [1, "red"]
        ],
        zmin: -1,
        zmax: 1,
        showscale: true
    }], {
        title: "Feature Correlation (Train)",
        plot_bgcolor: "white",
        paper_bgcolor: "white"
    });
});

// Export Functions
document.getElementById("exportCsvBtn").addEventListener("click", () => {
    if (!mergedData) {
        alert("No data to export.");
        return;
    }
    const headers = Object.keys(mergedData[0]);
    let csv = headers.join(",") + "\n";
    mergedData.forEach(row => {
        csv += headers.map(h => `"${String(row[h] ?? "").replace(/"/g, '""')}"`).join(",") + "\n";
    });
    downloadBlob(csv, "titanic_merged.csv", "text/csv");
});

document.getElementById("exportJsonBtn").addEventListener("click", () => {
    if (!mergedData) {
        alert("No data to export.");
        return;
    }
    const json = JSON.stringify({ data: mergedData }, null, 2);
    downloadBlob(json, "titanic_summary.json", "application/json");
});

document.getElementById("exportReportBtn").addEventListener("click", () => {
    if (!summaryReport) {
        alert("Run analysis steps first to generate a report.");
        return;
    }
    downloadBlob(summaryReport, "titanic_eda_report.txt", "text/plain");
});
