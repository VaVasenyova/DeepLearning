let fullData = null;
let trainData = null;

function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) continue;
        const values = [];
        let inQuotes = false;
        let current = '';

        for (let j = 0; j < line.length; j++) {
            const char = line[j];
            if (char === '"' && (j === 0 || line[j - 1] !== '\\')) {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                values.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        values.push(current);

        const row = {};
        headers.forEach((header, idx) => {
            let val = (values[idx] || '').trim();
            if (val === '') {
                row[header] = null;
            } else if (val.startsWith('"') && val.endsWith('"')) {
                row[header] = val.slice(1, -1);
            } else if (!isNaN(val) && val !== '') {
                row[header] = parseFloat(val);
            } else {
                row[header] = val;
            }
        });
        rows.push(row);
    }
    return rows;
}

function renderTable(data, columns) {
    if (!data.length) return '<p>No data</p>';
    let html = '<table><thead><tr>';
    columns.forEach(col => html += `<th>${col}</th>`);
    html += '</tr></thead><tbody>';
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const val = row[col];
            html += `<td>${val === null || val === undefined ? '' : val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please select both train.csv and test.csv');
        return;
    }

    document.getElementById('loading').style.display = 'block';

    try {
        const trainText = await trainFile.text();
        const testText = await testFile.text();

        let train = parseCSV(trainText);
        let test = parseCSV(testText);

        train = train.map(r => ({ ...r, source: 'train' }));
        test = test.map(r => ({ ...r, source: 'test' }));

        fullData = [...train, ...test];
        trainData = fullData.filter(d => d.source === 'train');

        updateOverview();
        updateMissingValues();
        updateStats();
        updateVisualizations();
        document.getElementById('export-section').classList.remove('hidden');

        document.getElementById('loading').style.display = 'none';
    } catch (err) {
        console.error(err);
        alert('Error processing files: ' + err.message);
        document.getElementById('loading').style.display = 'none';
    }
}

function updateOverview() {
    const shape = `${fullData.length} rows × ${Object.keys(fullData[0]).length} columns`;
    document.getElementById('shape-text').textContent = shape;

    const preview = fullData.slice(0, 10);
    const cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'source'];
    document.getElementById('preview-table').innerHTML = renderTable(preview, cols);
    document.getElementById('overview-section').classList.remove('hidden');
}

function updateMissingValues() {
    const cols = Object.keys(fullData[0]);
    const total = fullData.length;
    const missing = cols.map(col => {
        const count = fullData.filter(row => row[col] === null).length;
        const pct = ((count / total) * 100).toFixed(2);
        return { Column: col, 'Missing Count': count, 'Missing Percentage': `${pct}%` };
    });

    document.getElementById('missing-table').innerHTML = renderTable(missing, ['Column', 'Missing Count', 'Missing Percentage']);

    // График пропусков — белый фон
    Plotly.newPlot('missing-chart', [{
        x: missing.map(m => m.Column),
        y: missing.map(m => m['Missing Count']),
        type: 'bar',
        marker: { color: '#e74c3c' }
    }], {
        title: 'Missing Values per Column',
        xaxis: { title: 'Column', showgrid: false, zeroline: false },
        yaxis: { title: 'Count', showgrid: true, gridcolor: '#ddd' },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white'
    });

    document.getElementById('missing-section').classList.remove('hidden');
}

function updateStats() {
    const numericCols = ['Age', 'Fare', 'SibSp', 'Parch'];
    const numStats = numericCols.map(col => {
        const values = trainData
            .map(d => d[col])
            .filter(v => v !== null && !isNaN(v));
        if (values.length === 0) return { Feature: col, Mean: 'N/A', Median: 'N/A', 'Std Dev': 'N/A', Min: 'N/A', Max: 'N/A' };
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const sorted = [...values].sort((a, b) => a - b);
        const median = sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        const min = Math.min(...values);
        const max = Math.max(...values);
        const std = Math.sqrt(values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length);
        return {
            Feature: col,
            Mean: mean.toFixed(2),
            Median: median.toFixed(2),
            'Std Dev': std.toFixed(2),
            Min: min.toFixed(2),
            Max: max.toFixed(2)
        };
    });
    document.getElementById('numeric-stats').innerHTML = renderTable(numStats, ['Feature', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']);

    const pclassCounts = { 1: 0, 2: 0, 3: 0 };
    trainData.forEach(d => { if (d.Pclass in pclassCounts) pclassCounts[d.Pclass]++; });
    const pclassTotal = trainData.length;
    const pclassTable = Object.entries(pclassCounts).map(([cls, cnt]) => ({
        Pclass: cls,
        Count: cnt,
        Percentage: `${((cnt / pclassTotal) * 100).toFixed(2)}%`
    }));
    document.getElementById('pclass-table').innerHTML = renderTable(pclassTable, ['Pclass', 'Count', 'Percentage']);

    const sexCounts = { male: 0, female: 0 };
    trainData.forEach(d => { if (d.Sex in sexCounts) sexCounts[d.Sex]++; });
    const sexTable = Object.entries(sexCounts).map(([sex, cnt]) => ({
        Sex: sex,
        Count: cnt,
        Percentage: `${((cnt / pclassTotal) * 100).toFixed(2)}%`
    }));
    document.getElementById('sex-table').innerHTML = renderTable(sexTable, ['Sex', 'Count', 'Percentage']);

    const embarkedCounts = { S: 0, C: 0, Q: 0 };
    trainData.forEach(d => { if (d.Embarked in embarkedCounts) embarkedCounts[d.Embarked]++; });
    const embarkedTable = Object.entries(embarkedCounts).map(([port, cnt]) => ({
        Embarked: port,
        Count: cnt,
        Percentage: `${((cnt / pclassTotal) * 100).toFixed(2)}%`
    }));
    document.getElementById('embarked-table').innerHTML = renderTable(embarkedTable, ['Embarked', 'Count', 'Percentage']);

    const survPclass = [1, 2, 3].map(cls => {
        const total = trainData.filter(d => d.Pclass == cls).length;
        const survived = trainData.filter(d => d.Pclass == cls && d.Survived === 1).length;
        return {
            Pclass: cls,
            Total: total,
            Survived: survived,
            'Survival Rate': total ? `${((survived / total) * 100).toFixed(2)}%` : '0%'
        };
    });
    document.getElementById('survival-pclass').innerHTML = renderTable(survPclass, ['Pclass', 'Total', 'Survived', 'Survival Rate']);

    const survSex = ['male', 'female'].map(sex => {
        const total = trainData.filter(d => d.Sex === sex).length;
        const survived = trainData.filter(d => d.Sex === sex && d.Survived === 1).length;
        return {
            Sex: sex,
            Total: total,
            Survived: survived,
            'Survival Rate': total ? `${((survived / total) * 100).toFixed(2)}%` : '0%'
        };
    });
    document.getElementById('survival-sex').innerHTML = renderTable(survSex, ['Sex', 'Total', 'Survived', 'Survival Rate']);

    const survEmbarked = ['S', 'C', 'Q'].map(port => {
        const total = trainData.filter(d => d.Embarked === port).length;
        const survived = trainData.filter(d => d.Embarked === port && d.Survived === 1).length;
        return {
            Embarked: port,
            Total: total,
            Survived: survived,
            'Survival Rate': total ? `${((survived / total) * 100).toFixed(2)}%` : '0%'
        };
    });
    document.getElementById('survival-embarked').innerHTML = renderTable(survEmbarked, ['Embarked', 'Total', 'Survived', 'Survival Rate']);

    document.getElementById('stats-section').classList.remove('hidden');
}

function updateVisualizations() {
    const survived = trainData.filter(d => d.Survived === 1).length;
    const not = trainData.filter(d => d.Survived === 0).length;
    Plotly.newPlot('survival-bar', [
        { x: ['Survived'], y: [survived], type: 'bar', name: 'Survived', marker: { color: '#2ecc71' } },
        { x: ['Not Survived'], y: [not], type: 'bar', name: 'Not Survived', marker: { color: '#e74c3c' } }
    ], {
        title: 'Survival Distribution (Train Set)',
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: { showgrid: false },
        yaxis: { showgrid: true, gridcolor: '#eee' }
    });

    const ages = fullData.map(d => d.Age).filter(a => a != null);
    Plotly.newPlot('age-hist', [{
        x: ages,
        type: 'histogram',
        nbinsx: 30,
        marker: { color: '#3498db' }
    }], {
        title: 'Age Distribution',
        xaxis: { title: 'Age', showgrid: false },
        yaxis: { title: 'Count', showgrid: true, gridcolor: '#eee' },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white'
    });

    const fares = fullData.map(d => d.Fare).filter(f => f != null);
    Plotly.newPlot('fare-box', [{
        y: fares,
        type: 'box',
        name: 'Fare',
        boxpoints: 'outliers',
        marker: { color: '#9b59b6' }
    }], {
        title: 'Fare Distribution (with Outliers)',
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        yaxis: { showgrid: true, gridcolor: '#eee' }
    });

    const corrCols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
    const clean = trainData.filter(d =>
        corrCols.every(c => d[c] !== null && !isNaN(d[c]))
    );
    if (clean.length > 1) {
        const matrix = corrCols.map(c => clean.map(d => d[c]));
        function pearson(x, y) {
            const n = x.length;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
            const sumX2 = x.reduce((a, b) => a + b * b, 0);
            const sumY2 = y.reduce((a, b) => a + b * b, 0);
            const num = n * sumXY - sumX * sumY;
            const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            return den === 0 ? 0 : num / den;
        }
        const corr = corrCols.map((_, i) =>
            corrCols.map((_, j) => pearson(matrix[i], matrix[j]))
        );
        Plotly.newPlot('correlation-heatmap', [{
            z: corr,
            x: corrCols,
            y: corrCols,
            type: 'heatmap',
            colorscale: [
                [0.0, '#2ecc71'],
                [0.5, '#ffffff'],
                [1.0, '#f1c40f']
            ],
            zmin: -1,
            zmax: 1,
            colorbar: { title: 'Correlation' }
        }], {
            title: 'Correlation Matrix (Train Set)',
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            xaxis: { showgrid: false, tickangle: -30 },
            yaxis: { showgrid: false }
        });
    }

    document.getElementById('viz-section').classList.remove('hidden');
}

// Экспорт
function exportMergedCSV() {
    const headers = Object.keys(fullData[0]);
    const csv = [headers.join(',')].concat(
        fullData.map(row =>
            headers.map(h => {
                const val = row[h];
                if (val === null || val === undefined) return '';
                if (typeof val === 'string' && val.includes(',')) return `"${val}"`;
                return val;
            }).join(',')
        )
    ).join('\n');
    downloadBlob(csv, 'titanic_merged.csv', 'text/csv');
}

function exportMergedJSON() {
    const json = JSON.stringify(fullData, null, 2);
    downloadBlob(json, 'titanic_merged.json', 'application/json');
}

function exportSummaryText() {
    let summary = `Titanic Dataset Exploratory Data Analysis\n`;
    summary += `Dataset shape: ${fullData.length} rows × ${Object.keys(fullData[0]).length} columns\n\n`;

    const missing = Object.keys(fullData[0]).map(col => {
        const count = fullData.filter(r => r[col] === null).length;
        const pct = ((count / fullData.length) * 100).toFixed(2);
        return `${col}: ${count} (${pct}%)`;
    });
    summary += `Missing Values:\n${missing.join('\n')}\n\n`;

    const numericCols = ['Age', 'Fare', 'SibSp', 'Parch'];
    summary += `Numeric Features Summary (Train):\n`;
    numericCols.forEach(col => {
        const vals = trainData.map(d => d[col]).filter(v => v != null && !isNaN(v));
        if (vals.length === 0) return;
        const mean = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2);
        const sorted = [...vals].sort((a, b) => a - b);
        const median = sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        summary += `${col}: Mean=${mean}, Median=${median.toFixed(2)}, Min=${min}, Max=${max}\n`;
    });

    downloadBlob(summary, 'titanic_summary.txt', 'text/plain');
}