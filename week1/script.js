document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const fileNameDisplay = document.getElementById('file-name');
    const dataSection = document.getElementById('data-section');
    const edaSection = document.getElementById('eda-section');
    const findingsSection = document.getElementById('findings-section');
    const tableHeaders = document.getElementById('table-headers');
    const tableBody = document.getElementById('table-body');
    const rowCountDisplay = document.getElementById('row-count');

    // Plot containers
    const missingValuesPlot = document.getElementById('missing-values-plot');
    const survivalDistributionPlot = document.getElementById('survival-distribution-plot');
    const survivalBySexPlot = document.getElementById('survival-by-sex-plot');
    const survivalByPclassPlot = document.getElementById('survival-by-pclass-plot');
    const ageDistributionPlot = document.getElementById('age-distribution-plot');
    const correlationPlot = document.getElementById('correlation-plot');

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = `Selected file: ${file.name}`;
            loadData(file);
        }
    });

    // Load and process CSV data
    function loadData(file) {
        d3.csv(URL.createObjectURL(file)).then(function (data) {
            // Show data section
            dataSection.classList.remove('hidden');

            // Display data preview (first 50 rows)
            displayDataPreview(data);

            // Perform EDA and show visualizations after a short delay for better UX
            setTimeout(() => {
                edaSection.classList.remove('hidden');
                performEDA(data);
            }, 500);

            // Show findings section after another delay
            setTimeout(() => {
                findingsSection.classList.remove('hidden');
            }, 1000);

        }).catch(function (error) {
            console.error('Error loading CSV:', error);
            alert('Error loading CSV file. Please check the file format and try again.');
        });
    }

    // Display data preview in table
    function displayDataPreview(data) {
        // Clear previous content
        tableHeaders.innerHTML = '';
        tableBody.innerHTML = '';

        // Get headers from first row
        const headers = Object.keys(data[0]);

        // Create table headers
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            tableHeaders.appendChild(th);
        });

        // Display first 50 rows
        const displayRows = data.slice(0, 50);
        displayRows.forEach(row => {
            const tr = document.createElement('tr');
            headers.forEach(header => {
                const td = document.createElement('td');
                td.textContent = row[header] || '';
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });

        // Display row count
        rowCountDisplay.textContent = `Showing ${displayRows.length} of ${data.length} passengers`;
    }

    // Perform Exploratory Data Analysis and create visualizations
    function performEDA(data) {
        // 1. Missing Values Analysis
        createMissingValuesPlot(data);

        // 2. Survival Distribution
        createSurvivalDistributionPlot(data);

        // 3. Survival Rate by Gender
        createSurvivalBySexPlot(data);

        // 4. Survival Rate by Passenger Class
        createSurvivalByPclassPlot(data);

        // 5. Age Distribution by Survival
        createAgeDistributionPlot(data);

        // 6. Correlation Heatmap
        createCorrelationPlot(data);
    }

    // Create Missing Values Plot
    function createMissingValuesPlot(data) {
        const columns = Object.keys(data[0]);
        const missingCounts = {};

        // Initialize counts
        columns.forEach(col => missingCounts[col] = 0);

        // Count missing values
        data.forEach(row => {
            columns.forEach(col => {
                if (row[col] === '' || row[col] === null || row[col] === undefined) {
                    missingCounts[col]++;
                }
            });
        });

        // Prepare data for Plotly
        const missingData = {
            x: columns,
            y: columns.map(col => missingCounts[col]),
            type: 'bar',
            marker: {
                color: columns.map(col => missingCounts[col] > 0 ? '#c01c28' : '#26a269'),
                opacity: 0.8
            }
        };

        const layout = {
            title: 'Missing Values by Column',
            xaxis: { title: 'Columns' },
            yaxis: { title: 'Missing Value Count' },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            hovermode: 'closest'
        };

        Plotly.newPlot(missingValuesPlot, [missingData], layout, { responsive: true });
    }

    // Create Survival Distribution Plot
    function createSurvivalDistributionPlot(data) {
        let survivedCount = 0;
        let notSurvivedCount = 0;

        data.forEach(row => {
            if (row.Survived === '1') survivedCount++;
            else notSurvivedCount++;
        });

        const survivalData = {
            values: [survivedCount, notSurvivedCount],
            labels: ['Survived', 'Did Not Survive'],
            type: 'pie',
            marker: {
                colors: ['#26a269', '#c01c28']
            },
            textinfo: 'percent+label',
            hole: 0.4
        };

        const layout = {
            title: 'Overall Survival Distribution',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };

        Plotly.newPlot(survivalDistributionPlot, [survivalData], layout, { responsive: true });
    }

    // Create Survival Rate by Sex Plot
    function createSurvivalBySexPlot(data) {
        const sexGroups = { male: { survived: 0, total: 0 }, female: { survived: 0, total: 0 } };

        data.forEach(row => {
            const sex = row.Sex.toLowerCase();
            if (sex === 'male' || sex === 'female') {
                sexGroups[sex].total++;
                if (row.Survived === '1') sexGroups[sex].survived++;
            }
        });

        // Calculate survival rates
        const maleSurvivalRate = (sexGroups.male.survived / sexGroups.male.total * 100).toFixed(1);
        const femaleSurvivalRate = (sexGroups.female.survived / sexGroups.female.total * 100).toFixed(1);

        const survivalData = {
            x: ['Male', 'Female'],
            y: [parseFloat(maleSurvivalRate), parseFloat(femaleSurvivalRate)],
            type: 'bar',
            marker: {
                color: ['#c01c28', '#26a269'],
                opacity: 0.85
            },
            text: [`${maleSurvivalRate}%`, `${femaleSurvivalRate}%`],
            textposition: 'auto'
        };

        const layout = {
            title: 'Survival Rate by Gender',
            xaxis: { title: 'Gender' },
            yaxis: { title: 'Survival Rate (%)', range: [0, 80] },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            bargap: 0.5
        };

        Plotly.newPlot(survivalBySexPlot, [survivalData], layout, { responsive: true });
    }

    // Create Survival Rate by Pclass Plot
    function createSurvivalByPclassPlot(data) {
        const pclassGroups = { '1': { survived: 0, total: 0 }, '2': { survived: 0, total: 0 }, '3': { survived: 0, total: 0 } };

        data.forEach(row => {
            if (pclassGroups[row.Pclass]) {
                pclassGroups[row.Pclass].total++;
                if (row.Survived === '1') pclassGroups[row.Pclass].survived++;
            }
        });

        // Calculate survival rates
        const pclassRates = Object.keys(pclassGroups).map(pclass => {
            const group = pclassGroups[pclass];
            return (group.survived / group.total * 100).toFixed(1);
        });

        const survivalData = {
            x: ['1st Class', '2nd Class', '3rd Class'],
            y: pclassRates.map(rate => parseFloat(rate)),
            type: 'bar',
            marker: {
                color: ['#2a6f97', '#3c8dbc', '#6c9bcf'],
                opacity: 0.85
            },
            text: pclassRates.map(rate => `${rate}%`),
            textposition: 'auto'
        };

        const layout = {
            title: 'Survival Rate by Passenger Class',
            xaxis: { title: 'Passenger Class' },
            yaxis: { title: 'Survival Rate (%)', range: [0, 70] },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            bargap: 0.5
        };

        Plotly.newPlot(survivalByPclassPlot, [survivalData], layout, { responsive: true });
    }

    // Create Age Distribution Plot
    function createAgeDistributionPlot(data) {
        const survivedAges = [];
        const notSurvivedAges = [];

        data.forEach(row => {
            const age = parseFloat(row.Age);
            if (!isNaN(age)) {
                if (row.Survived === '1') survivedAges.push(age);
                else notSurvivedAges.push(age);
            }
        });

        const survivedTrace = {
            x: survivedAges,
            type: 'histogram',
            name: 'Survived',
            opacity: 0.7,
            marker: { color: '#26a269' },
            xbins: { size: 5 }
        };

        const notSurvivedTrace = {
            x: notSurvivedAges,
            type: 'histogram',
            name: 'Did Not Survive',
            opacity: 0.7,
            marker: { color: '#c01c28' },
            xbins: { size: 5 }
        };

        const layout = {
            title: 'Age Distribution by Survival Status',
            xaxis: { title: 'Age' },
            yaxis: { title: 'Count' },
            barmode: 'overlay',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            bargap: 0.1
        };

        Plotly.newPlot(ageDistributionPlot, [survivedTrace, notSurvivedTrace], layout, { responsive: true });
    }

    // Create Correlation Heatmap
    function createCorrelationPlot(data) {
        // Select numerical columns for correlation
        const numericalCols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
        const numericData = [];

        // Process data to extract numerical values
        data.forEach(row => {
            const numRow = {};
            let isValid = true;

            numericalCols.forEach(col => {
                const val = parseFloat(row[col]);
                if (isNaN(val)) {
                    isValid = false;
                } else {
                    numRow[col] = val;
                }
            });

            if (isValid) numericData.push(numRow);
        });

        // Calculate correlation matrix
        const corrMatrix = [];
        numericalCols.forEach(col1 => {
            const rowCorr = [];
            numericalCols.forEach(col2 => {
                rowCorr.push(calculateCorrelation(numericData, col1, col2));
            });
            corrMatrix.push(rowCorr);
        });

        // Create heatmap
        const heatmapData = {
            z: corrMatrix,
            x: numericalCols,
            y: numericalCols,
            type: 'heatmap',
            colorscale: [
                [0, '#c01c28'],
                [0.5, '#f6f5f4'],
                [1, '#26a269']
            ],
            zmin: -1,
            zmax: 1,
            text: corrMatrix.map(row => row.map(val => val.toFixed(2))),
            hoverinfo: 'text'
        };

        const layout = {
            title: 'Correlation Heatmap of Numerical Features',
            xaxis: { title: 'Features' },
            yaxis: { title: 'Features' },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };

        Plotly.newPlot(correlationPlot, [heatmapData], layout, { responsive: true });
    }

    // Calculate Pearson correlation coefficient
    function calculateCorrelation(data, col1, col2) {
        if (data.length === 0) return 0;

        // Calculate means
        const mean1 = data.reduce((sum, row) => sum + row[col1], 0) / data.length;
        const mean2 = data.reduce((sum, row) => sum + row[col2], 0) / data.length;

        // Calculate numerator and denominators
        let numerator = 0;
        let denom1 = 0;
        let denom2 = 0;

        data.forEach(row => {
            const diff1 = row[col1] - mean1;
            const diff2 = row[col2] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        });

        // Calculate correlation
        const denom = Math.sqrt(denom1) * Math.sqrt(denom2);
        return denom === 0 ? 0 : numerator / denom;
    }
});
