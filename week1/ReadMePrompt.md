Create an interactive EDA dashboard for the Titanic dataset (https://www.kaggle.com/competitions/titanic) that runs entirely in the browser and is suitable for deployment on GitHub Pages. 

The project consists of two files: 
- index.html with a dark-themed UI 
- app.js containing pure JavaScript logic (Plotly.js is loaded via CDN). 

The user uploads train.csv and test.csv, which are merged into a single dataset with an added source column indicating origin. 
The app displays basic dataset info of rows and columns, missing value analysis, and statistical summaries for both numerical and categorical features (computed on the training set only), including survival rates by passenger class, sex, and embarkation port. 

Visualization should include:
- an age histogram
- fare boxplot
- survival distribution
- correlation heatmap
All visualizations are rendered on a white background. The correlation heatmap uses a custom colorscale transitioning from green (negative correlation) through white (zero correlation) to yellow (positive correlation). 

The dashboard also supports exporting the merged dataset as CSV or JSON and downloading a text summary report with all computed statistics.
