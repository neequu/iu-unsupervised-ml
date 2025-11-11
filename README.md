# Mental Health Clustering Analysis

This project analyzes mental health survey data from tech employees using clustering to identify distinct groups for targeted HR interventions.

## What it does

- Takes messy survey data with 63 questions and cleans it up
- Groups employees into 3 meaningful clusters based on mental health patterns  
- Generates visual reports showing who needs what kind of support

## Files

- `main.py` - runs the full analysis from data loading to visualization
- `mental-heath-in-tech-2016_20161114.csv` - input survey data (download from Kaggle)
- `mental_health_clusters.csv` - output with cluster assignments

## Generated reports

After running the code, you get these images:

- `demographic_analysis.png` - shows gender, age, and mental health basics
- `cluster_comparison.png` - compares the 3 employee groups side by side  
- `clusters.png` - visualizes how the groups separate from each other
- `pca_projection.png` - shows the data in simplified 2D space

## Quick start

1. Put the CSV file in the same folder as main.py
2. Run: `python main.py`
3. Check the generated PNG files and CSV output

The code handles all the data cleaning and analysis automatically. Final clusters show which employees need intensive support vs general wellness programs.
