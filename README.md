# Cardiovascular-Risk-Factors-A-Data-Driven-Analysis
"An exploratory data analysis (EDA) of cardiovascular disease risk factors using 70,000 patient records. Analyzes demographics, clinical metrics (BP, cholesterol, BMI), and lifestyle habits to uncover key health trends. Delivers actionable insights for preventive care through Python-powered visualizations and statistical analysis."


# Exploring Cardiovascular Health Trends: An EDA Approach

## Table of Contents
 
 - [Project Overview](#project-overview)
 - [Data Source](#data-source)
 - [Tools](#tools)
 - [Data Cleaning](#data-cleaning)
 - [Exploratory Data Analysis](#exploratory-data-analysis)
 - [Results and Findings](#results-and-findings)

## Project Overview

This project analyzes cardiovascular disease risk factors using patient health data. We examine:

 - Demographic distributions (age, gender)
 - Physiological measurements (blood pressure, cholesterol, BMI)
 - Lifestyle factors (smoking, alcohol, activity levels)
 - Cardiovascular disease prevalence and correlations

## Data Source

Dataset: health_data.csv containing:

 - 70,000 patient records (age 30-65)
 - 12 clinical and behavioral features
 - Binary cardiovascular disease indicator
 - School location identifiers

Key Features:

- Age, gender, height, weight
- Blood pressure (systolic/diastolic)
- Cholesterol, glucose levels
- Smoking/alcohol habits, physical activity

## Tools

This project utilizes Python for Exploratory Data Analysis (EDA), leveraging the following libraries:
- pandas – Data manipulation and preprocessing
- numpy – Numerical operations
- matplotlib & seaborn – Data visualization
- scikit-learn – Statistical analysis and preprocessing
- Jupyter Notebook – Interactive code execution

```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization
plt.style.use('ggplot')
%matplotlib inline
```

## Data Cleaning

```python
# Load data
df = pd.read_csv('health_data.csv')  # Reads CSV into a DataFrame
print(f"Shape: {df.shape}")          # Checks dataset dimensions (rows, columns)
print(f"Missing values:\n{df.isnull().sum()}")  # Counts missing values per column

# Convert age from days to years
df['age'] = df['age'] / 365.25      # Assumes age was originally stored in days

# Calculate BMI: weight (kg) / (height (m))^2
df['bmi'] = df['weight'] / (df['height']/100)**2  # Converts height from cm to m

# Filter unrealistic blood pressure values
df = df[(df['ap_hi'] >= 90) & (df['ap_hi'] <= 250)]  # Systolic BP range
df = df[(df['ap_lo'] >= 60) & (df['ap_lo'] <= 150)]  # Diastolic BP range

# Map categorical variables
df['gender'] = df['gender'].map({1: 'male', 2: 'female'})  # Converts numeric to string labels
df['cardio'] = df['cardio'].map({0: 'no', 1: 'yes'})      # Binary CVD status to yes/no
```
- Data Loading: Reads CSV and inspects structure.
- Age Conversion: Transforms age from days to years for interpretability.
- BMI Calculation: Derives Body Mass Index using metric units.
- Outlier Removal: Filters invalid blood pressure readings (e.g., systolic BP < 90 or > 250).
- Categorical Mapping: Replaces numeric codes with descriptive labels (e.g., 1 → 'male').

## Exploratory Data Analysis 

### Demographic Analysis

```python
# Age distribution by cardio status
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', hue='cardio', bins=30, kde=True)
plt.title('Age Distribution by Cardiovascular Disease Status')
plt.xlabel('Age (years)')
```
- Purpose: Compare age distributions between patients with/without CVD.
- Output: A histogram with KDE curves showing:
  - CVD-positive patients tend to be older.
  - Bimodal distribution possible (peaks at ~50 and ~60 years).

### Physiological Factors

```python
# Blood pressure analysis
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x='cardio', y='ap_hi', data=df)
plt.title('Systolic Blood Pressure')

plt.subplot(1,2,2)
sns.boxplot(x='cardio', y='ap_lo', data=df)
plt.title('Diastolic Blood Pressure')
```
- Purpose: Compare blood pressure metrics across CVD groups.
- Output: Side-by-side boxplots revealing:
  - Higher median systolic/diastolic BP in CVD-positive patients.
  - Wider IQR (interquartile range) for CVD group, indicating greater variability.

### Lifestyle Factors

```python
# Activity vs cardio
activity_cardio = pd.crosstab(df['active'], df['cardio'], normalize='index')*100
activity_cardio.plot(kind='bar', stacked=True)
plt.title('Physical Activity vs Cardiovascular Disease')
plt.ylabel('Percentage (%)')
```
- Purpose: Analyze how physical activity affects CVD rates.
- Method:
  - crosstab calculates percentage of CVD cases per activity level.
  - Stacked bar chart visualizes proportions.
- Output: Active patients show lower CVD prevalence (e.g., 30% vs. 50% in inactive group).

### Correlation Analysis

```python
# Correlation matrix
corr = df[['age', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'cardio']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
```
- Purpose: Identify pairwise relationships between features.
- Output: Heatmap highlighting:
  - Strong positive correlation: ap_hi ↔ ap_lo (blood pressure metrics).
  - Moderate correlation: age ↔ cardio (older age → higher CVD risk).

## Results and Findings

1. Clinical Interventions:
   - Prioritize BP monitoring for patients over 50
   - Implement cholesterol management programs
2. Preventive Measures:
   - Promote physical activity initiatives
   - Target smoking cessation programs
3. Screening Protocols:
   - Annual cardiovascular screening after age 45
   - BMI and BP tracking for high-risk patients

