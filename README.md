# California Housing Data Analysis

## Project Overview
This project explores the California Housing dataset (1990 Census) to understand key factors influencing median house values. Through data preprocessing, exploratory data analysis (EDA), and correlation analysis, we examine the relationships between housing characteristics, location, and economic factors.

## Dataset Description
The dataset, California Housing Prices, used in the book ”Hands-On Machine Learning with Scikit-Learn and
TensorFlow” by Aur´elien G´eron, is based on the 1990 California census. It provides a beginner-friendly introduction
to machine learning with manageable data size and basic preprocessing needs.
It includes information about houses in California districts, with summary statistics like:
- Geographical data: longitude, latitude (numerical variables)
- Housing statistics: housing median age, total rooms, total bedrooms, population, households, median income,
median house value (numerical variables)
- Categorical variable: ocean proximity
  
The dataset contains 20,640 entries, with 207 missing values in the total bedrooms feature.

## Key Tasks in the Project
### 1. Data Preprocessing
- Handling Missing Values: Removed NaN rows where necessary (e.g., total_bedrooms) while ensuring minimal impact on statistics.
- One-Hot Encoding: Transformed the categorical feature ocean_proximity into five binary columns.
- Feature Engineering: Introduced new variables (rooms_per_household, bedrooms_per_person, people_per_household) for improved interpretability (alternative analysis).
- Logarithming Scaling for features with extremely skewed distributions (alternative analysis).
- Outlier removal using z-score (alternative analysis).
- Normalization: Applied min-max scaling to numerical features to handle non-normal distributions.

### 2. Exploratory Data Analysis (EDA)
- Distribution Analysis: Used histograms and box plots to examine feature distributions and detect skewness.
- Geospatial Visualization: Created scatterplots with color-coded values to analyze spatial housing trends.
- Correlation Analysis:
  - Calculated Pearson correlation coefficients to measure relationships between features.
  - Computed p-values to filter out statistically insignificant correlations.
  - Visualized relationships using correlation heatmaps.
  
### 3. Key Findings
- Housing location plays a crucial role: Coastal areas, particularly near the bay and islands, tend to have higher house values and incomes, while inland areas exhibit lower values.
- Newer developments are concentrated inland, while older homes are more common along the coast.
- Median income is the strongest predictor of house value, reinforcing the link between economic status and housing affordability.

## Python Libraries Used
- Pandas
- Numpy
- Matplotlib
- Sklearn
- Scipy.stats

## How to Use Code
- Clone the repository
- Install required dependencies (if not already installed)
- Run the main.py or the alternative_main.py
```
python main.py
python alternative_main.py
```
- Or run you can open the main_jupiter.ipynb and alternative_main_jupiter.ipynb from your jupiter notebook
- View results in output visualizations and correlation reports.

