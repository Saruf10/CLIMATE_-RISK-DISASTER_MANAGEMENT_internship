🌲 Forest Fire Risk Mapping
This project analyzes historical forest fire data to predict fire risk based on weather conditions. Early prediction can help prevent wildfires and protect lives and resources.

📂 Dataset Source

Forest Fire Dataset – Kaggle

🔑 Key Features
X, Y → Spatial coordinates
month, day → Time of observation
FFMC, DMC, DC, ISI → Fire weather indices
temp, RH, wind, rain → Weather conditions
area → Burned area (ha)
fire → Binary target (fire occurred or not)

✅ Work Completed

Data Loading & Exploration:
Loaded dataset using Pandas.
Displayed head, info, and descriptive statistics.
Verified no missing values.

Exploratory Data Analysis (EDA):
Univariate analysis: Histograms, count plots, and boxplots for continuous and categorical features.
Bivariate analysis: Scatterplots & boxplots to check relationships between variables and fire occurrence.
Correlation matrix: Visualized correlations among numerical features.
Pairplot: Plotted relationships between numerical variables, colored by fire occurrence.

Data Preprocessing:
Encoded categorical variables (month, day) using one-hot encoding.
Split data into training and testing sets (80/20).
Applied feature scaling with StandardScaler:
fit_transform() on training data
transform() on test data
