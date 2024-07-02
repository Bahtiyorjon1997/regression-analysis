### regression analysis ###

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'data/raw/bodyfat.csv'
data = pd.read_csv(file_path)

print(data.head())

# check missing values
missing_values = data.isnull().sum()
print(missing_values)

# description
data_description = data.describe()
print(data_description)

# Heatmap to visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
# plt.show()

# Histograms of all features
data.hist(figsize=(14, 10), bins=20)
plt.suptitle('Histograms of All Features')
# plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(data)
plt.suptitle('Pairplot of All Features', y=1.02)
plt.show()
