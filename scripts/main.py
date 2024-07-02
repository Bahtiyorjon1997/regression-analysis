import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'data/raw/bodyfat.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())


##########################
# Handle missiong values #
##########################

missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Option 1: Drop rows with missing values
data = data.dropna()

print("Missing values after cleaning:\n", data.isnull().sum())


##########################
# Visualizing dataset ####
##########################

sns.pairplot(data)
plt.suptitle('Pairplot of All Features', y=1.02)
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
