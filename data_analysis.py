import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
data_path = "./datasets/light_source_measurements.csv"
correlation_matrix_path = "./datasets/correlation_matrix.csv"
images_folder = "images"
heatmap_path = os.path.join(images_folder, "correlation_matrix.png")

# Load data
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file {data_path} was not found.")
    exit()

# Compute the correlation matrix
corr_matrix = data.corr()

# Create the "images" folder if it doesn't exist
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, annot=False, cmap="crest", vmin=-1, vmax=1, fmt=".2f", square=True
)
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Save the plot to the "images" folder
try:
    plt.savefig(heatmap_path)
except Exception as e:
    print(f"Error saving the heatmap: {e}")
    exit()

# Show the plot (optional)
plt.show()

# Save the correlation matrix to a CSV file
try:
    corr_matrix.to_csv(correlation_matrix_path)
except Exception as e:
    print(f"Error saving the correlation matrix CSV: {e}")
    exit()

# Identify and print columns highly correlated with 'm_avg'
target_column = "m_avg"

# Exclude 'intensity' and 'm_avg' columns from the analysis
columns_to_check = [
    col for col in corr_matrix.columns if col != "intensity" and col != target_column
]

# Create a dictionary to store correlation values
correlation_values = {
    col: corr_matrix.loc[target_column, col] for col in columns_to_check
}

# Sort the columns by correlation value in descending order
sorted_columns = sorted(
    correlation_values.items(), key=lambda x: abs(x[1]), reverse=True
)

# Print columns in order of highest correlation
print(
    f"Columns ordered by their correlation with '{target_column}' (excluding 'intensity'):"
)
for col, corr_value in sorted_columns:
    print(f" - {col}: {corr_value:.2f}")
