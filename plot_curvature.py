import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('v0_1.csv')

# List of curvature columns to plot
curvature_columns = ['curvature_5', 'curvature_10', 'curvature_15', 'curvature_20', 'curvature_25']

# Set up the plotting area: one row, five columns
fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

# Plot each curvature column against steering_angle
for ax, curvature in zip(axes, curvature_columns):
    ax.scatter(df[curvature], df['steering_angle'], alpha=0.5)
    ax.set_title(f'{curvature} vs Steering Angle')
    ax.set_xlabel(curvature)
    ax.set_ylabel('Steering Angle')
    ax.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()