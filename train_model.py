import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import DrivingModel

# Define the dataset class
class DrivingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[['min_dist_to_left_bnd', 'min_dist_to_right_bnd', 'heading_delta', 'speed',
                                   'curvature_5', 'curvature_10', 'curvature_15', 'curvature_20', 'curvature_25']].values
        self.labels = self.data[['accel', 'steering_angle']].values

        # Normalize the features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Load the data
data_name = 'data_2024-07-07_01-26-54'
csv_file = data_name + '.csv'
dataset = DrivingDataset(csv_file)

# Split the data into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("train_dataset.shape: ", len(train_dataset))
print("test_dataset.shape: ", len(test_dataset))

# Define data loaders for batching
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device is: ", device)

def weighted_loss(output, target, accel_weight, steering_weight):
    # Output and target should have the shape (batch_size, 2)
    accel_loss = nn.L1Loss()(output[:, 0], target[:, 0])  
    steering_loss = nn.L1Loss()(output[:, 1], target[:, 1])
    return accel_weight * accel_loss + steering_weight * steering_loss

# Initialize the model, loss function, and optimizer
model = DrivingModel().to(device)
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

ACCEL_WEIGHT = 0.0
STEERING_WEIGHT = 1.0

# Train the model
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = weighted_loss(outputs, labels, ACCEL_WEIGHT, STEERING_WEIGHT)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

model_path = data_name + '.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = weighted_loss(outputs, labels, ACCEL_WEIGHT, STEERING_WEIGHT)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")