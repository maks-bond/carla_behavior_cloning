import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network model
# class DrivingModel(nn.Module):
#     def __init__(self):
#         super(DrivingModel, self).__init__()
#         self.fc1 = nn.Linear(9, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 512)
#         self.fc5 = nn.Linear(512, 512)
#         self.fc6 = nn.Linear(512, 256)
#         self.fc7 = nn.Linear(256, 256)
#         self.fc8 = nn.Linear(256, 128)
#         self.fc9 = nn.Linear(128, 64)
#         self.fc10 = nn.Linear(64, 32)
#         self.fc11 = nn.Linear(32, 2)

#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.2)
#         self.dropout3 = nn.Dropout(0.2)
#         self.dropout4 = nn.Dropout(0.2)
#         self.dropout5 = nn.Dropout(0.2)

#         self.batch_norm1 = nn.BatchNorm1d(128)
#         self.batch_norm2 = nn.BatchNorm1d(256)
#         self.batch_norm3 = nn.BatchNorm1d(256)
#         self.batch_norm4 = nn.BatchNorm1d(512)
#         self.batch_norm5 = nn.BatchNorm1d(512)
#         self.batch_norm6 = nn.BatchNorm1d(256)
#         self.batch_norm7 = nn.BatchNorm1d(256)
#         self.batch_norm8 = nn.BatchNorm1d(128)
#         self.batch_norm9 = nn.BatchNorm1d(64)
#         self.batch_norm10 = nn.BatchNorm1d(32)

#     def forward(self, x):
#         x = F.relu(self.batch_norm1(self.fc1(x)))
#         x = F.relu(self.batch_norm2(self.fc2(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.batch_norm3(self.fc3(x)))
#         x = F.relu(self.batch_norm4(self.fc4(x)))
#         x = self.dropout2(x)
#         x = F.relu(self.batch_norm5(self.fc5(x)))
#         x = F.relu(self.batch_norm6(self.fc6(x)))
#         x = self.dropout3(x)
#         x = F.relu(self.batch_norm7(self.fc7(x)))
#         x = F.relu(self.batch_norm8(self.fc8(x)))
#         x = self.dropout4(x)
#         x = F.relu(self.batch_norm9(self.fc9(x)))
#         x = self.dropout5(x)
#         x = F.relu(self.batch_norm10(self.fc10(x)))
#         x = self.fc11(x)
#         return x

# class DrivingModel(nn.Module):
#     def __init__(self):
#         super(DrivingModel, self).__init__()
#         self.fc1 = nn.Linear(9, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)

#         self.dropout = nn.Dropout(0.2)
#         self.batch_norm1 = nn.BatchNorm1d(64)
#         self.batch_norm2 = nn.BatchNorm1d(32)

#     def forward(self, x):
#         x = F.relu(self.batch_norm1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.batch_norm2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

class DrivingModel(nn.Module):
    def __init__(self):
        super(DrivingModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)

        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.batch_norm4 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x