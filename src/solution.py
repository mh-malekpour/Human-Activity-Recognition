# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class mlp(nn.Module):
    def __init__(self, time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.time_periods * 3, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# # WRITE CODE HERE

class cnn(nn.Module):
    def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.n_sensors = n_sensors
        self.time_periods = time_periods
        self.n_classes = n_classes
        self.conv1 = nn.Conv1d(n_sensors, 100, kernel_size=10)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=10)
        self.conv3 = nn.Conv1d(100, 160, kernel_size=10)
        self.conv4 = nn.Conv1d(160, 160, kernel_size=10)
        self.maxpool = nn.MaxPool1d(3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(160, self.n_classes)

    def forward(self, x):
        x = x.view(-1, self.n_sensors, self.time_periods)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        x = self.dropout(x)
        x = x.view(-1, 160)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
