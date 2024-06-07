import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class MLP_L2(nn.Module):
    def __init__(self, X_dim):
        super(MLP_L2, self).__init__()
        
        self.input = X_dim
        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, X):
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return torch.sigmoid(x)
    
class MLP_L3(nn.Module):
    def __init__(self, X_dim):
        super(MLP_L3, self).__init__()
        
        self.input = X_dim
        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, X):
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)   


class MLP_L4(nn.Module):
    def __init__(self, X_dim):
        super(MLP_L4, self).__init__()
        
        self.input = X_dim
        
        # 1D CNN layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # fully connected layers
        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, X):
        # Reshape input for 1D CNN
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        
        return torch.sigmoid(x)
    

class CNN_MLP_L1L2(nn.Module):
    def __init__(self, X_dim):
        super(CNN_MLP_L1L2, self).__init__()
        
        self.input = X_dim
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)
 
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, X):
        # Reshape input for 1D CNN
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.conv1(X))
        x = F.max_pool1d(x, kernel_size=2)  
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return torch.sigmoid(x)
    
class CNN_MLP_L2L2(nn.Module):
    def __init__(self, X_dim):
        super(CNN_MLP_L1L2, self).__init__()
        
        self.input = X_dim
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)
 
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, X):
        # Reshape input for 1D CNN
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.conv1(X))
        x = F.max_pool1d(x, kernel_size=2) 
        x = F.relu(self.conv2(X))
        x = F.max_pool1d(x, kernel_size=2) 
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return torch.sigmoid(x)

class CNN_MLP_L2L3(nn.Module):
    def __init__(self, X_dim):
        super(CNN_MLP_L1L2, self).__init__()
        
        self.input = X_dim
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(X_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
 
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, X):
        # Reshape input for 1D CNN
        X = X.unsqueeze(1) # Add channel dimension (batch_size, 1, X_dim)

        x = F.relu(self.conv1(X))
        x = F.max_pool1d(x, kernel_size=2) 
        x = F.relu(self.conv2(X))
        x = F.max_pool1d(x, kernel_size=2) 
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(X))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        return torch.sigmoid(x)



            
