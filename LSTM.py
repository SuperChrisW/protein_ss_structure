import torch
from torch import nn
import torch.nn.functional as F

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, droprate = 0.2):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        
        self.LSTM_layer1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.LSTM_layer2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(droprate)
        self.batch = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.LSTM_layer1(x)
        x, _ = self.LSTM_layer2(x)
 
        x = x[:, -1, :]
      
        # Apply dropout and batch normalization
        x = self.drop(x)
        x = self.batch(x)
        
        # Pass through the fully connected layer
        x = F.relu(self.fc1(x))
        print(x.shape)
        
        return x

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), padding='same')
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(kernel_size=(9, 9), stride=(1, 1), padding=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=170, hidden_size=8, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(16, 8)  # 16 units total (8 in each direction)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        #print('initial: ', x.shape)
        x=x.unsqueeze(1)
        x = x.view(-1, 1, 60, 25)  # Reshape input
        x = F.relu(self.bn1(self.conv1(x)))
        #print('conv1: ', x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print('conv2: ', x.shape)
        x = self.pool(x)
        #print('maxpooling: ', x.shape)
        x = x.permute(0, 2, 1, 3)  # Flatten for LSTM
        #print('maxpooling: ', x.shape)
        x = x.reshape(-1, 52, 170)
        #print('flatten:', x.shape)

        x = self.dropout(x)
        x, _ = self.lstm(x)  # Reshape for LSTM
        #print('lstm:', x.shape)
        x = self.dropout(x[:, -1, :])  # Use output of last LSTM sequence
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 

        return x