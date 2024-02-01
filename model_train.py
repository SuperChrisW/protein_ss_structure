import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM import LSTM_model, BiLSTM

import matplotlib.pyplot as plt

root_dir = '/home/lwang/models/protein_ss_prediction'
x = np.load(f'{root_dir}/data/x_feature.npy')
y = np.load(f'{root_dir}/data/y_feature.npy')
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33, random_state=42)

print("Train x set shape:", x_train.shape)
print("Train y set shape:", y_train.shape)
print("Val x set shape:", x_val.shape)
print("Val y set shape:", y_val.shape)
print("Test x set shape:", x_test.shape)
print("Test y set shape:", y_test.shape)

# Convert data to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)
x_val_tensor = torch.FloatTensor(x_val)
y_val_tensor = torch.FloatTensor(y_val)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_train_tensor[:100000, :], y_train_tensor[:100000, :])
val_dataset = TensorDataset(x_val_tensor[:20000, :], y_val_tensor[:20000, :])
test_dataset = TensorDataset(x_test_tensor[:10000, :], y_test_tensor[:10000, :])

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, output_size)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#model = SimpleNN(x_train.shape[1], y_train.shape[1]).to(device)
#model_name = 'SimpleNN'
#model = LSTM_model(x_train.shape[1], x_train.shape[1]*2, y_train.shape[1]).to(device)
#model_name = 'LSTM'
model = BiLSTM().to(device)
model_name = 'CNN+BiLSTM'

lossfn = nn.CrossEntropyLoss()
plot = True

epochs = 100
batch = 64
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(epochs):  # number of epochs
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == torch.max(labels, 1)[1]).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = lossfn(outputs, torch.max(labels, 1)[1])
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.max(labels, 1)[1]).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/100], '
            f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% '
            f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

if plot:
    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'/home/lwang/models/protein_ss_prediction/results/{model_name}_train.png')
    plt.close()

    all_preds = []
    all_true = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(torch.max(labels, 1)[1].cpu().numpy())

    conf_mat = confusion_matrix(all_true, all_preds)

    class_labels = list(set(all_true + all_preds))
    class_labels.sort()

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'/home/lwang/models/protein_ss_prediction/results/{model_name}_test.png')



