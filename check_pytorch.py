import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import numpy as np

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1).to(device)
# Printing a module shows all its submodules
print(model)


for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")



import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import numpy as np



# Corrected Continuous XOR logic function
def continuous_xor(x, y):
    # Convert to integers for bitwise XOR, then convert back to float if needed
    return ((x > 0).float().int() ^ (y > 0).float().int()).float()


# Generate random (x, y) pairs in the range of [-1, 1]
n_samples = 10000



x = (torch.rand(n_samples, 1, device=device) * 2 - 1).to(device)  # Scale to [-1, 1]
y = (torch.rand(n_samples, 1, device=device) * 2 - 1).to(device)  # Scale to [-1, 1]

# Apply the continuous XOR logic
labels = continuous_xor(x, y)

# Combine the (x, y) pairs
inputs = torch.cat((x, y), dim=1)

# Create a TensorDataset
dataset = TensorDataset(inputs, labels)

# Split the dataset into train, validation, and test sets
train_size = int(n_samples * 0.8)  # 80% of the dataset
val_size = int(n_samples * 0.05)  # 5% of the dataset
test_size = n_samples - (train_size + val_size)  # The remaining 15%
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))



# Create DataLoader for each set
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



# Visualization of a batch from the training set




optimizer = optim.SGD(model.parameters(), lr=0.01)

#import torch.optim as optim

#optimizer = optim.Adam(model.parameters(), lr=0.001)



# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()

# Training Loop
epochs = 100
loss_log = []
epoch_log = []
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    loss_log.append(val_loss/len(val_loader))
    epoch_log.append(epoch)
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}')

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'epochs' and 'val_loss' are your existing lists for 100 epochs
# epochs = [1, 2, 3, ..., 100]
# val_loss = [loss_value1, loss_value2, loss_value3, ..., loss_value100]

# Set the seaborn style for plotting
sns.set(style="whitegrid")

# Plotting with enhanced aesthetics for 100 epochs
plt.figure(figsize=(14, 8))
plt.plot(epoch_log, loss_log, label='Validation Loss',  markersize=8, linewidth=2)

# Adjusting titles and labels with enhanced font settings
plt.title('Validation Loss Over 100 Epochs', fontsize=20, fontweight='bold', color='darkslateblue')
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Validation Loss', fontsize=16, fontweight='bold')

# Adjusting x-axis to show every 10th epoch for better readability
plt.xticks(range(1, 101, 10), fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(fontsize=14, frameon=True, shadow=True, borderpad=1)

# Optional: Remove the top and right spines for a cleaner look and adjust the grid

plt.show()



state_dict = model.state_dict()
print(state_dict)


# torch.save(object, filename). For the filename, any extension can be used
torch.save(state_dict, "our_model.tar")


# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("our_model.tar")

# Create a new model and load the state
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)

# Verify that the parameters are the same
print("Original model\n", model.state_dict())
print("\nLoaded model\n", new_model.state_dict())



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def evaluate_model_and_metrics(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Convert to binary predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plotting
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Bar chart for metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_names, y=metrics)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming test_loader is defined and contains the test dataset
evaluate_model_and_metrics(model, test_loader, device)



# Commented out IPython magic to ensure Python compatibility.
# Import tensorboard logger from PyTorch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/xor_experiment_1')
# Load tensorboard extension for Jupyter Notebook, only need to start TB in the notebook
# %load_ext tensorboard



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# Assuming SimpleClassifier, train_loader, val_loader are defined

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
model_board = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1).to(device)

# Optimizer and Criterion
optimizer = optim.SGD(model_board.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()

# TensorBoard Writer
writer = SummaryWriter()

# Training Loop
epochs = 100
for epoch in range(epochs):
    model_board.train()
    train_loss, train_correct = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_board(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        train_correct += predictions.eq(labels.unsqueeze(1).data.view_as(predictions)).sum().item()

    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)

    # Log training metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    # Log gradients and weights histograms
    for name, param in model_board.named_parameters():
        writer.add_histogram(f'{name}/gradients', param.grad, epoch)
        writer.add_histogram(f'{name}/weights', param, epoch)

    # Validation phase
    model_board.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_board(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            val_correct += predictions.eq(labels.unsqueeze(1).data.view_as(predictions)).sum().item()

    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader)

    # Log validation metrics
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Close the writer when done
writer.close()



# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir runs
