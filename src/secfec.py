import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# Step 1: Load and Preprocess the Datasets

# Load datasets
normal_data_path = r'C:\Users\milad\capstone java code\matpower_csv\normal_data.csv'
compromised_strong_path = r'C:\Users\milad\capstone java code\matpower_csv\compromised_data_strong.csv'

normal_data = pd.read_csv(normal_data_path)
compromised_strong = pd.read_csv(compromised_strong_path).sample(500)

# Add labels: 0 for normal, 1 for compromised
normal_data['Label'] = 0
compromised_strong['Label'] = 1

# Combine datasets
combined_data = pd.concat([normal_data, compromised_strong])

# Select features and labels
features = combined_data.columns[:-1]  # All columns except the last one
X = combined_data[features].values
y = combined_data['Label'].values.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42,stratify=y)

print("train:",Counter(y_train[:,0]))
print("test:",Counter(y_test[:,0]))

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Define the Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * np.sqrt(self.d_model)
        return x + self.pe[:x.size(0), :x.size(1)].reshape(x.size(0), x.size(1))

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, nclasses, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(input_dim, nhid)
        self.nhid = nhid
        self.decoder = nn.Linear(nhid, nclasses)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output

# Step 3: Initialize and Train the Model

input_dim = X_train.shape[1]  # Ensure this matches the number of features
nhead = 2
nhid1 = X_train.shape[1]
nlayers = 3
nclasses = 1
dropout = 0.01

global_model = TransformerModel(input_dim, nhead, nhid1, nlayers, nclasses, dropout)
criterion = nn.BCELoss()
optimizer = optim.Adam(global_model.parameters(), lr=0.001)

def local_train(client_model, train_loader, optimizer, criterion, epochs):
    client_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return client_model.state_dict(), running_loss / len(train_loader)

def local_predict(client_model, test_loader, optimizer, criterion):
    running_loss = 0.0
    result=[]
    for inputs, labels in test_loader:
        optimizer.zero_grad()
        outputs = client_model(inputs)
        result += list(outputs.reshape(-1).detach().numpy())
        loss = criterion(outputs, labels)
        optimizer.step()
        running_loss += loss.item()
    return result, running_loss / len(test_loader)

avg_weights=None
def federated_averaging(local_weights):
    global avg_weights
    if avg_weights is None:
        avg_weights = local_weights
        for key in avg_weights.keys(): 
            avg_weights[key] = torch.div(avg_weights[key], 1/(num_clients))
    else:
        for key in avg_weights.keys(): 
            local_weights[key] = torch.div(local_weights[key], 1/(num_clients))
            avg_weights[key] += local_weights[key]
   
    return avg_weights

num_clients = 3
local_epochs = 2
rounds = 4

# Split the combined dataset into distinct subsets for each client
subsets = np.array_split(combined_data, num_clients)

client_model=[]

for round in range(rounds):
    avg_weights = None
    print(f"Round {round+1}/{rounds}")
    round_weights = []
    round_losses = []

    for client in range(num_clients):
        client_data = subsets[client]
        X_client = client_data[features].values
        y_client = client_data['Label'].values.reshape(-1, 1)
        
        # Standardize features for this client
        X_scaled_client = scaler.fit_transform(X_client)
        
        # Split into training and test sets
        X_train_client, X_test_client, y_train_client, y_test_client = train_test_split(
            X_scaled_client, y_client, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train_tensor_client = torch.tensor(X_train_client, dtype=torch.float32)
        y_train_tensor_client = torch.tensor(y_train_client, dtype=torch.float32)
        
        train_dataset_client = TensorDataset(X_train_tensor_client, y_train_tensor_client)
        train_loader_client = DataLoader(train_dataset_client, batch_size=32, shuffle=True)

        client_model.append(TransformerModel(input_dim, nhead, nhid1, nlayers, nclasses, dropout))
        client_optimizer = optim.Adam(client_model[-1].parameters(), lr=0.001)

        # Training starts here for each client
        local_w, local_loss = local_train(client_model[-1], train_loader_client, client_optimizer, criterion, local_epochs)

        global_weights = federated_averaging(local_w)
                
        round_weights.append(local_w)
        round_losses.append(local_loss)
    
    global_model.load_state_dict(avg_weights)
    
    for client in range(num_clients):
        client_model[client].load_state_dict(global_model.state_dict())
        client_optimizer = optim.Adam(client_model[client].parameters(), lr=0.001)
        local_pred, local_loss = local_predict(client_model[client], test_loader, client_optimizer, criterion)
        #print(type(y_test),type(local_pred[0]))
        print(client, "accuracy:",sklm.accuracy_score(np.round(np.array(local_pred)),y_test),"local loss: ",local_loss)
        




    #print(f"Round {round+1} losses:", round_losses)





    # Federated Averaging for this round
    

