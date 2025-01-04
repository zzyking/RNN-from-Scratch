import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Data preparation
with open('names.txt', 'r', encoding='utf-8') as f:
    names = f.read().splitlines()

# Clean and preprocess data
names = [name.strip().lower() for name in names if name.isalpha()]

# Create character to index and index to character mappings
chars = sorted(list(set(''.join(names))))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

# Add end-of-sequence token
EOS_TOKEN = '<EOS>'
char_to_idx[EOS_TOKEN] = len(chars)
idx_to_char[len(chars)] = EOS_TOKEN
vocab_size += 1

# Convert names to integer sequences with EOS token
names_indices = [[char_to_idx[char] for char in name] + [char_to_idx[EOS_TOKEN]] for name in names]

# Define custom Dataset
class NameDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        if len(seq) < self.seq_length:
            seq += [char_to_idx[EOS_TOKEN]] * (self.seq_length - len(seq))
        else:
            seq = seq[:self.seq_length]
        return torch.tensor(seq, dtype=torch.long)

# Set sequence length and batch size
seq_length = 20
batch_size = 64

# Create DataLoader with pin_memory for GPU transfer
dataset = NameDataset(names_indices, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Define the RNN model
class NameRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NameRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# Model hyperparameters
input_size = vocab_size
hidden_size = 128
output_size = vocab_size
num_layers = 2

model = NameRNN(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if os.path.exists('model.pth'): 
    model.load_state_dict(torch.load('model.pth', map_location=device))
else:
    # Instantiate model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with dynamic batch size handling
    num_epochs = 300
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            batch_size = inputs.shape[0]
            
            # One-hot encoding using F.one_hot
            inputs_onehot = F.one_hot(inputs, num_classes=vocab_size).float().to(device)
            
            # Initialize hidden state
            hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            
            # Forward pass
            outputs, hidden = model(inputs_onehot, hidden)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.pth')

"""
def generate_name(model, start_str, max_length=20):
    model.eval()
    with torch.no_grad():
        # Convert starting string to tensor indices
        start_indices = [char_to_idx[char] for char in start_str]
        input_seq = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Initialize hidden state
        hidden = torch.zeros(num_layers, 1, hidden_size, device=device)
        
        # Generate name
        generated_indices = start_indices.copy()
        prob_dist = []
        for _ in range(max_length):
            # One-hot encode the input sequence
            input_onehot = F.one_hot(input_seq, num_classes=vocab_size).float().to(device)
            
            # Forward pass
            output, hidden = model(input_onehot, hidden)
            output = output[:, -1, :]
            
            # Get probability distribution
            prob = torch.softmax(output, dim=1)
            prob_dist.append(prob.squeeze().cpu().numpy())
            
            # Select the most probable character
            _, predicted = torch.topk(prob, k=1)
            index = torch.randint(0, predicted.size(1), (1,))
            choice = predicted[0, index]
            predicted_idx = choice.item()
            generated_indices.append(predicted_idx)
            
            # If EOS token is generated, stop
            if predicted_idx == char_to_idx[EOS_TOKEN]:
                break
            
            # Update input sequence
            predicted = predicted.to(device)
            input_seq = torch.cat((input_seq, predicted), dim=1)
        
        # Convert indices to characters and remove EOS token
        generated_name = ''.join([idx_to_char[idx] for idx in generated_indices if idx != char_to_idx[EOS_TOKEN]])
        return generated_name, prob_dist
"""

def generate_name(model, start_str, max_length=20, decode_method='top-k', k=5, p=0.9, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Convert starting string to tensor indices
        start_indices = [char_to_idx[char] for char in start_str]
        input_seq = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Initialize hidden state
        hidden = torch.zeros(num_layers, 1, hidden_size, device=device)
        
        # Generate name
        generated_indices = start_indices.copy()
        prob_dist = []
        for _ in range(max_length):
            # One-hot encode the input sequence
            input_onehot = F.one_hot(input_seq, num_classes=vocab_size).float().to(device)
            
            # Forward pass
            output, hidden = model(input_onehot, hidden)
            output = output[:, -1, :]
            
            # Get probability distribution
            prob = torch.softmax(output, dim=1)
            prob_dist.append(prob.squeeze().cpu().numpy())
            
            if decode_method == 'top-k':
                # Top-k decoding
                top_k_prob, top_k_idx = torch.topk(prob, k=k)
                # Sample from top-k
                choice = torch.multinomial(top_k_prob.squeeze(), num_samples=1)
                predicted_idx = top_k_idx.squeeze()[choice].item()
            elif decode_method == 'temperature':
                # Temperature sampling
                prob_temp = prob.pow(1 / temperature)
                prob_temp = prob_temp / prob_temp.sum()
                predicted_idx = torch.multinomial(prob_temp.squeeze(), num_samples=1).item()
            elif decode_method == 'nucleus':
                # Nucleus (top-p) sampling
                sorted_probs, sorted_indices = torch.sort(prob, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=1)
                sorted_indices = sorted_indices.squeeze()
                cumulative_probs = cumulative_probs.squeeze()
                # Create a mask where cumulative probabilities exceed p
                mask = cumulative_probs >= p
                # Find the first index where the mask is True
                idx = torch.argmax(mask.int()) if torch.any(mask) else len(cumulative_probs)-1
                top_p_probs = sorted_probs[0, :idx+1]
                top_p_indices = sorted_indices[:idx+1]
                # Normalize probabilities
                top_p_probs = top_p_probs / top_p_probs.sum()
                # Sample from top-p
                predicted_idx = torch.multinomial(top_p_probs, num_samples=1).item()
            else:
                # Default to top-k decoding
                top_k_prob, top_k_idx = torch.topk(prob, k=k)
                choice = torch.multinomial(top_k_prob.squeeze(), num_samples=1)
                predicted_idx = top_k_idx.squeeze()[choice].item()
            
            generated_indices.append(predicted_idx)
            
            # If EOS token is generated, stop
            if predicted_idx == char_to_idx[EOS_TOKEN]:
                break
            
            # Update input sequence
            predicted = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)
            input_seq = torch.cat((input_seq, predicted), dim=1)
        
        # Convert indices to characters and remove EOS token
        generated_name = ''.join([idx_to_char[idx] for idx in generated_indices if idx != char_to_idx[EOS_TOKEN]])
        return generated_name, prob_dist

# Generate names using different decoding methods
strs = "abcdefghijklmnopqrstuvwxyz"

for start_str in strs:
    # Generate a name with top-k decoding
    name_topk, probs_topk = generate_name(model, start_str, max_length=20, decode_method='top-k', k=5)
    print(f'Generated name (top-k): {name_topk}')
    
    # Generate a name with temperature sampling
    name_temp, probs_temp = generate_name(model, start_str, max_length=20, decode_method='temperature', temperature=0.7)
    print(f'Generated name (temperature): {name_temp}')
    
    # Generate a name with nucleus (top-p) sampling
    name_nucleus, probs_nucleus = generate_name(model, start_str, max_length=20, decode_method='nucleus', p=0.8)
    print(f'Generated name (nucleus): {name_nucleus}')

# Visualize the top 5 candidate characters at each time step
for i, prob in enumerate(probs_topk):
    top5_idx = prob.argsort()[-5:][::-1]
    top5_prob = prob[top5_idx]
    top5_chars = [idx_to_char[idx] for idx in top5_idx]
    plt.figure(figsize=(6, 4))
    plt.bar(top5_chars, top5_prob)
    plt.title(f'Time step {i+1}')
    plt.savefig(f"topk_{i+1}.png")

for i, prob in enumerate(probs_temp):
    top5_idx = prob.argsort()[-5:][::-1]
    top5_prob = prob[top5_idx]
    top5_chars = [idx_to_char[idx] for idx in top5_idx]
    plt.figure(figsize=(6, 4))
    plt.bar(top5_chars, top5_prob)
    plt.title(f'Time step {i+1}')
    plt.savefig(f"temp_{i+1}.png")

for i, prob in enumerate(probs_nucleus):
    top5_idx = prob.argsort()[-5:][::-1]
    top5_prob = prob[top5_idx]
    top5_chars = [idx_to_char[idx] for idx in top5_idx]
    plt.figure(figsize=(6, 4))
    plt.bar(top5_chars, top5_prob)
    plt.title(f'Time step {i+1}')
    plt.savefig(f"nucleus_{i+1}.png")