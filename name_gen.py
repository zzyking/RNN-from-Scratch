import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Data preparation
with open('female.txt', 'r', encoding='utf-8') as f:
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
num_layers = 5

model_male = NameRNN(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_male.to(device)

model_female = NameRNN(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_female.to(device)


def train(model, gender):
    # Instantiate model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with dynamic batch size handling
    num_epochs = 500
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

    torch.save(model.state_dict(), f'{gender}.pth')

if os.path.exists('male.pth'): 
    model_male.load_state_dict(torch.load('male.pth', map_location=device))
else:
    train(model_male, "male")

if os.path.exists('female.pth'): 
    model_female.load_state_dict(torch.load('female.pth', map_location=device))
else:
    train(model_female, "female")


def sample_from_probs(probs, top_k=None, top_p=None, temperature=1.0):
    """
    Sample from a probability distribution with optional top-k, top-p (nucleus), and temperature scaling.
    """
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)
    
    if top_k is not None:
        top_k_idx = np.argsort(probs)[-top_k:]
        probs = np.zeros_like(probs)
        probs[top_k_idx] = 1.0 / top_k
    
    if top_p is not None:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = sorted_probs[np.argmax(cumulative_probs >= top_p)]
        probs[probs < cutoff] = 0.0
        probs = probs / np.sum(probs)
    
    return np.random.choice(len(probs), p=probs)


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



def beam_search(model, start_str, max_length=20, beam_width=5, fixed_positions=None, temperature=1.0):
    """
    Generate a name using Beam Search.
    - beam_width: Number of candidate sequences to maintain.
    """
    model.eval()
    with torch.no_grad():
        # Convert starting string to tensor indices
        start_indices = [char_to_idx[char] for char in start_str]
        initial_input = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Initialize beams: (sequence, log_prob, hidden_state)
        beams = [(
            start_indices.copy(),  # Current sequence
            0.0,  # Log probability of the sequence
            torch.zeros(num_layers, 1, hidden_size, device=device)  # Hidden state
        )]
        
        # Generate name
        for step in range(max_length):
            new_beams = []
            for beam in beams:
                seq, log_prob, hidden = beam
                
                # If the current position is fixed, use the fixed character
                if fixed_positions is not None and step in fixed_positions:
                    fixed_char = fixed_positions[step]
                    fixed_idx = char_to_idx[fixed_char]
                    new_seq = seq + [fixed_idx]
                    new_beams.append((new_seq, log_prob, hidden))
                    continue
                
                # Prepare input
                input_seq = torch.tensor([seq[-1]], dtype=torch.long).unsqueeze(0).to(device)
                input_onehot = F.one_hot(input_seq, num_classes=vocab_size).float().to(device)
                
                # Forward pass
                output, new_hidden = model(input_onehot, hidden)
                output = output[:, -1, :]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    output = output / temperature
                
                # Get log probabilities
                log_probs = torch.log_softmax(output, dim=1).squeeze().cpu().numpy()
                
                # Select top-k candidates
                top_k_indices = np.argsort(log_probs)[-beam_width:]
                for idx in top_k_indices:
                    new_seq = seq + [idx]
                    new_log_prob = log_prob + log_probs[idx]
                    new_beams.append((new_seq, new_log_prob, new_hidden))
            
            # Keep only the top-k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Select the best beam
        best_beam = beams[0]
        best_seq = best_beam[0]
        
        # Convert indices to characters and remove EOS token
        generated_name = ''.join([idx_to_char[idx] for idx in best_seq if idx != char_to_idx[EOS_TOKEN]])
        return generated_name


def generate_name_with_constraints(model, start_str, max_length=20, target_length=None, fixed_positions=None, top_k=None, top_p=None, temperature=1.0, use_beam_search=False, beam_width=5):
    """
    Generate a name with constraints:
    - target_length: Desired length of the generated name (optional).
    - fixed_positions: A dictionary of {position: character} to fix specific characters at specific positions.
    """
    if use_beam_search:
        return beam_search(
            model, 
            start_str, 
            max_length=max_length, 
            beam_width=beam_width, 
            fixed_positions=fixed_positions, 
            temperature=temperature
        )
    else:
        model.eval()
        with torch.no_grad():
            # Convert starting string to tensor indices
            start_indices = [char_to_idx[char] for char in start_str]
            input_seq = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # Initialize hidden state
            hidden = torch.zeros(num_layers, 1, hidden_size, device=device)
            
            # Initialize generated indices with the starting string
            generated_indices = start_indices.copy()
            
            # Generate name
            for step in range(max_length):
                # If the current position is fixed, use the fixed character
                if fixed_positions is not None and step in fixed_positions:
                    fixed_char = fixed_positions[step]
                    fixed_idx = char_to_idx[fixed_char]
                    generated_indices.append(fixed_idx)
                    predicted = torch.tensor([[fixed_idx]], dtype=torch.long).to(device)
                    input_seq = torch.cat((input_seq, predicted), dim=1)
                    continue
                
                # One-hot encode the input sequence
                input_onehot = F.one_hot(input_seq, num_classes=vocab_size).float().to(device)
                
                # Forward pass
                output, hidden = model(input_onehot, hidden)
                output = output[:, -1, :]
                
                # Get probability distribution
                prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                
                # Sample from the probability distribution
                predicted_idx = sample_from_probs(prob, top_k, top_p, temperature)
                generated_indices.append(predicted_idx)
                
                # If EOS token is generated and no target_length is specified, stop
                if predicted_idx == char_to_idx[EOS_TOKEN] and target_length is None:
                    break
                
                # If target_length is specified, continue until the target length is reached
                if target_length is not None:
                    if len(generated_indices) > target_length:
                        generated_indices.pop()
                        break
                    elif predicted_idx == char_to_idx[EOS_TOKEN]: generated_indices.pop()
                
                # Update input sequence
                predicted = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)
                input_seq = torch.cat((input_seq, predicted), dim=1)
            
            # Convert indices to characters and remove EOS token
            generated_name = ''.join([idx_to_char[idx] for idx in generated_indices if idx != char_to_idx[EOS_TOKEN]])
            return generated_name

# 测试生成函数
start_str = 'li'

name_male, probs_male = generate_name(model_male, start_str, max_length=20)
print(f'Completed name (male): {name_male}')

name_female, probs_female = generate_name(model_female, start_str, max_length=20)
print(f'Completed name (female): {name_female}')

# 可视化生成过程

if not os.path.exists('output'):
    os.makedirs('output')

for i, prob in enumerate(probs_male):
    top5_idx = prob.argsort()[-5:][::-1]
    top5_prob = prob[top5_idx]
    top5_chars = [idx_to_char[idx] for idx in top5_idx]
    plt.figure(figsize=(10, 6))
    plt.bar(top5_chars, top5_prob)
    plt.title(f'Time step {i+1+len(start_str)}')
    plt.savefig(f"output/male_{i+1+len(start_str)}.png")

for i, prob in enumerate(probs_female):
    top5_idx = prob.argsort()[-5:][::-1]
    top5_prob = prob[top5_idx]
    top5_chars = [idx_to_char[idx] for idx in top5_idx]
    plt.figure(figsize=(10, 6))
    plt.bar(top5_chars, top5_prob)
    plt.title(f'Time step {i+1+len(start_str)}')
    plt.savefig(f"output/female_{i+1+len(start_str)}.png")