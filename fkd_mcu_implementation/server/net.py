import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURAZIONE ARCHITETTURA RETE
# ============================================================================
INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
WEIGHTS_BYTE_SIZE = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE) * 4

class ServerMLP(nn.Module):
    def __init__(self):
        super(ServerMLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# SERIALIZZAZIONE / DESERIALIZZAZIONE BINARIA
# ============================================================================
def deserialize_weights_to_model(binary_payload, model):
    flat_weights = np.frombuffer(binary_payload, dtype=np.float32)
    
    idx = 0
    
    # Layer 1: In C++ è W1[784][32]. Ricostruiamo la matrice 784x32 e poi facciamo la trasposta per PyTorch (32, 784)
    w1_size = INPUT_SIZE * HIDDEN_SIZE
    W1_cpp = flat_weights[idx : idx + w1_size].reshape((INPUT_SIZE, HIDDEN_SIZE))
    W1 = W1_cpp.T
    idx += w1_size
    
    b1_size = HIDDEN_SIZE
    b1 = flat_weights[idx : idx + b1_size]
    idx += b1_size
    
    # Layer 2: In C++ è W2[32][10]. Ricostruiamo la matrice 32x10 e poi facciamo la trasposta per PyTorch (10, 32)
    w2_size = HIDDEN_SIZE * OUTPUT_SIZE
    W2_cpp = flat_weights[idx : idx + w2_size].reshape((HIDDEN_SIZE, OUTPUT_SIZE))
    W2 = W2_cpp.T
    idx += w2_size
    
    b2_size = OUTPUT_SIZE
    b2 = flat_weights[idx : idx + b2_size]

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(W1.copy()))
        model.fc1.bias.copy_(torch.from_numpy(b1.copy()))
        model.fc2.weight.copy_(torch.from_numpy(W2.copy()))
        model.fc2.bias.copy_(torch.from_numpy(b2.copy()))


def serialize_model_to_binary(model):
    state = model.state_dict()
    
    # PyTorch usa la forma (32, 784). Dobbiamo trasporla in (784, 32) prima di appiattirla per C++
    W1 = state['fc1.weight'].cpu().numpy().T.flatten()
    b1 = state['fc1.bias'].cpu().numpy().flatten()
    
    # PyTorch usa la forma (10, 32). Dobbiamo trasporla in (32, 10) prima di appiattirla per C++
    W2 = state['fc2.weight'].cpu().numpy().T.flatten()
    b2 = state['fc2.bias'].cpu().numpy().flatten()
    
    flat_weights = np.concatenate([W1, b1, W2, b2]).astype(np.float32)
    return flat_weights.tobytes()


# ============================================================================
# EVALUATION HELPER
# ============================================================================
def evaluate_model(model, data_loader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, INPUT_SIZE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100