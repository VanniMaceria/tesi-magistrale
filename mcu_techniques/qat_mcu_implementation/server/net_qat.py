import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architettura MLP (deve coincidere con model_weights.h dello sketch)
INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10

# ============================================================================
# FORMATO PAYLOAD INT8 (identico allo sketch esp32_qat.ino, little-endian)
#   W1 int8 (784x32 [i][j]) | W2 int8 (32x10 [j][k]) | b1 f32 (32) | b2 f32 (10)
#   | s_w1, s_w2, s_a1, s_a2 (f32)
# ============================================================================
N_W1 = INPUT_SIZE * HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE
N_BIAS = HIDDEN_SIZE + OUTPUT_SIZE
N_SCALES = 4
FLOAT_BYTES = np.dtype(np.float32).itemsize   # 4, equivalente di sizeof(float)

QAT_BYTE_SIZE = N_W1 + N_W2 + (N_BIAS + N_SCALES) * FLOAT_BYTES   # 25.592
FP32_BYTE_SIZE = (N_W1 + N_W2 + N_BIAS) * FLOAT_BYTES             # 101.800

QMIN, QMAX = -128, 127
INPUT_SCALE = 1.0 / 127.0


def fake_quant(x, scale):
    """Round-to-nearest-even + clipping in [-128, 127], poi dequantizzazione."""
    return torch.clamp(torch.round(x / scale), QMIN, QMAX) * scale


def weight_scale(w):
    return torch.clamp(w.abs().max(), min=1e-8) / 127.0


class QATServerMLP(nn.Module):
    """Stesso MLP 784-32-10 di ServerMLP, con forward int8 fake-quant identico al client ESP32.
    I pesi fc1/fc2 sono i latenti float32; s_a1/s_a2 sono le scale (observer) delle attivazioni."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.register_buffer("s_a1", torch.tensor(1.0 / 127.0))
        self.register_buffer("s_a2", torch.tensor(1.0 / 127.0))

    def forward(self, x, quantized=True):
        if not quantized:
            return self.fc2(torch.relu(self.fc1(x)))
        x = fake_quant(x, INPUT_SCALE)                                   # QuantStub
        w1 = fake_quant(self.fc1.weight, weight_scale(self.fc1.weight))
        h = torch.relu(F.linear(x, w1, self.fc1.bias))
        h = fake_quant(h, self.s_a1)
        w2 = fake_quant(self.fc2.weight, weight_scale(self.fc2.weight))
        logits = F.linear(h, w2, self.fc2.bias)
        return fake_quant(logits, self.s_a2)                             # -> DeQuantStub


# ============================================================================
# SERIALIZZAZIONE / DESERIALIZZAZIONE INT8
# ============================================================================
def _quantize_np(w):
    s = max(float(np.abs(w).max()), 1e-8) / 127.0
    q = np.clip(np.rint(w / s), QMIN, QMAX).astype(np.int8)
    return q, np.float32(s)


def serialize_model_to_int8(model):
    sd = model.state_dict()
    W1 = sd["fc1.weight"].cpu().numpy().T.copy()    # (784, 32) come in C++
    W2 = sd["fc2.weight"].cpu().numpy().T.copy()    # (32, 10)
    q1, s1 = _quantize_np(W1)
    q2, s2 = _quantize_np(W2)
    b1 = sd["fc1.bias"].cpu().numpy().astype(np.float32)
    b2 = sd["fc2.bias"].cpu().numpy().astype(np.float32)
    scales = np.array([s1, s2, float(sd["s_a1"]), float(sd["s_a2"])], dtype=np.float32)
    return (q1.tobytes() + q2.tobytes() + b1.tobytes() + b2.tobytes() + scales.tobytes())


def deserialize_int8_to_model(payload, model):
    assert len(payload) == QAT_BYTE_SIZE, f"payload {len(payload)} != {QAT_BYTE_SIZE}"
    o = 0
    q1 = np.frombuffer(payload, dtype=np.int8, count=N_W1, offset=o).reshape(INPUT_SIZE, HIDDEN_SIZE); o += N_W1
    q2 = np.frombuffer(payload, dtype=np.int8, count=N_W2, offset=o).reshape(HIDDEN_SIZE, OUTPUT_SIZE); o += N_W2
    b1 = np.frombuffer(payload, dtype=np.float32, count=HIDDEN_SIZE, offset=o); o += HIDDEN_SIZE * 4
    b2 = np.frombuffer(payload, dtype=np.float32, count=OUTPUT_SIZE, offset=o); o += OUTPUT_SIZE * 4
    sc = np.frombuffer(payload, dtype=np.float32, count=4, offset=o)

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy((q1.astype(np.float32) * sc[0]).T.copy()))
        model.fc2.weight.copy_(torch.from_numpy((q2.astype(np.float32) * sc[1]).T.copy()))
        model.fc1.bias.copy_(torch.from_numpy(b1.copy()))
        model.fc2.bias.copy_(torch.from_numpy(b2.copy()))
        model.s_a1.fill_(float(sc[2]))
        model.s_a2.fill_(float(sc[3]))


# ============================================================================
# FEDAVG SUI PESI LATENTI (+ media delle scale delle attivazioni)
# ============================================================================
def fedavg(models_dict, n_samples=None):
    ids = list(models_dict.keys())
    if n_samples is None:
        w = {cid: 1.0 / len(ids) for cid in ids}
    else:
        tot = float(sum(n_samples[c] for c in ids))
        w = {cid: n_samples[cid] / tot for cid in ids}

    global_model = QATServerMLP()
    avg_state = {}
    for key in global_model.state_dict().keys():
        avg_state[key] = sum(models_dict[cid].state_dict()[key].float() * w[cid] for cid in ids)
    global_model.load_state_dict(avg_state)
    return global_model


# ============================================================================
# VALUTAZIONE
# ============================================================================
def evaluate_model_qat(model, data_loader, quantized=True):
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device).view(-1, INPUT_SIZE), labels.to(device)
            pred = model(inputs, quantized=quantized).argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total