import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import paho.mqtt.client as mqtt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================================
# CONFIGURAZIONE RETE E MQTT
# ============================================================================
MQTT_BROKER = "10.0.0.101"
MQTT_PORT = 1883
TOPIC_COMMAND = "fl/global/command"
TOPIC_WEIGHTS = "fl/client_0/weights"
TOPIC_GLOBAL_WEIGHTS = "fl/client_0/global_weights"

INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
WEIGHTS_BYTE_SIZE = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE) * 4

received_client_weights = None

# ============================================================================
# DEFINIZIONE ARCHITETTURA MODELLO (Identica all'ESP32)
# ============================================================================
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
# CALLBACK MQTT
# ============================================================================
def on_message(client, userdata, msg):
    global received_client_weights
    if msg.topic == TOPIC_WEIGHTS:
        print(f"\n[MQTT] Ricevuti pesi dal client! Dimensione: {len(msg.payload)} byte")
        if len(msg.payload) == WEIGHTS_BYTE_SIZE:
            received_client_weights = msg.payload
        else:
            print(f"[ERRORE] Dimensione payload non valida! Attesi {WEIGHTS_BYTE_SIZE}, ricevuti {len(msg.payload)}")

# ============================================================================
# SERIALIZZAZIONE / DESERIALIZZAZIONE BINARIA
# ============================================================================
def deserialize_weights_to_model(binary_payload, model):
    flat_weights = np.frombuffer(binary_payload, dtype=np.float32)
    
    idx = 0
    w1_size = INPUT_SIZE * HIDDEN_SIZE
    W1 = flat_weights[idx : idx + w1_size].reshape((HIDDEN_SIZE, INPUT_SIZE))
    idx += w1_size
    
    b1_size = HIDDEN_SIZE
    b1 = flat_weights[idx : idx + b1_size]
    idx += b1_size
    
    w2_size = HIDDEN_SIZE * OUTPUT_SIZE
    W2 = flat_weights[idx : idx + w2_size].reshape((OUTPUT_SIZE, HIDDEN_SIZE))
    idx += w2_size
    
    b2_size = OUTPUT_SIZE
    b2 = flat_weights[idx : idx + b2_size]

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(W1))
        model.fc1.bias.copy_(torch.from_numpy(b1))
        model.fc2.weight.copy_(torch.from_numpy(W2))
        model.fc2.bias.copy_(torch.from_numpy(b2))

def serialize_model_to_binary(model):
    state = model.state_dict()
    W1 = state['fc1.weight'].cpu().numpy().flatten()
    b1 = state['fc1.bias'].cpu().numpy().flatten()
    W2 = state['fc2.weight'].cpu().numpy().flatten()
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

# ============================================================================
# KNOWLEDGE DISTILLATION (Client = Teacher, Server = Student)
# ============================================================================
def perform_knowledge_distillation(client_model, server_model, proxy_loader):
    print("\n[Server] Avvio Knowledge Distillation (Teacher: Client ESP32, Student: Server)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.to(device)
    server_model.to(device)
    
    client_model.eval()
    server_model.train()
    
    optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    temperature = 3.0
    alpha = 0.5  
    
    for epoch in range(3): 
        running_loss = 0.0
        for inputs, labels in proxy_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, INPUT_SIZE)
            
            optimizer.zero_grad()
            
            student_logits = server_model(inputs)
            with torch.no_grad():
                teacher_logits = client_model(inputs)
                
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            
            soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
            soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=1)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
            
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"[Distillation] Epoca {epoch+1}/3 - Loss Media Server: {running_loss/len(proxy_loader):.4f}")
        
    print("[Server] Distillazione completata!")
    
    # Calcolo accuracy finale del modello globale (server) sul proxy dataset
    global_accuracy = evaluate_model(server_model, proxy_loader)
    print(f"\n >>> [ACCURACY MODELLO GLOBALE SERVER] -> {global_accuracy:.2f}% <<<\n")
    
    return server_model

# ============================================================================
# SCRIPT PRINCIPALE
# ============================================================================
def main():
    global received_client_weights
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FL_Server_Coordinator")
    client.on_message = on_message
    
    print(f"[Server] Connessione al broker MQTT {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_WEIGHTS)
    client.loop_start()

    print("[Server] Caricamento del Proxy Dataset (500 campioni da MNIST)...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    proxy_indices = list(range(500))
    proxy_subset = Subset(mnist_test, proxy_indices)
    proxy_loader = DataLoader(proxy_subset, batch_size=32, shuffle=True)

    client_model_recreated = ServerMLP()
    server_model = ServerMLP() 

    try:
        while True:
            # Input utente per i round globali
            user_input = input("\n>>> Inserisci il numero di round globali per questo esperimento (o premi INVIO per default=1): ")
            total_rounds = int(user_input) if user_input.strip().isdigit() else 1
            
            for r in range(1, total_rounds + 1):
                print(f"\n==================================================")
                print(f"=== INIZIO ROUND FEDERATO GLOBALE {r}/{total_rounds} ===")
                print(f"==================================================")
                
                # 1. Pubblica il comando di training all'ESP32
                print(f"[MQTT] Pubblicazione comando di training all'ESP32...")
                client.publish(TOPIC_COMMAND, '{"action":"train"}')
                
                print("[Server] In attesa dei pesi dal client ESP32-S3...")
                received_client_weights = None
                
                while received_client_weights is None:
                    time.sleep(0.5)
                    
                # 2. Ricostruisci il modello del client
                deserialize_weights_to_model(received_client_weights, client_model_recreated)
                print("[Server] Modello del client ricreato correttamente in memoria.")
                
                # 3. Knowledge Distillation & Stampa Accuracy Globale
                updated_server_model = perform_knowledge_distillation(
                    client_model_recreated, server_model, proxy_loader
                )
                
                # 4. Serializza e ridistribuisci la rete aggiornata del server
                new_binary_weights = serialize_model_to_binary(updated_server_model)
                print(f"[MQTT] Distribuzione nuova rete globale ({len(new_binary_weights)} byte) sul topic {TOPIC_GLOBAL_WEIGHTS}...")
                client.publish(TOPIC_GLOBAL_WEIGHTS, new_binary_weights)
                print(f"[Server] Round {r} completato e nuovo modello globale distribuito!")
                
                # Pausa di sicurezza per dare tempo all'ESP32 di ricevere i pesi via MQTT prima del round successivo
                if r < total_rounds:
                    print("[Server] Pausa di 3 secondi per sincronizzazione client...")
                    time.sleep(3)
                    
            print("\n>>> ESPERIMENTO DI FEDERATED LEARNING COMPLETATO CON SUCCESSO! <<<")

    except KeyboardInterrupt:
        print("\n[Server] Chiusura in corso...")
        client.loop_stop()
        client.disconnect()
if __name__ == "__main__":
    main()