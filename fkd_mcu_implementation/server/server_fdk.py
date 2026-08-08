import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import paho.mqtt.client as mqtt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================================
# CONFIGURAZIONE RETE E MQTT (Multi-Client)
# ============================================================================
MQTT_BROKER = "10.0.0.100"
MQTT_PORT = 1883
TOPIC_COMMAND = "fl/global/command"
TOPIC_GLOBAL_WEIGHTS = "fl/global/global_weights" # Topic unico di broadcast in discesa (con Retain)

# Lista dei client attivi partecipanti al Federated Learning
CLIENT_IDS = ["client_0", "client_1"]

INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
WEIGHTS_BYTE_SIZE = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE) * 4

# Dizionario per raccogliere i pesi binari dei vari client in modo sincronizzato
received_client_weights = {}

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
# CALLBACK MQTT (Gestione dinamica dei topic in salita)
# ============================================================================
def on_message(client, userdata, msg):
    global received_client_weights
    
    # Intercetta i messaggi dai topic del tipo: fl/<client_id>/weights
    if "/weights" in msg.topic and "global" not in msg.topic:
        parts = msg.topic.split("/")
        if len(parts) == 3 and parts[0] == "fl" and parts[2] == "weights":
            client_id = parts[1]
            if client_id in CLIENT_IDS:
                print(f"\n[MQTT] Ricevuti pesi dal client '{client_id}'! Dimensione: {len(msg.payload)} byte")
                if len(msg.payload) == WEIGHTS_BYTE_SIZE:
                    received_client_weights[client_id] = msg.payload
                else:
                    print(f"[ERRORE] Dimensione payload non valida per {client_id}! Attesi {WEIGHTS_BYTE_SIZE}, ricevuti {len(msg.payload)}")

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
        model.fc1.weight.copy_(torch.from_numpy(W1.copy()))
        model.fc1.bias.copy_(torch.from_numpy(b1.copy()))
        model.fc2.weight.copy_(torch.from_numpy(W2.copy()))
        model.fc2.bias.copy_(torch.from_numpy(b2.copy()))

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
# KNOWLEDGE DISTILLATION MULTI-TEACHER (Clients = Teachers, Server = Student)
# ============================================================================
def perform_knowledge_distillation_multi(client_models_dict, server_model, proxy_loader):
    print(f"\n[Server] Avvio Knowledge Distillation Multi-Teacher ({len(client_models_dict)} client collegati) -> Student: Server...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model.to(device)
    for client_id, model in client_models_dict.items():
        model.to(device)
        model.eval() # I modelli dei client fanno da Teacher (bloccati)
        
    server_model.train() # Il server fa da Student
    
    optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    temperature = 3.0
    alpha = 0.5  
    
    for epoch in range(3): 
        running_loss = 0.0
        for inputs, labels in proxy_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, INPUT_SIZE)
            
            optimizer.zero_grad()
            
            # Logit dello Student (Server)
            student_logits = server_model(inputs)
            
            # Calcolo dei logit medi da tutti i teacher (clienti attivi)
            with torch.no_grad():
                teacher_logits_list = [m(inputs) for m in client_models_dict.values()]
                mean_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
                
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            
            soft_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
            soft_teacher = nn.functional.softmax(mean_teacher_logits / temperature, dim=1)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
            
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"[Distillation] Epoca {epoch+1}/3 - Loss Media Server: {running_loss/len(proxy_loader):.4f}")
        
    print("[Server] Distillazione Multi-Teacher completata!")
    
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
    
    # Sottoscrizione dinamica ai topic di tutti i client definiti in CLIENT_IDS
    for cid in CLIENT_IDS:
        topic_w = f"fl/{cid}/weights"
        client.subscribe(topic_w)
        print(f"[Server] Sottoscritto al topic: {topic_w}")
        
    client.loop_start()

    print("[Server] Caricamento del Proxy Dataset (500 campioni da MNIST)...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    proxy_indices = list(range(500))
    proxy_subset = Subset(mnist_test, proxy_indices)
    proxy_loader = DataLoader(proxy_subset, batch_size=32, shuffle=True)

    server_model = ServerMLP() 

    try:
        while True:
            user_input = input("\n>>> Inserisci il numero di round globali per questo esperimento (o premi INVIO per default=1): ")
            total_rounds = int(user_input) if user_input.strip().isdigit() else 1
            
            for r in range(1, total_rounds + 1):
                print(f"\n==================================================")
                print(f"=== INIZIO ROUND FEDERATED GLOBALE {r}/{total_rounds} ===")
                print(f"==================================================")
                
                # Svuota il dizionario dei pesi prima di ogni round
                received_client_weights = {}
                
                # 1. Pubblica il comando di training in broadcast a tutti i client
                print(f"[MQTT] Pubblicazione comando di training sul topic '{TOPIC_COMMAND}'...")
                client.publish(TOPIC_COMMAND, '{"action":"train"}')
                
                print(f"[Server] In attesa dei pesi da tutti i client ({len(CLIENT_IDS)} attesi: {CLIENT_IDS})...")
                
                # Barriera di sincronizzazione: attende finché non arrivano i pesi da TUTTI i client
                while len(received_client_weights) < len(CLIENT_IDS):
                    time.sleep(0.5)
                    
                print("[Server] Tutti i client hanno risposto. Ricostruzione dei modelli locali...")
                
                # 2. Ricostruisci i modelli di tutti i client ricevuti
                client_models_dict = {}
                for client_id in CLIENT_IDS:
                    client_model_recreated = ServerMLP()
                    deserialize_weights_to_model(received_client_weights[client_id], client_model_recreated)
                    client_models_dict[client_id] = client_model_recreated
                print("[Server] Modelli dei client ricreati correttamente in memoria.")
                
                # 3. Knowledge Distillation Multi-Teacher & Stampa Accuracy Globale
                updated_server_model = perform_knowledge_distillation_multi(
                    client_models_dict, server_model, proxy_loader
                )
                
                # 4. Serializza e ridistribuisci la rete globale aggiornata in broadcast (con retain=True)
                new_binary_weights = serialize_model_to_binary(updated_server_model)
                print(f"[MQTT] Distribuzione nuova rete globale in broadcast ({len(new_binary_weights)} byte) sul topic {TOPIC_GLOBAL_WEIGHTS}...")
                client.publish(TOPIC_GLOBAL_WEIGHTS, new_binary_weights, retain=True)
                
                print(f"[Server] Round {r} completato e nuovo modello globale distribuito a tutti i client!")
                
                if r < total_rounds:
                    print("[Server] Pausa di 3 secondi per sincronizzazione...")
                    time.sleep(3)
                    
            print("\n>>> ESPERIMENTO DI FEDERATED LEARNING COMPLETATO CON SUCCESSO! <<<")

    except KeyboardInterrupt:
        print("\n[Server] Chiusura in corso...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()