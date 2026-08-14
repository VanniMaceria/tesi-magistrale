import time
import paho.mqtt.client as mqtt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from net import ServerMLP, deserialize_weights_to_model, serialize_model_to_binary, WEIGHTS_BYTE_SIZE
from distillation import KnowledgeDistillationManager

# ============================================================================
# CONFIGURAZIONE RETE E MQTT (Multi-Client)
# ============================================================================
MQTT_BROKER = "10.0.0.104"
MQTT_PORT = 1883
TOPIC_COMMAND = "fl/global/command"
TOPIC_GLOBAL_WEIGHTS = "fl/global/global_weights" # Topic unico di broadcast in discesa (con Retain)

# Lista dei client attivi partecipanti al Federated Learning
CLIENT_IDS = ["client_0",]

# Dizionario per raccogliere i pesi binari dei vari client in modo sincronizzato
received_client_weights = {}

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

def main():
    global received_client_weights
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FL_Server_Coordinator")
    client.on_message = on_message # Callback per la ricezione dei messaggi dai client
    
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
    kd_manager = KnowledgeDistillationManager(temperature=3.0, alpha=0.5, epochs=3)

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
                updated_server_model = kd_manager.perform_knowledge_distillation_multi(
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