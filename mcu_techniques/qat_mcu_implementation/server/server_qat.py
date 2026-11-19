import time
import paho.mqtt.client as mqtt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from net_qat import (QATServerMLP, deserialize_int8_to_model, serialize_model_to_int8,
                     fedavg, evaluate_model_qat, QAT_BYTE_SIZE, FP32_BYTE_SIZE)

# ============================================================================
# CONFIGURAZIONE RETE E MQTT (Multi-Client) - topic separati dalla distillazione
# ============================================================================
MQTT_BROKER = "10.59.21.81"
MQTT_PORT = 1883
TOPIC_COMMAND = "fl_qat/global/command"
TOPIC_GLOBAL_WEIGHTS = "fl_qat/global/global_weights"   # broadcast in discesa (retain)

CLIENT_IDS = ["client_0", "client_1"]
ROUND_TIMEOUT_S = 900

received_client_weights = {}


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[Server] Connesso al broker MQTT con codice: {reason_code}")
    for cid in CLIENT_IDS:
        topic_w = f"fl_qat/{cid}/weights"
        client.subscribe(topic_w)
        print(f"[Server] Sottoscrizione a: {topic_w}")


def on_message(client, userdata, msg):
    parts = msg.topic.split("/")
    if len(parts) == 3 and parts[0] == "fl_qat" and parts[2] == "weights" and parts[1] in CLIENT_IDS:
        client_id = parts[1]
        print(f"\n[MQTT] Ricevuti pesi INT8 dal client '{client_id}'! Dimensione: {len(msg.payload)} byte")
        if len(msg.payload) == QAT_BYTE_SIZE:
            received_client_weights[client_id] = msg.payload
        else:
            print(f"[ERRORE] Payload non valido per {client_id}! Attesi {QAT_BYTE_SIZE}, ricevuti {len(msg.payload)}")


def main():
    global received_client_weights

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FL_QAT_Server_Coordinator")
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[Server] Connessione al broker MQTT {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # Cancella un eventuale modello globale retained di esperimenti precedenti
    client.publish(TOPIC_GLOBAL_WEIGHTS, b"", retain=True)

    print("[Server] Caricamento MNIST (Test Set globale disgiunto dai client: indici 2500-9999)...")
    mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_subset = Subset(mnist_test, list(range(2500, len(mnist_test))))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    print(f"[Server] Test Set = {len(test_subset)} campioni")
    print(f"[Server] Payload per direzione: {QAT_BYTE_SIZE} B (INT8) vs {FP32_BYTE_SIZE} B (FP32): "
          f"-{100 * (1 - QAT_BYTE_SIZE / FP32_BYTE_SIZE):.1f}%")

    try:
        while True:
            user_input = input("\n>>> Inserisci il numero di round globali (INVIO per default=1): ")
            total_rounds = int(user_input) if user_input.strip().isdigit() else 1

            for r in range(1, total_rounds + 1):
                print("\n==================================================")
                print(f"=== INIZIO ROUND FEDERATED QAT {r}/{total_rounds} ===")
                print("==================================================")

                received_client_weights = {}
                print(f"[MQTT] Comando di training su '{TOPIC_COMMAND}'...")
                client.publish(TOPIC_COMMAND, '{"action":"train"}')

                t0 = time.time()
                while len(received_client_weights) < len(CLIENT_IDS):
                    if time.time() - t0 > ROUND_TIMEOUT_S:
                        raise TimeoutError(f"Timeout round: ricevuti solo {list(received_client_weights)}")
                    time.sleep(0.5)

                print("[Server] Tutti i client hanno risposto. Dequantizzazione e FedAvg sui latenti...")
                client_models = {}
                for cid in CLIENT_IDS:
                    m = QATServerMLP()
                    deserialize_int8_to_model(received_client_weights[cid], m)
                    client_models[cid] = m
                    print(f"   - {cid}: acc int8 sul test globale = {evaluate_model_qat(m, test_loader):.2f}%")

                global_model = fedavg(client_models)

                acc_q = evaluate_model_qat(global_model, test_loader, quantized=True)
                acc_f = evaluate_model_qat(global_model, test_loader, quantized=False)
                print(f"\n >>> [ACCURACY GLOBALE ({len(test_subset)} campioni)] INT8 fake-quant: {acc_q:.2f}% | "
                      f"float32: {acc_f:.2f}% <<<\n")

                new_binary = serialize_model_to_int8(global_model)
                print(f"[MQTT] Broadcast modello globale INT8 ({len(new_binary)} byte) su {TOPIC_GLOBAL_WEIGHTS}...")
                client.publish(TOPIC_GLOBAL_WEIGHTS, new_binary, retain=True)
                print(f"[Server] Round {r} completato.")

                if r < total_rounds:
                    time.sleep(3)

            print("\n>>> ESPERIMENTO FEDERATED QAT COMPLETATO! <<<")

    except KeyboardInterrupt:
        print("\n[Server] Chiusura in corso...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[Server] Disconnessione completata.")


if __name__ == "__main__":
    main()
