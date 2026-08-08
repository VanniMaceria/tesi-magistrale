#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoMqttClient.h>
#include <ArduinoJson.h>
#include "mnist_flash_dataset.h" // Dataset MNIST allocato in Flash (PROGMEM)
#include "model_weights.h"       // Definizione dell'architettura e pesi iniziali
#include "wifi_credentials.h"

// ============================================================================
// CONFIGURAZIONE RETE E BROKER MQTT
// ============================================================================
const char* MQTT_SERVER   = "10.0.0.100"; // IP Broker MQTT / Server FL
const int   MQTT_PORT     = 1883;

const char* CLIENT_ID     = "client_1"; 

const char* TOPIC_COMMAND        = "fl/global/command";
const char* TOPIC_GLOBAL_WEIGHTS = "fl/global/global_weights"; // Topic unico condiviso in discesa
char TOPIC_WEIGHTS[64];                                      // Topic dinamico in salita

#define TOTAL_WEIGHTS_FLOATS 25450
#define WEIGHTS_BYTE_SIZE    (TOTAL_WEIGHTS_FLOATS * sizeof(float))

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

// ============================================================================
// 1. IPERPARAMETRI DI ADDESTRAMENTO E ARCHITETTURA DELLA RETE
// ============================================================================
#define LEARNING_RATE 0.03f  // Tasso di apprendimento per l'aggiornamento SGD
#define LOCAL_EPOCHS  10     // Epoche locali
float current_lr = LEARNING_RATE;

// Pesi e Bias allocati in memoria RAM dinamica (modificabili durante il Gradient Descent)
// Architettura MLP: 784 ingressi (28x28 pixel) -> 32 neuroni nascosti -> 10 classi di output
float W1[INPUT_SIZE][HIDDEN_SIZE];   // Matrice pesi Layer 1 (784 x 32)
float b1[HIDDEN_SIZE];              // Vettore bias Layer 1 (32)
float W2[HIDDEN_SIZE][OUTPUT_SIZE];  // Matrice pesi Layer 2 (32 x 10)
float b2[OUTPUT_SIZE];              // Vettore bias Layer 2 (10)

// Buffer temporanei per conservare i valori del Forward Pass (RAM)
float hidden_activations[HIDDEN_SIZE]; // Attivazioni ReLU dello strato nascosto
float output_probs[OUTPUT_SIZE];       // Probabilità uscenti Softmax dallo strato di output

// Buffer per i gradienti calcolati durante la Backpropagation (RAM)
float d_output[OUTPUT_SIZE]; // Gradiente dell'errore rispetto ai logits di output
float d_hidden[HIDDEN_SIZE]; // Gradiente dell'errore retropropagato allo strato nascosto

// ============================================================================
// 2. INIZIALIZZAZIONE DEI PESI CON ROMPIMENTO DELLA SIMMETRIA (Symmetry Breaking)
// ============================================================================
// Copia i pesi dalla Flash alla RAM e aggiunge un leggero disturbo casuale (jitter)
// per evitare che i neuroni calcolino lo stesso gradiente durante l'addestramento.
void init_weights_with_random_jitter() {
  srand(42); // Seed fisso per la generazione casuale per garantire la riproducibilità
  
  for (int i = 0; i < INPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      float jitter = ((float)rand() / RAND_MAX - 0.5f) * 0.10f; // Valore casuale tra -0.05 e +0.05
      W1[i][j] = init_W1[i][j] + jitter;
    }
  }
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    b1[j] = init_b1[j];
    for (int k = 0; k < OUTPUT_SIZE; k++) {
      float jitter = ((float)rand() / RAND_MAX - 0.5f) * 0.10f;
      W2[j][k] = init_W2[j][k] + jitter;
    }
  }
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    b2[k] = init_b2[k];
  }
}

// ============================================================================
// 3. FUNZIONI MATEMATICHE DI ATTIVAZIONE E DERIVATE
// ============================================================================

// Rectified Linear Unit (ReLU): azzera i valori negativi, lascia inalterati i positivi
inline float relu(float x) { 
  return x > 0.0f ? x : 0.0f; 
}

// Derivata della ReLU per la Backpropagation: 1 se x > 0, altrimenti 0
inline float relu_derivative(float x) { 
  return x > 0.0f ? 1.0f : 0.0f; 
}

// Softmax con sottrazione del valore massimo per stabilità numerica (previene l'overflow di exp)
void softmax(float* input, float* output, int size) {
  float max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val) max_val = input[i];
  }
  
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }
  for (int i = 0; i < size; i++) {
    output[i] /= sum; // Normalizzazione per far sì che la somma delle probabilità sia uguale a 1.0
  }
}

// ============================================================================
// 4. FORWARD PASS (Inferenza Locale)
// ============================================================================
void forward(const uint8_t* image_bytes) {
  // Step A: Input Layer -> Hidden Layer (Moltiplicazione Matrice-Vettore + ReLU)
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    float sum = b1[j];
    for (int i = 0; i < INPUT_SIZE; i++) {
      float input_val = (float)image_bytes[i] / 255.0f; // Normalizzazione dei pixel nell'intervallo [0.0, 1.0]
      sum += input_val * W1[i][j];
    }
    hidden_activations[j] = relu(sum);
  }

  // Step B: Hidden Layer -> Output Logits
  float logits[OUTPUT_SIZE];
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    float sum = b2[k];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += hidden_activations[j] * W2[j][k];
    }
    logits[k] = sum;
  }

  // Step C: Softmax per ottenere la distribuzione di probabilità sulle 10 classi
  softmax(logits, output_probs, OUTPUT_SIZE);
}

// ============================================================================
// 5. BACKWARD PASS & AGGIORNAMENTO PESI (Stochastic Gradient Descent)
// ============================================================================
float train_single_sample(const uint8_t* image_bytes, uint8_t label) {
  // 1. Esecuzione Forward Pass per aggiornare le attivazioni
  forward(image_bytes);

  // 2. Calcolo Cross-Entropy Loss: L = -log(P_target)
  float loss = -logf(output_probs[label] + 1e-7f); // Addizione di 1e-7 per evitare log(0)

  // 3. Calcolo del gradiente per lo strato di Output: dL/dLogits = Probs - Target_One_Hot
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    float target = (k == label) ? 1.0f : 0.0f;
    d_output[k] = output_probs[k] - target;
  }

  // 4. Retropropagazione dell'errore allo strato Nascosto (Chain Rule / Regola della catena)
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    float grad_sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; k++) {
      grad_sum += d_output[k] * W2[j][k];
    }
    // Applicazione della derivata della ReLU per applicare il gradiente al livello nascosto
    d_hidden[j] = grad_sum * relu_derivative(hidden_activations[j]);
  }

  // 5. Aggiornamento Pesi W2 e Bias b2 tramite SGD (Batch Size = 1)
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      W2[j][k] -= current_lr * d_output[k] * hidden_activations[j];
    }
    b2[k] -= current_lr * d_output[k];
  }

  // 6. Aggiornamento Pesi W1 e Bias b1 tramite SGD
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    for (int i = 0; i < INPUT_SIZE; i++) {
      float input_val = (float)image_bytes[i] / 255.0f;
      W1[i][j] -= current_lr * d_hidden[j] * input_val;
    }
    b1[j] -= current_lr * d_hidden[j];
  }

  return loss;
}

// ============================================================================
// 6. VALUTAZIONE DELL'ACCURATEZZA SUL TEST SET SEPARATO
// ============================================================================
float evaluate_test_set() {
  int correct = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    forward(test_images[i]); // Infezzione (solamente Forward pass)
    
    // Ricerca della classe predetta (Argmax)
    int pred = 0;
    float max_p = output_probs[0];
    for (int k = 1; k < OUTPUT_SIZE; k++) {
      if (output_probs[k] > max_p) {
        max_p = output_probs[k];
        pred = k;
      }
    }
    // Incremento delle predizioni giuste
    if (pred == test_labels[i]) {
      correct++;
    }
  }
  return ((float)correct / NUM_TEST_SAMPLES) * 100.0f;
}

// ============================================================================
// ESECUZIONE DEL CICLO DI ADDESTRAMENTO
// ============================================================================
void run_local_training(int epochs, float lr) {
  current_lr = lr;
  unsigned long start_time_ms = millis(); // Cronometro per la profilazione dell'addestramento

  // Ciclo principale di addestramento su più epoche locali
  for (int epoch = 1; epoch <= epochs; epoch++) {
    float epoch_loss = 0.0f;
    
    // Ciclo sui 200 campioni di training disponibili in Flash
    for (int n = 0; n < NUM_TRAIN_SAMPLES; n++) {
      float l = train_single_sample(train_images[n], train_labels[n]);
      epoch_loss += l;
    }
    
    // Log di progresso dell'epoca
    Serial.printf("[EPOCH %2d/%d] Loss Media Training: %.4f | Current Test Acc: %.2f%%\n", 
                  epoch, epochs, epoch_loss / NUM_TRAIN_SAMPLES, evaluate_test_set());
  }

  unsigned long elapsed_time_ms = millis() - start_time_ms;
  float final_acc = evaluate_test_set();

  Serial.println("-------------------------------------------------");
  Serial.printf("Tempo Totale Training (%d epoche x %d campioni): %lu ms (%.2f s)\n", 
                epochs, NUM_TRAIN_SAMPLES, elapsed_time_ms, elapsed_time_ms / 1000.0f);
  Serial.printf("Tempo Medio per Epoca:      %.2f ms\n", (float)elapsed_time_ms / epochs);
  Serial.printf("Accuracy FINALE Test Set:   %.2f%%\n", final_acc);
  Serial.println("-------------------------------------------------");
}

// ============================================================================
// TRASMISSIONE BINARIA DEI PESI VIA MQTT
// ============================================================================
void send_weights_mqtt() {
  float* flat_weights = (float*) malloc(WEIGHTS_BYTE_SIZE);
  if (flat_weights == NULL) {
    Serial.println("[MQTT Error] Allocazione RAM fallita per l'invio pesi!");
    return;
  }

  // Serializzazione dei blocchi della rete
  memcpy(flat_weights, W1, sizeof(W1));
  size_t offset = (INPUT_SIZE * HIDDEN_SIZE);
  
  memcpy(flat_weights + offset, b1, sizeof(b1));
  offset += HIDDEN_SIZE;
  
  memcpy(flat_weights + offset, W2, sizeof(W2));
  offset += (HIDDEN_SIZE * OUTPUT_SIZE);
  
  memcpy(flat_weights + offset, b2, sizeof(b2));

  const uint8_t* binary_payload = reinterpret_cast<const uint8_t*>(flat_weights);

  Serial.printf("[MQTT] Invio pesi (%d byte) sul topic %s...\n", WEIGHTS_BYTE_SIZE, TOPIC_WEIGHTS);
  
  // Invio in streaming con ArduinoMqttClient sul topic dedicato di questo client
  mqttClient.beginMessage(TOPIC_WEIGHTS, WEIGHTS_BYTE_SIZE, false, 0, false);
  mqttClient.write(binary_payload, WEIGHTS_BYTE_SIZE);
  bool success = mqttClient.endMessage();

  if (success) {
    Serial.println("[MQTT] Pesi inviati correttamente al server!");
  } else {
    Serial.println("[MQTT Error] Errore nell'invio del messaggio MQTT.");
  }

  free(flat_weights);
}

// ============================================================================
// RICEZIONE E SCRITTURA DEI PESI GLOBALI DALLO STREAM BINARIO MQTT
// ============================================================================
void receive_global_weights(int messageSize) {
  if (messageSize != WEIGHTS_BYTE_SIZE) {
    Serial.printf("[MQTT Error] Dimensione pesi globali errata! Attesi %d byte, ricevuti %d\n", WEIGHTS_BYTE_SIZE, messageSize);
    while (mqttClient.available()) { mqttClient.read(); }
    return;
  }

  float* flat_weights = (float*) malloc(WEIGHTS_BYTE_SIZE);
  if (flat_weights == NULL) {
    Serial.println("[MQTT Error] Allocazione RAM fallita per la ricezione pesi globali!");
    return;
  }

  int bytesRead = mqttClient.read((uint8_t*)flat_weights, WEIGHTS_BYTE_SIZE);
  if (bytesRead != WEIGHTS_BYTE_SIZE) {
    Serial.println("[MQTT Error] Lettura incompleta dello stream dei pesi globali!");
    free(flat_weights);
    return;
  }

  size_t offset = 0;
  memcpy(W1, flat_weights + offset, sizeof(W1));
  offset += (INPUT_SIZE * HIDDEN_SIZE);
  
  memcpy(b1, flat_weights + offset, sizeof(b1));
  offset += HIDDEN_SIZE;
  
  memcpy(W2, flat_weights + offset, sizeof(W2));
  offset += (HIDDEN_SIZE * OUTPUT_SIZE);
  
  memcpy(b2, flat_weights + offset, sizeof(b2));

  free(flat_weights);
  Serial.println("[MQTT] Modello globale ricevuto e applicato con successo in RAM!");
}

// ============================================================================
// GESTIONE MESSAGGI RICEVUTI DA BROKER MQTT
// ============================================================================
void onMqttMessage(int messageSize) {
  String topic = mqttClient.messageTopic();

  // Controllo se il messaggio in arrivo è il nuovo modello globale in broadcast
  if (topic == String(TOPIC_GLOBAL_WEIGHTS)) {
    Serial.printf("\n[MQTT] Ricevuto modello globale dal server sul topic broadcast: %s (%d byte)\n", topic.c_str(), messageSize);
    receive_global_weights(messageSize);
    return;
  }

  // Altrimenti gestiamo i messaggi in formato JSON (es. comandi di training)
  Serial.printf("\n[MQTT] Messaggio ricevuto sul topic: %s (%d byte)\n", topic.c_str(), messageSize);

  StaticJsonDocument<128> doc;
  DeserializationError error = deserializeJson(doc, mqttClient);

  if (error) {
    Serial.print("[MQTT Error] Parsing JSON fallito: ");
    Serial.println(error.f_str());
    return;
  }

  const char* action = doc["action"];

  if (action && strcmp(action, "train") == 0) {
    Serial.println("\n=== RICEVUTO COMANDO FL: Avvio Round Locale ===");
    
    run_local_training(LOCAL_EPOCHS, LEARNING_RATE);

    Serial.println("=== Addestramento completato. Avvio invio pesi... ===");
    send_weights_mqtt();
  }
}

void setup_wifi() {
  delay(10);
  Serial.printf("[WiFi] Connessione a %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connesso! IP: %s\n", WiFi.localIP().toString().c_str());
}

void reconnect_mqtt() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] Connessione al broker MQTT...");
    
    char mqtt_device_id[40];
    snprintf(mqtt_device_id, sizeof(mqtt_device_id), "ESP32S3_FL_%s", CLIENT_ID);
    mqttClient.setId(mqtt_device_id);
    
    if (mqttClient.connect(MQTT_SERVER, MQTT_PORT)) {
      Serial.println(" Connesso!");
      
      // Sottoscrizione al topic dei comandi globali
      mqttClient.subscribe(TOPIC_COMMAND);
      Serial.printf("[MQTT] Sottoscritto al topic: %s\n", TOPIC_COMMAND);
      
      // Sottoscrizione al topic unico di broadcast per il modello globale
      mqttClient.subscribe(TOPIC_GLOBAL_WEIGHTS);
      Serial.printf("[MQTT] Sottoscritto al topic broadcast: %s\n", TOPIC_GLOBAL_WEIGHTS);
    } else {
      Serial.printf(" Fallito (errore=%d). Riprovo tra 5 secondi...\n", mqttClient.connectError());
      delay(5000);
    }
  }
}

// ============================================================================
// 7. SETUP E LOOP PRINCIPALE
// ============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); } // Attesa della connessione della Seriale USB
  delay(1000);

  Serial.println("\n=================================================");
  Serial.println("=== FDK: ESP32-S3 FULL FLASH DATASET TRAINING ===");
  Serial.println("=================================================");

  // Configurazione dinamica del solo topic di invio pesi in base al CLIENT_ID
  snprintf(TOPIC_WEIGHTS, sizeof(TOPIC_WEIGHTS), "fl/%s/weights", CLIENT_ID);

  // Caricamento e perturbazione iniziale dei pesi
  init_weights_with_random_jitter();

  Serial.printf("[DATASET INFO] Training Samples: %d | Test Samples: %d\n", 
                NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES);

  float initial_acc = evaluate_test_set();
  Serial.printf("[PRE-TRAINING] Accuracy Test Set (%d campioni): %.2f%%\n", NUM_TEST_SAMPLES, initial_acc);
  Serial.println("-------------------------------------------------");

  setup_wifi();
  mqttClient.onMessage(onMqttMessage);
  reconnect_mqtt();
}

void loop() {
  if (!mqttClient.connected()) {
    reconnect_mqtt();
  }
  mqttClient.poll();
}