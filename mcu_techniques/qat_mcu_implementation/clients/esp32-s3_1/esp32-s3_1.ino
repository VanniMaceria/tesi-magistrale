#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoMqttClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "mnist_flash_dataset.h" // Dataset MNIST allocato in Flash
#include "model_weights.h"       // Architettura (784 -> 32 -> 10) e pesi iniziali
#include "wifi_credentials.h"
#include "iot_consumption_const.h"

// ============================================================================
// CONFIGURAZIONE RETE E BROKER MQTT
// ============================================================================
const char* MQTT_SERVER   = "10.59.21.81";
const int   MQTT_PORT     = 1883;

const char* CLIENT_ID     = "client_1";

const char* TOPIC_COMMAND        = "fl_qat/global/command";
const char* TOPIC_GLOBAL_WEIGHTS = "fl_qat/global/global_weights";
char TOPIC_WEIGHTS[64];

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

// ============================================================================
// FORMATO DEL PAYLOAD BINARIO INT8 (little-endian, identico lato server)
//   [ W1 int8 (784x32, riga-maggiore [i][j]) | W2 int8 (32x10, [j][k]) |
//     b1 float32 (32) | b2 float32 (10) |
//     scale float32 x4: s_w1, s_w2, s_a1 (attivazioni hidden), s_a2 (logits) ]
// I bias restano float32 (come l'accumulatore int32 dei kernel int8 reali).
// ============================================================================
#define N_W1 (INPUT_SIZE * HIDDEN_SIZE)
#define N_W2 (HIDDEN_SIZE * OUTPUT_SIZE)
#define N_BIAS (HIDDEN_SIZE + OUTPUT_SIZE)
#define N_SCALES 4
#define QAT_BYTE_SIZE (N_W1 + N_W2 + (N_BIAS + N_SCALES) * sizeof(float))   // 25.592 byte
#define FP32_BYTE_SIZE ((N_W1 + N_W2 + N_BIAS) * 4)                         // 101.800 byte (baseline)

// ============================================================================
// 1. IPERPARAMETRI
// ============================================================================
#define LEARNING_RATE 0.03f
#define LOCAL_EPOCHS  10
float current_lr = LEARNING_RATE;

#define QMIN -128
#define QMAX  127
#define INPUT_SCALE   (1.0f / 127.0f) // QuantStub: pixel [0,1] -> int8 [0,127]
#define OBS_MOMENTUM  0.1f            // EMA dell'observer di max per le attivazioni

// ==== ML CORE BEGIN =========================================================

// ----------------------------------------------------------------------------
// STATO DEL MODELLO
// ----------------------------------------------------------------------------
// Pesi LATENTI float32: su questi si applica lo SGD (STE) e si fa FedAvg
float W1[INPUT_SIZE][HIDDEN_SIZE];
float b1[HIDDEN_SIZE];
float W2[HIDDEN_SIZE][OUTPUT_SIZE];
float b2[OUTPUT_SIZE];

// Copia int8 dei pesi usata dal forward (fake-quantization) e per l'uplink
int8_t Wq1[INPUT_SIZE][HIDDEN_SIZE];
int8_t Wq2[HIDDEN_SIZE][OUTPUT_SIZE];
float s_w1 = 1.0f, s_w2 = 1.0f;     // scale pesi (simmetrica per-tensor: max|W|/127)

// Observer per le attivazioni (EMA del massimo)
float ema_h_max = 1.0f, ema_l_max = 1.0f;
bool  obs_init  = false;
float s_a1 = 1.0f / 127.0f;         // scale attivazioni hidden (post-ReLU)
float s_a2 = 1.0f / 127.0f;         // scale logits

// Buffer del forward pass
int8_t x_q[INPUT_SIZE];             // input quantizzato
float  hidden_pre[HIDDEN_SIZE];     // pre-attivazione (per derivata ReLU)
float  hidden_f[HIDDEN_SIZE];       // ReLU float (prima del fake-quant)
float  a_dq[HIDDEN_SIZE];           // attivazione hidden fake-quantizzata
float  logits_f[OUTPUT_SIZE];       // logits float (prima del fake-quant)
float  output_probs[OUTPUT_SIZE];

// Buffer gradienti
float d_output[OUTPUT_SIZE];
float d_hidden[HIDDEN_SIZE];

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

void softmax(float* input, float* output, int size) {
  float max_val = input[0];
  for (int i = 1; i < size; i++) if (input[i] > max_val) max_val = input[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) { output[i] = expf(input[i] - max_val); sum += output[i]; }
  for (int i = 0; i < size; i++) output[i] /= sum;
}

// Fake-quant: round-to-nearest-even + clipping in [-128, 127]
inline int8_t quant_i8(float v, float inv_scale) {
  float r = rintf(v * inv_scale);
  if (r > QMAX) r = QMAX; else if (r < QMIN) r = QMIN;
  return (int8_t)r;
}

// Ricalcola scale e copia int8 dei pesi a partire dai pesi latenti.
// Va chiamata dopo ogni aggiornamento SGD e dopo ogni caricamento di pesi.
void requantize_weights() {
  float m1 = 0.0f, m2 = 0.0f;
  for (int i = 0; i < INPUT_SIZE; i++)
    for (int j = 0; j < HIDDEN_SIZE; j++) { float a = fabsf(W1[i][j]); if (a > m1) m1 = a; }
  for (int j = 0; j < HIDDEN_SIZE; j++)
    for (int k = 0; k < OUTPUT_SIZE; k++) { float a = fabsf(W2[j][k]); if (a > m2) m2 = a; }
  if (m1 < 1e-8f) m1 = 1e-8f;
  if (m2 < 1e-8f) m2 = 1e-8f;
  s_w1 = m1 / 127.0f;
  s_w2 = m2 / 127.0f;
  float inv1 = 1.0f / s_w1, inv2 = 1.0f / s_w2;
  for (int i = 0; i < INPUT_SIZE; i++)
    for (int j = 0; j < HIDDEN_SIZE; j++) Wq1[i][j] = quant_i8(W1[i][j], inv1);
  for (int j = 0; j < HIDDEN_SIZE; j++)
    for (int k = 0; k < OUTPUT_SIZE; k++) Wq2[j][k] = quant_i8(W2[j][k], inv2);
}

// Observer: restituisce la scale (max/127). In training aggiorna l'EMA,
// in valutazione la congela (prima del primo training usa il massimo osservato).
float observe_scale(float observed_max, float &ema, bool training) {
  if (observed_max < 1e-6f) observed_max = 1e-6f;
  float m;
  if (training) {
    ema = obs_init ? (1.0f - OBS_MOMENTUM) * ema + OBS_MOMENTUM * observed_max : observed_max;
    m = ema;
  } else {
    m = obs_init ? ema : observed_max;
  }
  return m / 127.0f;
}

// ----------------------------------------------------------------------------
// FORWARD PASS con fake-quantization (aritmetica int8 x int8 -> int32)
// ----------------------------------------------------------------------------
void forward(const uint8_t* image_bytes, bool training) {
  // QuantStub: pixel uint8 -> int8 [0,127]
  for (int i = 0; i < INPUT_SIZE; i++)
    x_q[i] = (int8_t)rintf((float)image_bytes[i] * (127.0f / 255.0f));

  // Layer 1: MAC interi, poi riscalatura in float + bias + ReLU
  const float k1 = INPUT_SCALE * s_w1;
  float h_max = 0.0f;
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    int32_t acc = 0;
    for (int i = 0; i < INPUT_SIZE; i++) acc += (int32_t)x_q[i] * (int32_t)Wq1[i][j];
    float z = (float)acc * k1 + b1[j];
    hidden_pre[j] = z;
    float h = relu(z);
    hidden_f[j] = h;
    if (h > h_max) h_max = h;
  }
  s_a1 = observe_scale(h_max, ema_h_max, training);
  float inv_a1 = 1.0f / s_a1;
  int8_t a_q[HIDDEN_SIZE];
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    a_q[j] = quant_i8(hidden_f[j], inv_a1);
    a_dq[j] = (float)a_q[j] * s_a1;
  }

  // Layer 2
  const float k2 = s_a1 * s_w2;
  float l_max = 0.0f;
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    int32_t acc = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) acc += (int32_t)a_q[j] * (int32_t)Wq2[j][k];
    float l = (float)acc * k2 + b2[k];
    logits_f[k] = l;
    if (fabsf(l) > l_max) l_max = fabsf(l);
  }
  s_a2 = observe_scale(l_max, ema_l_max, training);
  float inv_a2 = 1.0f / s_a2;

  // Fake-quant dei logits + DeQuantStub -> Softmax in float
  float logits_dq[OUTPUT_SIZE];
  for (int k = 0; k < OUTPUT_SIZE; k++) logits_dq[k] = (float)quant_i8(logits_f[k], inv_a2) * s_a2;
  softmax(logits_dq, output_probs, OUTPUT_SIZE);

  if (training) obs_init = true;
}

// ----------------------------------------------------------------------------
// BACKWARD con Straight-Through Estimator + SGD sui pesi latenti float32
// ----------------------------------------------------------------------------
float train_single_sample(const uint8_t* image_bytes, uint8_t label) {
  forward(image_bytes, true);

  float loss = -logf(output_probs[label] + 1e-7f);

  // dL/dLogits = Probs - OneHot; STE: gradiente nullo dove il logit è stato saturato
  const float l_lim = 127.0f * s_a2;
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    float target = (k == label) ? 1.0f : 0.0f;
    float ste = (fabsf(logits_f[k]) <= l_lim) ? 1.0f : 0.0f;
    d_output[k] = (output_probs[k] - target) * ste;
  }

  // Retropropagazione allo strato nascosto con i pesi W2 usati nel forward (quantizzati)
  const float h_lim = 127.0f * s_a1;
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    float grad_sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; k++) grad_sum += d_output[k] * ((float)Wq2[j][k] * s_w2);
    float ste = (hidden_pre[j] > 0.0f && hidden_f[j] <= h_lim) ? 1.0f : 0.0f; // ReLU' * clip-mask
    d_hidden[j] = grad_sum * ste;
  }

  // Update dei latenti float32 (STE: il gradiente sui pesi passa identico al quantizzatore)
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) W2[j][k] -= current_lr * d_output[k] * a_dq[j];
    b2[k] -= current_lr * d_output[k];
  }
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    for (int i = 0; i < INPUT_SIZE; i++) W1[i][j] -= current_lr * d_hidden[j] * ((float)x_q[i] * INPUT_SCALE);
    b1[j] -= current_lr * d_hidden[j];
  }

  requantize_weights(); // il forward successivo vede i pesi aggiornati e ri-quantizzati
  return loss;
}

float evaluate_test_set() {
  int correct = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    forward(test_images[i], false);
    int pred = 0;
    float max_p = output_probs[0];
    for (int k = 1; k < OUTPUT_SIZE; k++) if (output_probs[k] > max_p) { max_p = output_probs[k]; pred = k; }
    if (pred == test_labels[i]) correct++;
  }
  return ((float)correct / NUM_TEST_SAMPLES) * 100.0f;
}

// Serializzazione payload int8 in un buffer di QAT_BYTE_SIZE byte
void pack_model_int8(uint8_t* out) {
  size_t o = 0;
  memcpy(out + o, Wq1, N_W1); o += N_W1;
  memcpy(out + o, Wq2, N_W2); o += N_W2;
  memcpy(out + o, b1, sizeof(b1)); o += sizeof(b1);
  memcpy(out + o, b2, sizeof(b2)); o += sizeof(b2);
  float sc[4] = { s_w1, s_w2, s_a1, s_a2 };
  memcpy(out + o, sc, sizeof(sc));
}

// Deserializzazione: ricostruisce i latenti dequantizzando, riallinea observer e copie int8
void unpack_model_int8(const uint8_t* in) {
  size_t o = 0;
  const int8_t* q1 = (const int8_t*)(in + o); o += N_W1;
  const int8_t* q2 = (const int8_t*)(in + o); o += N_W2;
  memcpy(b1, in + o, sizeof(b1)); o += sizeof(b1);
  memcpy(b2, in + o, sizeof(b2)); o += sizeof(b2);
  float sc[4];
  memcpy(sc, in + o, sizeof(sc));

  for (int i = 0; i < INPUT_SIZE; i++)
    for (int j = 0; j < HIDDEN_SIZE; j++) W1[i][j] = (float)q1[i * HIDDEN_SIZE + j] * sc[0];
  for (int j = 0; j < HIDDEN_SIZE; j++)
    for (int k = 0; k < OUTPUT_SIZE; k++) W2[j][k] = (float)q2[j * OUTPUT_SIZE + k] * sc[1];

  s_a1 = sc[2]; s_a2 = sc[3];
  ema_h_max = sc[2] * 127.0f;
  ema_l_max = sc[3] * 127.0f;
  obs_init = true;
  requantize_weights();
}
// ==== ML CORE END ===========================================================

// ============================================================================
// 2. INIZIALIZZAZIONE PESI (Symmetry Breaking, come nel client di distillazione)
// ============================================================================
void init_weights_with_random_jitter() {
  srand(42);
  for (int i = 0; i < INPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      float jitter = ((float)rand() / RAND_MAX - 0.5f) * 0.10f;
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
  for (int k = 0; k < OUTPUT_SIZE; k++) b2[k] = init_b2[k];
  requantize_weights();
}

// ============================================================================
// ESECUZIONE DEL CICLO DI ADDESTRAMENTO LOCALE
// ============================================================================
void run_local_training(int epochs, float lr) {
  current_lr = lr;
  unsigned long start_time_ms = millis();

  for (int epoch = 1; epoch <= epochs; epoch++) {
    float epoch_loss = 0.0f;
    for (int n = 0; n < NUM_TRAIN_SAMPLES; n++) {
      epoch_loss += train_single_sample(train_images[n], train_labels[n]);
    }
    mqttClient.poll();
    Serial.printf("[EPOCH %2d/%d] Loss Media Training: %.4f | Test Acc (int8): %.2f%% | s_a1=%.4f s_a2=%.4f\n",
                  epoch, epochs, epoch_loss / NUM_TRAIN_SAMPLES, evaluate_test_set(), s_a1, s_a2);
  }

  unsigned long elapsed_time_ms = millis() - start_time_ms;
  float final_acc = evaluate_test_set();

  Serial.println("-------------------------------------------------");
  Serial.printf("Tempo Totale Training QAT (%d epoche x %d campioni): %lu ms (%.2f s)\n",
                epochs, NUM_TRAIN_SAMPLES, elapsed_time_ms, elapsed_time_ms / 1000.0f);
  Serial.printf("Tempo Medio per Epoca:      %.2f ms\n", (float)elapsed_time_ms / epochs);
  Serial.printf("Accuracy FINALE Test Set (int8): %.2f%%\n", final_acc);
  Serial.println("-------------------------------------------------");
}

// ============================================================================
// TRASMISSIONE E RICEZIONE BINARIA INT8 VIA MQTT
// ============================================================================
void send_weights_mqtt() {
  uint8_t* payload = (uint8_t*) malloc(QAT_BYTE_SIZE);
  if (payload == NULL) {
    Serial.println("[MQTT Error] Allocazione RAM fallita per l'invio pesi!");
    return;
  }
  pack_model_int8(payload);

  Serial.printf("[MQTT] Invio pesi INT8 (%d byte) sul topic %s...\n", (int)QAT_BYTE_SIZE, TOPIC_WEIGHTS);
  mqttClient.beginMessage(TOPIC_WEIGHTS, QAT_BYTE_SIZE, false, 0, false);
  mqttClient.write(payload, QAT_BYTE_SIZE);
  bool success = mqttClient.endMessage();

  Serial.println(success ? "[MQTT] Pesi inviati correttamente al server!"
                         : "[MQTT Error] Errore nell'invio del messaggio MQTT.");
  free(payload);
}

// Lettura robusta di n byte dallo stream del messaggio corrente
bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms = 5000) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int r = mqttClient.read(dst + got, n - got);
    if (r > 0) { got += r; t0 = millis(); }
    else {
      if (millis() - t0 > timeout_ms) return false;
      delay(1);
    }
  }
  return true;
}

void receive_global_weights(int messageSize) {
  if (messageSize == 0) return; // retained cancellato dal server a inizio esperimento

  if (messageSize != QAT_BYTE_SIZE) {
    Serial.printf("[MQTT Error] Dimensione pesi globali errata! Attesi %d byte, ricevuti %d\n", (int)QAT_BYTE_SIZE, messageSize);
    while (mqttClient.available()) { mqttClient.read(); }
    return;
  }

  uint8_t* payload = (uint8_t*) malloc(QAT_BYTE_SIZE);
  if (payload == NULL) {
    Serial.println("[MQTT Error] Allocazione RAM fallita per la ricezione pesi globali!");
    return;
  }
  if (!read_exact(payload, QAT_BYTE_SIZE)) {
    Serial.println("[MQTT Error] Lettura incompleta dello stream dei pesi globali!");
    free(payload);
    return;
  }
  unpack_model_int8(payload);
  free(payload);
  Serial.println("[MQTT] Modello globale INT8 ricevuto e applicato (latenti float32 ricostruiti)!");
}

// ============================================================================
// GESTIONE MESSAGGI RICEVUTI DAL BROKER MQTT
// ============================================================================
void onMqttMessage(int messageSize) {
  String topic = mqttClient.messageTopic();

  if (topic == String(TOPIC_GLOBAL_WEIGHTS)) {
    Serial.printf("\n[MQTT] Ricevuto modello globale: %s (%d byte)\n", topic.c_str(), messageSize);
    receive_global_weights(messageSize);
    return;
  }

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
    Serial.println("\n=== RICEVUTO COMANDO FL: Avvio Round Locale QAT ===");
    run_local_training(LOCAL_EPOCHS, LEARNING_RATE);
    Serial.println("=== Addestramento completato. Avvio invio pesi... ===");
    send_weights_mqtt();
  }
}

void setup_wifi() {
  delay(10);
  Serial.printf("[WiFi] Connessione a %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\n[WiFi] Connesso! IP: %s\n", WiFi.localIP().toString().c_str());
}

void reconnect_mqtt() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] Connessione al broker MQTT...");
    char mqtt_device_id[48];
    snprintf(mqtt_device_id, sizeof(mqtt_device_id), "ESP32S3_QAT_%s", CLIENT_ID);
    mqttClient.setId(mqtt_device_id);

    if (mqttClient.connect(MQTT_SERVER, MQTT_PORT)) {
      Serial.println(" Connesso!");
      mqttClient.subscribe(TOPIC_COMMAND);
      Serial.printf("[MQTT] Sottoscritto al topic: %s\n", TOPIC_COMMAND);
      mqttClient.subscribe(TOPIC_GLOBAL_WEIGHTS);
      Serial.printf("[MQTT] Sottoscritto al topic broadcast: %s\n", TOPIC_GLOBAL_WEIGHTS);
    } else {
      Serial.printf(" Fallito (errore=%d). Riprovo tra 5 secondi...\n", mqttClient.connectError());
      delay(5000);
    }
  }
}

// ============================================================================
// METRICHE IoT DEL ROUND
// ============================================================================
// FLOPs per campione identici al baseline FP32 (stesso numero di operazioni logiche, per
// confronto diretto con la distillazione); a cambiare è la banda: 1 byte/peso anziché 4.
void report_round_metrics(int train_samples, int epochs) {
  unsigned long long total_flops = FLOPS_PER_SAMPLE * (unsigned long long)train_samples * (unsigned long long)epochs;
  double mflops = (double)total_flops / 1e6;

  size_t rx_bytes = QAT_BYTE_SIZE;
  size_t tx_bytes = QAT_BYTE_SIZE;
  size_t total_bytes = rx_bytes + tx_bytes;
  double total_mb = (double)total_bytes / (1024.0 * 1024.0);
  double fp32_mb  = (double)(2 * FP32_BYTE_SIZE) / (1024.0 * 1024.0);

  double energy_comp_j = (double)total_flops * JOULE_PER_FLOP;
  double energy_comm_j = total_mb * JOULE_PER_MB;
  double total_energy_j = energy_comp_j + energy_comm_j;

  Serial.println("\n========== METRICHE IOT ROUND (QAT int8) ==========");
  Serial.printf("Campioni Addestrati:     %d | Epoche: %d\n", train_samples, epochs);
  Serial.printf("Computazione Totale:      %.2f MFLOPs (%.2f GFLOPs)\n", mflops, mflops / 1000.0);
  Serial.printf("Banda Totale Scambiata:   %.4f MB (%zu byte) | baseline FP32: %.4f MB (-%.1f%%)\n",
                total_mb, total_bytes, fp32_mb, 100.0 * (1.0 - total_mb / fp32_mb));
  Serial.printf("Energia Totale Consumata: %.6f J (%.3f mJ)\n", total_energy_j, total_energy_j * 1e3);
  Serial.println("===========================================================\n");
}

// ============================================================================
// SETUP E LOOP PRINCIPALE
// ============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(1000);

  Serial.println("\n=================================================");
  Serial.println("=== FL-QAT: ESP32-S3 INT8 FAKE-QUANT TRAINING ===");
  Serial.println("=================================================");

  snprintf(TOPIC_WEIGHTS, sizeof(TOPIC_WEIGHTS), "fl_qat/%s/weights", CLIENT_ID);

  init_weights_with_random_jitter();

  Serial.printf("[DATASET INFO] Training Samples: %d | Test Samples: %d\n", NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES);
  Serial.printf("[PRE-TRAINING] Accuracy Test Set (%d campioni): %.2f%%\n", NUM_TEST_SAMPLES, evaluate_test_set());
  Serial.println("-------------------------------------------------");

  report_round_metrics(NUM_TRAIN_SAMPLES, LOCAL_EPOCHS);

  setup_wifi();
  mqttClient.onMessage(onMqttMessage);
  reconnect_mqtt();
}

void loop() {
  if (!mqttClient.connected()) reconnect_mqtt();
  mqttClient.poll();
}
