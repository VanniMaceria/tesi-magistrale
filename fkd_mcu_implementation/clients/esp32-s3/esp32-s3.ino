#include <Arduino.h>
#include "mnist_flash_dataset.h" // Dataset MNIST allocato in Flash (PROGMEM)
#include "model_weights.h"       // Definizione dell'architettura e pesi iniziali

// ============================================================================
// 1. IPERPARAMETRI DI ADDESTRAMENTO E ARCHITETTURA DELLA RETE
// ============================================================================
#define LEARNING_RATE 0.03f  // Tasso di apprendimento per l'aggiornamento SGD
#define LOCAL_EPOCHS  10     // Epoche locali
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
      W2[j][k] -= LEARNING_RATE * d_output[k] * hidden_activations[j];
    }
    b2[k] -= LEARNING_RATE * d_output[k];
  }

  // 6. Aggiornamento Pesi W1 e Bias b1 tramite SGD
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    for (int i = 0; i < INPUT_SIZE; i++) {
      float input_val = (float)image_bytes[i] / 255.0f;
      W1[i][j] -= LEARNING_RATE * d_hidden[j] * input_val;
    }
    b1[j] -= LEARNING_RATE * d_hidden[j];
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
// 7. SETUP ED ESECUZIONE DEL CICLO DI ADDESTRAMENTO
// ============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); } // Attesa della connessione della Seriale USB
  delay(1000);

  Serial.println("\n=================================================");
  Serial.println("=== FDK: ESP32-S3 FULL FLASH DATASET TRAINING ===");
  Serial.println("=================================================");

  // Caricamento e perturbazione iniziale dei pesi
  init_weights_with_random_jitter();

  Serial.printf("[DATASET INFO] Training Samples: %d | Test Samples: %d\n", 
                NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES);

  // Valutazione iniziale del modello NON addestrato sul Test Set
  float initial_acc = evaluate_test_set();
  Serial.printf("[PRE-TRAINING] Accuracy Test Set (%d campioni): %.2f%%\n", NUM_TEST_SAMPLES, initial_acc);
  Serial.println("-------------------------------------------------");

  unsigned long start_time_ms = millis(); // Cronometro per la profilazione dell'addestramento

  // Ciclo principale di addestramento su più epoche locali
  for (int epoch = 1; epoch <= LOCAL_EPOCHS; epoch++) {
    float epoch_loss = 0.0f;
    
    // Ciclo sui 200 campioni di training disponibili in Flash
    for (int n = 0; n < NUM_TRAIN_SAMPLES; n++) {
      float l = train_single_sample(train_images[n], train_labels[n]);
      epoch_loss += l;
    }
    
    // Log di progresso dell'epoca
    Serial.printf("[EPOCH %2d/%d] Loss Media Training: %.4f | Current Test Acc: %.2f%%\n", 
                  epoch, LOCAL_EPOCHS, epoch_loss / NUM_TRAIN_SAMPLES, evaluate_test_set());
  }

  unsigned long elapsed_time_ms = millis() - start_time_ms;

  // Valutazione finale post-training sul Test Set
  float final_acc = evaluate_test_set();

  Serial.println("-------------------------------------------------");
  Serial.printf("[POST-TRAINING STATS]\n");
  Serial.printf("Tempo Totale Training (%d epoche x %d campioni): %lu ms (%.2f s)\n", 
                LOCAL_EPOCHS, NUM_TRAIN_SAMPLES, elapsed_time_ms, elapsed_time_ms / 1000.0f);
  Serial.printf("Tempo Medio per Epoca:      %.2f ms\n", (float)elapsed_time_ms / LOCAL_EPOCHS);
  Serial.printf("Accuracy INIZIALE Test Set: %.2f%%\n", initial_acc);
  Serial.printf("Accuracy FINALE Test Set:   %.2f%%\n", final_acc);
  Serial.println("=================================================");
}

void loop() {
  delay(5000); // Mantiene in esecuzione lo sketch senza sovraccaricare la CPU
}