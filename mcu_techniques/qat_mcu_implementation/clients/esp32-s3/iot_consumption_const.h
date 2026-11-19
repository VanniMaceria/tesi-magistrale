// ============================================================================
// GIUSTIFICAZIONE TEORICA DEI FLOPs E MODELLO ENERGETICO (Horowitz 2014)
// ============================================================================
// Architettura MLP: Input (784) -> Hidden (32) + ReLU -> Output (10)
// Convenzione standard: 1 operazione MAC (Multiply-Accumulate) = 2 FLOPs.
//
// 1. FORWARD PASS (~50.920 FLOPs):
//    - Layer 1 (784x32):  784 * 32 * 2 (MAC) + 32 (bias) = 50.208 FLOPs
//    - ReLU (32 neuroni): 32 FLOPs (operazioni di thresholding)
//    - Layer 2 (32x10):   32 * 10 * 2 (MAC) + 10 (bias) = 650 FLOPs
//    - Loss (Softmax+CE): ~30 FLOPs (exp, sum, log su 10 classi)
//    - Totale Forward:    ~50.920 FLOPs
//
// 2. BACKWARD PASS (~51.530 FLOPs):
//    - Gradienti Layer 2 (d_out * a_hid^T) & delta_hid: (32*10*2) + 10 + (32*10*2) = 1.290 FLOPs
//    - Maschera gradiente ReLU (32 neuroni): 32 FLOPs
//    - Gradienti Layer 1 (delta_hid * x_in^T): (784*32*2) + 32 = 50.208 FLOPs
//    - Totale Backward:   ~51.530 FLOPs
//
// 3. WEIGHT UPDATE (~50.964 FLOPs):
//    - Parametri totali: (784*32 + 32) + (32*10 + 10) = 25.482 pesi/bias
//    - Aggiornamento (W = W - lr * grad): 2 FLOPs/parametro -> 25.482 * 2 = 50.964 FLOPs
//
// TOTALE PER CAMPIONE: Forward + Backward + Update = 153.414 FLOPs (~3x Forward Pass)
// ============================================================================

const unsigned long long FLOPS_PER_SAMPLE = 153414ULL; // FLOPs esatti per ciclo di addestramento su singolo campione
const double JOULE_PER_FLOP = 1e-11;                   // 10 pJ per operazione matematica FP32 + SRAM (Horowitz 2014)
const double JOULE_PER_MB   = 0.05;                    // 50 mJ per Megabyte trasmesso/ricevuto via Wi-Fi