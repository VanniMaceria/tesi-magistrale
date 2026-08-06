#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <Arduino.h>

// Architettura MLP per MNIST / FEMNIST (784 -> 32 -> 10)
#define INPUT_SIZE  784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10

// Pesi e Bias iniziali inviati dal Server Federato (salvati in Flash con PROGMEM)
const float PROGMEM init_W1[INPUT_SIZE][HIDDEN_SIZE] = { {0.01f} }; // tutti i pesi di questo layer sono 0.01
const float PROGMEM init_b1[HIDDEN_SIZE] = {0.0f};
const float PROGMEM init_W2[HIDDEN_SIZE][OUTPUT_SIZE] = { {0.01f} };
const float PROGMEM init_b2[OUTPUT_SIZE] = {0.0f};

#endif