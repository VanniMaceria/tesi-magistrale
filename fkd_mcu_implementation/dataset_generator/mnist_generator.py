import torchvision
import torchvision.transforms as transforms
import numpy as np

# ==========================================
# CONFIGURAZIONE CLIENT & DATASET
# ==========================================
TOTAL_SAMPLES = 1200   # Numero totale di campioni da assegnare a questo client
TRAIN_RATIO = 0.8     # 80% Training, 20% Testing
CLIENT_ID = 1         # ID del client (cambialo per estrarre porzioni diverse per client diversi)

NUM_TRAIN = int(TOTAL_SAMPLES * TRAIN_RATIO)  
NUM_TEST = TOTAL_SAMPLES - NUM_TRAIN         

print(f"Scaricamento MNIST in corso...")
print(f"Assegnazione a Client {CLIENT_ID}: {NUM_TRAIN} campioni per Training (80%) e {NUM_TEST} per Test (20%)...")

# 1. Download del Dataset
transform = transforms.ToTensor()
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. Estrazione dello slice unico per questo client
start_offset = CLIENT_ID * TOTAL_SAMPLES
client_data = [mnist_dataset[i] for i in range(start_offset, start_offset + TOTAL_SAMPLES)]

train_data = client_data[:NUM_TRAIN]
test_data = client_data[NUM_TRAIN:]

# 3. Generazione del file C Header (.h) per l'MCU
def export_to_flash_header(train_set, test_set, filename="mnist_flash_dataset.h"):
    with open(filename, "w") as f:
        f.write("/* \n")
        f.write(f" * Dataset MNIST formattato per Flash dell'MCU (Client ID: {CLIENT_ID})\n")
        f.write(f" * Train Samples: {len(train_set)} (80%) | Test Samples: {len(test_set)} (20%)\n")
        f.write(" */\n\n")
        f.write("#ifndef MNIST_FLASH_DATASET_H\n")
        f.write("#define MNIST_FLASH_DATASET_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"#define NUM_TRAIN_SAMPLES {len(train_set)}\n")
        f.write(f"#define NUM_TEST_SAMPLES {len(test_set)}\n")
        f.write("#define IMAGE_SIZE 784 // 28x28 pixel\n\n")
        
        # --- TRAIN IMAGES ---
        # L'uso di 'const' garantisce l'allocazione diretta in FLASH!
        f.write("// Immagini di Training salvate in memoria FLASH\n")
        f.write("const uint8_t train_images[NUM_TRAIN_SAMPLES][IMAGE_SIZE] = {\n")
        for img, _ in train_set:
            img_flat = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
            f.write("  {" + ", ".join(map(str, img_flat)) + "},\n")
        f.write("};\n\n")
        
        # --- TRAIN LABELS ---
        f.write("const uint8_t train_labels[NUM_TRAIN_SAMPLES] = {\n  ")
        f.write(", ".join(str(label) for _, label in train_set))
        f.write("\n};\n\n")
        
        # --- TEST IMAGES ---
        f.write("// Immagini di Test salvate in memoria FLASH\n")
        f.write("const uint8_t test_images[NUM_TEST_SAMPLES][IMAGE_SIZE] = {\n")
        for img, _ in test_set:
            img_flat = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
            f.write("  {" + ", ".join(map(str, img_flat)) + "},\n")
        f.write("};\n\n")
        
        # --- TEST LABELS ---
        f.write("const uint8_t test_labels[NUM_TEST_SAMPLES] = {\n  ")
        f.write(", ".join(str(label) for _, label in test_set))
        f.write("\n};\n\n")
        
        f.write("#endif // MNIST_FLASH_DATASET_H\n")

export_to_flash_header(train_data, test_data)
print("-> Generato con successo il file 'mnist_flash_dataset.h'!")