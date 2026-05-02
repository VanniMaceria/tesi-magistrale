"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

class Net(nn.Module):
    """Rete baseline originale resa elastica tramite Ordered Dropout. Questa è la rete con p=1"""
    def __init__(self):
        super(Net, self).__init__()
        # Conv1: 1 canale input (MNIST è B/W), 6 filtri in uscita (feature map), kernel 5x5
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        # Conv2: 6 canali input, 16 filtri in uscita, kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Strati Fully Connected (Dense)
        # 16*4*4 è la dimensione piatta post-convoluzioni per MNIST
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, p=1.0):
        """
        Il cuore dell'Ordered Dropout. 
        Invece di usare moduli nn.Module fissi, usiamo le funzioni F (functional)
        per ritagliare (slice) le matrici dei pesi in tempo reale. I pesi più importanti
        si trovano nell'angolo in alto a sinistra poichè tutti i client in base al loro 'p'
        aggiornano quella zona della matrice, mentre le zone più a destra e in basso vengono
        aggiornate solo dai client con p più alto.
        """

        # 1. CALCOLO DINAMICO DELLE AMPIEZZE (Width Slicing)
        # max(1, ...) evita che il layer si svuoti completamente se p è molto piccolo.
        c1_out = max(1, int(6 * p)) # Numero filtri attivi in Conv1
        c2_out = max(1, int(16 * p))  # Numero filtri attivi in Conv2
        f1_out = max(1, int(120 * p))  # Numero neuroni attivi in FC1
        f2_out = max(1, int(84 * p))  # Numero neuroni attivi in FC2

        # 2. CONVOLUZIONE 1
        # Prendiamo solo i primi 'c1_out' filtri dai 6 disponibili.
        # Questo forza il modello a mettere le info più importanti nei primi indici.
        x = F.relu(F.conv2d(x, self.conv1.weight[:c1_out], self.conv1.bias[:c1_out], stride=1, padding=0))
        x = self.pool(x)
        
        # 3. CONVOLUZIONE 2
        # Usiamo [:c2_out, :c1_out] per selezionare una sottomatrice
        # dall'angolo in alto a sinistra della matrice originale.
        # - c2_out: riduce il numero di filtri in uscita (righe della matrice).
        # - c1_out: riduce il numero di canali in ingresso (colonne della matrice).
        x = F.relu(F.conv2d(x, self.conv2.weight[:c2_out, :c1_out], self.conv2.bias[:c2_out], stride=1, padding=0))
        x = self.pool(x)
        
        # 4. FLATTEN (Appiattimento)
        # Trasformiamo la mappa di feature 3D in un vettore 1D per i livelli densi (FC).
        # - c2_out: è il numero di filtri attivi (profondità).
        # - 4 * 4: è l'area residua dell'immagine dopo convoluzioni e pooling.
        # - 4 * 4 vale per MNIST (28 * 28), da modificare in base al dataset
        # Esempio: se c2_out=8, avremo un vettore di 8 * 16 = 128 neuroni in ingresso a FC1.
        x = x.view(-1, c2_out * 4 * 4) 

        # 5. LIVELLI FULLY CONNECTED
        # Applichiamo lo slicing 'Matrioska' anche qui: [uscita, ingresso]
        x = F.relu(F.linear(x, self.fc1.weight[:f1_out, :c2_out * 4 * 4], self.fc1.bias[:f1_out]))
        x = F.relu(F.linear(x, self.fc2.weight[:f2_out, :f1_out], self.fc2.bias[:f2_out]))

        # L'uscita finale deve essere sempre 10, quindi tagliamo solo l'ingresso
        x = F.linear(x, self.fc3.weight[:, :f2_out], self.fc3.bias)
        
        return x

def get_p_from_id(partition_id):
    """Associa un valore p fisso basato sull'ID del client (0.25, 0.5, 0.75, 1.0)."""
    p_values = [0.25, 0.5, 0.75, 1.0]
    return p_values[partition_id % len(p_values)]

fds = None  # Cache FederatedDataset
pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

def set_all_seeds(seed):
    """Set all random seeds to make results reproducible."""
    import torch
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int = 42):
    """Load partition ylecun/mnist data for clients."""
    global fds
    if fds is None:
        #partitioner = IidPartitioner(num_partitions=num_partitions)
        dirichlet_partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.1, partition_by="label")
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": dirichlet_partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition = partition.rename_column("image", "img")
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset():
    """Load the entire test set as a centralized dataset for evaluation on the server"""
    test_dataset = load_dataset("ylecun/mnist", split="test")
    test_dataset = test_dataset.rename_column("image", "img")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(net, trainloader, epochs, lr, device, p_fixed=1.0):
    """Train the model on the training set with Ordered Dropout profile."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    num_examples = len(trainloader.dataset)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images, p=p_fixed)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss, num_examples

def test(net, testloader, device, p=1.0):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images, p=p)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_model_iot_metrics(p=1.0):
    """
    Calcola analiticamente parametri totali e i FLOPs per l'Ordered Dropout 
    dato che Thop non in grado di farlo con i layer dinamici della rete.
    """
    # Ripetiamo il calcolo delle fette per coerenza matematica.
    c1_out = max(1, int(6 * p))
    c2_out = max(1, int(16 * p))
    f1_out = max(1, int(120 * p))
    f2_out = max(1, int(84 * p))

   # --- PARAMETRI (Memoria) ---
    # Formula: (Kernel_H * Kernel_W * In_Channels + Bias) * Out_Channels
    p_conv1 = (1 * 5 * 5 + 1) * c1_out
    p_conv2 = (c1_out * 5 * 5 + 1) * c2_out
    p_fc1 = (c2_out * 16 + 1) * f1_out
    p_fc2 = (f1_out + 1) * f2_out
    p_fc3 = (f2_out + 1) * 10
    total_params = p_conv1 + p_conv2 + p_fc1 + p_fc2 + p_fc3

    # --- FLOPs ---
    # Usiamo i MACs (Multiply-Accumulate) e li moltiplichiamo per 2.
    # Formula generale: MACs = (Volume_Filtro) * (Numero_Filtri) * (Area_Mappa_Output)

    # 1. CONV1: (5x5 kernel * 1 canale) * c1_out filtri * mappa 24x24
    macs_conv1 = (5 * 5 * 1) * c1_out * 24 * 24

    # 2. CONV2: (5x5 kernel * c1_out canali) * c2_out filtri * mappa 8x8
    macs_conv2 = (5 * 5 * c1_out) * c2_out * 8 * 8

    # 3. LIVELLI FC: (Neuroni_Input * Neuroni_Output)
    # FC1: Input 16 (mappa 4x4) * c2_out canali
    macs_fc1 = (c2_out * 16) * f1_out
    macs_fc2 = f1_out * f2_out
    macs_fc3 = f2_out * 10

    total_macs = macs_conv1 + macs_conv2 + macs_fc1 + macs_fc2 + macs_fc3
    total_flops = total_macs * 2

    return int(total_params), int(total_flops)