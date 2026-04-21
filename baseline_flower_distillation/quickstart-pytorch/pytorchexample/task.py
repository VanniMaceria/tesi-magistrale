"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class SmallNet(nn.Module):
    """Versione compressa del modello per i nodi IoT."""
    def __init__(self):
        super(SmallNet, self).__init__()
        # Convolutional layers
        # in_channels resta uguale (1), out_channels dimezzato (6 -> 3), kernel_size resta 5 
        self.conv1 = nn.Conv2d(1, 3, 5) 
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer resta uaguale (2X2)
        # in_channels = 3 -> Deve coincidere con gli out_channels di conv1.
        # out_channels = 8 -> Dimezzato (era 16). Rappresenta il numero di filtri complessi applicati (kernel).
        # kernel_size = 5 -> Dimensione del filtro (5x5). Resta uguale per mantenere il campo ricettivo.
        self.conv2 = nn.Conv2d(3, 8, 5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(8 * 4 * 4, 60) # (8 out_channels di conv2) * (4x4 dimensione spaziale dopo conv+pool) = 128
        self.fc2 = nn.Linear(60, 40)  # 60 neuroni in ingresso, 40 in uscita
        self.fc3 = nn.Linear(40, 10)  # 40 neuroni in ingresso, 10 classi in uscita (corrispondenti alle cifre 0-9)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

def set_all_seeds(seed):
    """Set all random seeds to make results reproducible."""
    import torch
    import numpy as np
    import random
    
    # Inizializza il generatore di numeri casuali standard di Python
    random.seed(seed)
    
    # Inizializza il generatore di numeri casuali di NumPy (usato per i dataset)
    np.random.seed(seed)
    
    # Imposta il seed per le operazioni PyTorch su CPU
    torch.manual_seed(seed)
    
    # Imposta il seed per le operazioni PyTorch su GPU (se disponibile)
    torch.cuda.manual_seed(seed)
    
    # Forza PyTorch a usare solo algoritmi deterministici (evita variazioni infinitesimali)
    torch.backends.cudnn.deterministic = True
    
    # Disabilita la ricerca automatica dell'algoritmo di convoluzione più veloce, 
    # che potrebbe introdurre variabilità tra un avvio e l'altro
    torch.backends.cudnn.benchmark = False

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int = 42):   # seed va a 42 se non lo specifico da shell
    """Load partition ylecun/mnist data for clients."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        #partitioner = IidPartitioner(num_partitions=num_partitions)
        dirichlet_partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.1, partition_by="label")
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": dirichlet_partitioner},
        )
    partition = fds.load_partition(partition_id)
    #MNIST dataset has "image" column, but our model expects "img" column, so we rename it here
    partition = partition.rename_column("image", "img")
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load the entire test set as a centralized dataset for evaluation on the server"""
    test_dataset = load_dataset("ylecun/mnist", split="test")
    test_dataset = test_dataset.rename_column("image", "img")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
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
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss, num_examples


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_model_iot_metrics():
    """Calcola i parametri totali e i FLOPs per una singola inferenza."""
    try:
        from thop import profile
        import torch
    except ImportError:
        print("\n[ERRORE] Libreria 'thop' mancante. Esegui: pip install thop\n")
        return 0, 0
    
    # 1. Inizializza il modello temporaneamente su CPU
    model = SmallNet()
    model_cpu = model.to('cpu')
    
    # 2. Crea un input finto della dimensione di un'immagine MNIST (1 canale, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # 3. Calcola MACs (Multiply-Accumulate) e Parametri (verbose=False nasconde i log di thop)
    macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False)
    
    # 4. Converti MACs in FLOPs (1 MAC = 1 Moltiplicazione + 1 Addizione = 2 FLOPs)
    flops = macs * 2 
    
    return int(params), int(flops)

def load_proxy_dataset(num_samples=500, seed=42):
    """
    Carica un piccolo set di dati (Proxy) dallo stesso dataset dei client
    per la distillazione lato server.
    """
    from datasets import load_dataset
    from torchvision.transforms import Compose, Normalize, ToTensor
    from torch.utils.data import DataLoader

    # Carichiamo un sottoinsieme casuale del test set di MNIST come proxy dataset (500 campioni di default)
    ds = load_dataset("ylecun/mnist", split="test") 
    ds = ds.rename_column("image", "img")
    proxy_ds = ds.shuffle(seed=seed).select(range(num_samples))
    
    # Definiamo le trasformazioni (identiche a quelle globali nel file)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    def apply_proxy_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    
    # Applichiamo le trasformazioni e creiamo il DataLoader
    proxy_ds = proxy_ds.with_transform(apply_proxy_transforms)
    return DataLoader(proxy_ds, batch_size=32)