"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.ao.quantization import QuantStub, DeQuantStub

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        
        # =====================================================================
        # FAKE QUANTIZATION: STUB DI INGRESSO E USCITA
        # =====================================================================
        # QuantStub simula la conversione del sensore (es. fotocamera di Arduino)
        # che acquisisce l'immagine in float e la converte in int8 prima di 
        # passarla alla rete neurale.
        self.quant = QuantStub() 
        
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # DeQuantStub simula il passaggio finale: l'output della rete a 8-bit
        # viene riportato in float32 per poter calcolare l'errore (Loss) 
        # durante l'addestramento.
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x) # INIZIO ZONA INT8 (Simulata)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = self.dequant(x) # FINE ZONA INT8 (Ritorno a float)
        return x

fds = None  
# Per MNIST
#pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
# Per Fashion-MNIST
pytorch_transforms = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])

def set_all_seeds(seed):
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
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int = 42): 
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        #dirichlet_partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.1, partition_by="label")
        fds = FederatedDataset(dataset="zalando-datasets/fashion_mnist", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    partition = partition.rename_column("image", "img")
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset():
    test_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
    test_dataset = test_dataset.rename_column("image", "img")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  #momentum=0.9 se batch-size è 32
    num_examples = len(trainloader.dataset)
    
    # La rete è già in modalità QAT, quindi i gradienti si calcoleranno sui "pesi latenti" (float32)
    # mentre la forward pass subirà il "rumore" e il clipping tipici dell'int8.
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
    # Per la valutazione mettiamo il modello in eval()
    # Questo congela le statistiche della quantizzazione per un'inferenza stabile
    net.eval()
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
    try:
        from thop import profile
        import torch
    except ImportError:
        return 0, 0
    model = Net()
    model_cpu = model.to('cpu')
    dummy_input = torch.randn(1, 1, 28, 28)
    macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 
    return int(params), int(flops)