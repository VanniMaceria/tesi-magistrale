"""pytorchexample: A Flower / PyTorch app."""

import torch
from collections import OrderedDict
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from pytorchexample.task import Net, load_data, get_model_iot_metrics, get_p_from_id
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

class PyTorchClient(NumPyClient):
    """Flower client implementation."""
    
    def __init__(self, partition_id, num_partitions, run_config):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.run_config = run_config

    def fit(self, parameters, config):
        current_seed = self.run_config.get("seed", 42)
        batch_size = self.run_config["batch-size"]
        local_epochs = self.run_config["local-epochs"]
        lr = config.get("lr", self.run_config.get("learning-rate", 0.01))

        from pytorchexample.task import set_all_seeds
        set_all_seeds(current_seed)

        # Inizializzazione modello
        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the data and profile P
        p_fixed = get_p_from_id(self.partition_id)
        trainloader, valloader = load_data(self.partition_id, self.num_partitions, batch_size, current_seed)

        # Call the training function
        train_loss, num_examples_locali = train_fn(model, trainloader, local_epochs, lr, device, p_fixed=p_fixed)
        
        # Valutazione locale post-training per ottenere l'accuracy del singolo client
        _, accuracy_locale = test_fn(model, valloader, device, p=p_fixed)

        # ========================================================================
        # CALCOLO METRICHE IOT LOCALI (Ordered Dropout)
        # ========================================================================
        num_params, flops_inferenza = get_model_iot_metrics(p=p_fixed)
        
        # 1. Calcolo Banda (MB)
        # 32 bit sono 4 byte, quel 2 significa che l'operazione viene fatta 2 volte (upload + download)
        # 1024 * 1024 è per convertire da byte a megabyte
        banda_mb = (num_params * 4 * 2) / (1024 * 1024)

        # 2. STIMA DEI FLOPS TOTALI DI TRAINING PER ROUND (LOCALE)
        flops_totali_client = flops_inferenza * 3 * num_examples_locali * local_epochs
        
        # ========================================================================
        # SPIEGAZIONE DELLE COSTANTI ENERGETICHE
        # Dalla tabella di Mark Horowitz (2014) fonte -> https://ieeexplore.ieee.org/document/6757323
        # Ragionando con float a 32 bit, un'addizione costa 0.9 pJ e una moltiplicazione 3.7 pJ.
        # Il costo logico totale è quindi di ~4.6 pJ per un'operazione MAC (Multiply-Accumulate).
        # A questo dobbiamo sommare l'overhead dovuto al recupero dei dati dalla memoria cache locale: 
        # leggere da una SRAM da 8KB costa infatti circa 10 pJ. 
        # Non si considera la memoria RAM (DRAM, >1300 pJ) perché l'hardware sfrutta il data reuse, 
        # rendendo l'accesso alla SRAM il costo dominante per singola operazione matematica.
        # Sommando il costo computazionale e quello di memoria locale (4.6 + 10 pJ), si ottiene 
        # un valore di ~15 pJ. Per convenzione nella letteratura TinyML/Edge AI, si approssima questo
        # costo all'ordine di grandezza di 10 picoJoule (1e-11 J), che rappresenta un compromesso 
        # realistico per modellare un'operazione FP32 su dispositivi IoT ottimizzati.
        # ========================================================================
        JOULE_PER_FLOP = 1e-11  # 10 picoJoule per operazione matematica
        JOULE_PER_MB = 0.05     # 50 milliJoule per MB trasmesso (Wi-Fi)
        
        energia_totale_joule = (flops_totali_client * JOULE_PER_FLOP) + (banda_mb * JOULE_PER_MB)

        metrics = {
            "client_id": int(self.partition_id),
            "p_profile": float(p_fixed),
            "seed": int(current_seed),
            "accuracy": float(accuracy_locale),
            "loss": float(train_loss),
            "energia": float(energia_totale_joule),
            "banda": float(banda_mb),
            "flops_inferenza": float(flops_inferenza),
            "num-examples": int(num_examples_locali),
        }
        
        ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return ndarrays, num_examples_locali, metrics

    def evaluate(self, parameters, config):
        current_seed = self.run_config.get("seed", 42)
        batch_size = self.run_config["batch-size"]
        
        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        p_fixed = get_p_from_id(self.partition_id)
        _, valloader = load_data(self.partition_id, self.num_partitions, batch_size, current_seed)
        
        eval_loss, eval_acc = test_fn(model, valloader, device, p=p_fixed)
        
        return float(eval_loss), len(valloader.dataset), {"accuracy": float(eval_acc)}

def client_fn(context: fl.common.Context):
    """Costruisce e restituisce l'istanza del client definita sopra."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    return PyTorchClient(partition_id, num_partitions, context.run_config).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)