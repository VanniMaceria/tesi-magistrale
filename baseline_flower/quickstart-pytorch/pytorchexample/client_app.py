"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, get_model_iot_metrics
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    current_seed = context.run_config.get("seed", 42)
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    from pytorchexample.task import set_all_seeds
    set_all_seeds(current_seed)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, current_seed)

    # Call the training function
    train_loss, num_examples_locali = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
    )
    
    # Valutazione locale post-training per ottenere l'accuracy del singolo client
    _, accuracy_locale = test_fn(model, valloader, device)

    # ========================================================================
    # CALCOLO METRICHE IOT LOCALI
    # ========================================================================
    num_params, flops_inferenza = get_model_iot_metrics()
    
    # 1. Calcolo Banda (MB)
    # 32 bit sono 4 byte, quel 2 significa che l'operazione viene fatta 2 volte (upload + download)
    # 1024 * 1024 è per convertire da byte a megabyte
    banda_mb = (num_params * 4 * 2) / (1024 * 1024)

    # 2. STIMA DEI FLOPS TOTALI DI TRAINING PER ROUND (LOCALE)
    # La formula calcola il costo computazionale complessivo per QUESTO nodo.
    # 1. flops_inferenza: Costo in FLOPS per singola immagine.
    # 2. Moltiplicatore 3: Approssimazione standard in letteratura per l'addestramento 
    #    (Forward Pass = 1x FLOPs, Backward Pass/Backpropagation = ~2x FLOPs).
    # 3. num_examples_locali: Numero REALE di campioni nel dataset di questo client.
    # 4. local_epochs: Numero di volte che il nodo itera sul proprio dataset.
    # ========================================================================
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

    # Definizione del record dei parametri (Essenziale per evitare NameError)
    model_record = ArrayRecord(model.state_dict())

    # Pack metrics
    metrics = {
        "client_id": partition_id,
        "seed": current_seed,
        "accuracy": float(accuracy_locale),
        "loss": float(train_loss),
        "energia": float(energia_totale_joule),
        "banda": float(banda_mb),
        "flops_inferenza": float(flops_inferenza),
        "num-examples": num_examples_locali,
    }
    
    return Message(
        content=RecordDict({
            "arrays": model_record, # Pesi del modello 
            "metrics": MetricRecord(metrics)
        }), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    current_seed = context.run_config.get("seed", 42) 
    _, valloader = load_data(partition_id, num_partitions, batch_size, current_seed)
    
    eval_loss, eval_acc = test_fn(model, valloader, device)
    
    metrics = {"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(valloader.dataset)}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)