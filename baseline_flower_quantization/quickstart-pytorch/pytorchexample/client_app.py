"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, get_model_iot_metrics
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

app = ClientApp()

def prepare_local_model_for_qat(model):
    """Uguale al server: modifica la topologia per accettare i parametri di quantizzazione."""
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    return torch.ao.quantization.prepare_qat(model, inplace=True)

@app.train()
def train(msg: Message, context: Context):
    current_seed = context.run_config.get("seed", 42)
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    from pytorchexample.task import set_all_seeds
    set_all_seeds(current_seed)

    # 1. Inizializza modello vuoto
    model = Net()
    
    # 2. ESPANSIONE MODELLO (FONDAMENTALE)
    # Aggiungiamo i buffer di quantizzazione *prima* del load_state_dict.
    # Così il dizionario inviato dal Server corrisponderà perfettamente alla struttura del Client.
    model = prepare_local_model_for_qat(model)
    
    # 3. Caricamento Pesi
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, current_seed)

    # 4. TRAINING SIMULATO INT8
    # Grazie alla Fake Quantization, i calcoli subiranno perdite di precisione (clipping/arrotondamento),
    # ma PyTorch potrà comunque calcolare i gradienti usando i "pesi latenti" float32 sottostanti.
    train_loss, num_examples_locali = train_fn(model, trainloader, local_epochs, lr, device)
    _, accuracy_locale = test_fn(model, valloader, device)

    # ========================================================================
    # CALCOLO METRICHE IOT LOCALI (SCENARIO 8-BIT)
    # ========================================================================
    num_params, flops_inferenza = get_model_iot_metrics()
    
    # SIMULAZIONE BANDA INT8: 
    # Stiamo lavorando con int8 quindi ogni peso è 1 byte
    banda_mb = (num_params * 1 * 2) / (1024 * 1024)

    flops_totali_client = flops_inferenza * 3 * num_examples_locali * local_epochs
    JOULE_PER_FLOP = 1e-11  
    JOULE_PER_MB = 0.05     
    energia_totale_joule = (flops_totali_client * JOULE_PER_FLOP) + (banda_mb * JOULE_PER_MB)

    model_record = ArrayRecord(model.state_dict())

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
        content=RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)}), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model = prepare_local_model_for_qat(model) # Stessa precauzione del train
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