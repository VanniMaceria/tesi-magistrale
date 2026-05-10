"""pytorchexample: A Flower / PyTorch app."""

import torch
import os
import csv
import time
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test, set_all_seeds

app = ServerApp()

latest_metrics = {"energia": 0.0, "banda": 0.0, "flops_inf": 0.0, "acc": 0.0, "loss": 0.0}
current_round = 0
CSV_COLUMNS = ["id_esperimento", "seed", "round", "accuracy", "loss", "energia(J)", "banda(MB)", "flops_inferenza", "samples"]

def get_next_experiment_id():
    server_file = "results/server/server_aggregate.csv"
    if not os.path.exists(server_file): return 1
    try:
        with open(server_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2: return 1
            last_line = lines[-1].split(",")
            return int(last_line[0]) + 1
    except Exception: return int(time.time())

ID_ESPERIMENTO = get_next_experiment_id()

def prepare_model_for_qat(model):
    # =============================================================================================
    # GIUSTIFICAZIONE DELLA "FAKE QUANTIZATION" (QAT)
    # =============================================================================================
    # I motori di calcolo standard (PyTorch) non permettono la backpropagation 
    # direttamente su tipi di dato intero (int8) perché i gradienti, essendo molto piccoli, 
    # verrebbero arrotondati a zero, bloccando l'apprendimento del modello.
    #
    # Abbiamo quindi adottato la "Fake Quantization" per i seguenti motivi:
    #
    # 1. SIMULAZIONE REALISTICA: Durante il forward pass, i pesi vengono arrotondati e 
    #    limitati (clipping) all'intervallo degli 8-bit (-128, 127). Questo permette di 
    #    misurare l'impatto reale della perdita di precisione sull'accuratezza finale.
    #
    # 2. TRAINING STABILE: Durante il backward pass, il modello aggiorna i "pesi latenti" 
    #    in float32. Questo permette di mantenere la precisione necessaria per accumulare 
    #    i gradienti, simulando però un modello che si comporta come se fosse a 8-bit.
    #
    # 3. COMPATIBILITÀ IOT: Questa tecnica prepara il modello alla "Post-Training 
    #    Quantization" finale. Il modello così addestrato è già "abituato" agli errori 
    #    di arrotondamento e funzionerà meglio una volta convertito in int8 per Arduino.
    #
    # 4. OTTIMIZZAZIONE BANDA: In un sistema reale, questo ci permette di trasmettere 
    #    solo i valori interi (1 byte per peso), riducendo così il carico sulla rete.
    # =============================================================================================
    model.train() # Obbligatorio essere in train() per la configurazione QAT
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack') # Setup per ARM/Microcontrollori (8 bit)
    return torch.ao.quantization.prepare_qat(model, inplace=True)

def aggregate_fit_metrics(replies: list, weight_key: str) -> MetricRecord:
    global latest_metrics, current_round
    current_round += 1
    count = len(replies)
    
    if count > 0:
        os.makedirs("results/clients", exist_ok=True)
        os.makedirs("results/server", exist_ok=True)
        
        sums = {k: 0.0 for k in ["acc", "loss", "energia", "banda", "flops_inf", "samples"]}
        seed_corrente = replies[0]["metrics"]["seed"]

        for reply in replies:
            m = reply["metrics"] 
            c_id = int(m["client_id"])
            
            sums["acc"] += m["accuracy"]
            sums["loss"] += m["loss"]
            sums["energia"] += m["energia"]
            sums["banda"] += m["banda"]
            sums["flops_inf"] += m["flops_inferenza"]
            sums["samples"] += m["num-examples"]
            
            client_file = f"results/clients/client_{c_id}.csv"
            file_exists = os.path.isfile(client_file)
            with open(client_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists: writer.writerow(CSV_COLUMNS)
                writer.writerow([
                    ID_ESPERIMENTO, int(m["seed"]), current_round, 
                    f"{m['accuracy']:.4e}", f"{m['loss']:.4e}", f"{m['energia']:.4e}", 
                    f"{m['banda']:.4e}", f"{m['flops_inferenza']:.4e}", int(m["num-examples"])
                ])

        latest_metrics["acc"] = sums["acc"] / count
        latest_metrics["loss"] = sums["loss"] / count
        latest_metrics["energia"] = sums["energia"] / count
        latest_metrics["banda"] = sums["banda"] / count
        latest_metrics["flops_inf"] = sums["flops_inf"] / count
        latest_metrics["samples"] = sums["samples"] / count

        server_file = "results/server/server_aggregate.csv"
        file_exists = os.path.isfile(server_file)
        with open(server_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(CSV_COLUMNS)
            writer.writerow([
                ID_ESPERIMENTO, int(seed_corrente), current_round, 
                f"{latest_metrics['acc']:.4e}", f"{latest_metrics['loss']:.4e}", 
                f"{latest_metrics['energia']:.4e}", f"{latest_metrics['banda']:.4e}", 
                f"{latest_metrics['flops_inf']:.4e}", int(latest_metrics["samples"])
            ])

    return MetricRecord({})

@app.main()
def main(grid: Grid, context: Context) -> None:
    os.makedirs("results/clients", exist_ok=True)
    os.makedirs("results/server", exist_ok=True)

    current_seed = context.run_config.get("seed", 42)
    set_all_seeds(current_seed)
    
    # Preparazione modello globale con struttura QAT per la corretta serializzazione
    global_model = prepare_model_for_qat(Net())
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_evaluate=context.run_config["fraction-evaluate"],
        train_metrics_aggr_fn=aggregate_fit_metrics,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=context.run_config["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    global latest_metrics
    
    # Prepara il modello server prima di caricare i pesi aggregati
    model = prepare_model_for_qat(Net())
    model.load_state_dict(arrays.to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "avg_energy_joule": float(latest_metrics["energia"]),
        "avg_banda_mb": float(latest_metrics["banda"]),
        "avg_flops_inferenza": float(latest_metrics["flops_inf"]),
    })