"""pytorchexample: A Flower / PyTorch app."""

import torch
import os
import csv
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import SmallNet, load_centralized_dataset, test, set_all_seeds
from pytorchexample.compression.strategy import DistillationStrategy
from pytorchexample.task import load_proxy_dataset

# Student nella Knowledge Distillation: è il modello globale che viene addestrato sul server usando i modelli dei client come Teacher.
app = ServerApp()

# Variabili globali per il monitoraggio
latest_metrics = {"energia": 0.0, "banda": 0.0, "flops_inf": 0.0, "acc": 0.0, "loss": 0.0}
current_round = 0
CSV_COLUMNS = ["id_esperimento", "seed", "round", "accuracy", "loss", "energia(J)", "banda(MB)", "flops_inferenza", "samples"]

def get_next_experiment_id():
    server_file = "results/server/server_aggregate.csv"
    # Se il file non esiste, è il primo esperimento assoluto
    if not os.path.exists(server_file):
        return 1
    try:
        with open(server_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2:  # Solo header o file vuoto
                return 1
            # Prendi l'ID dall'ultima riga valida
            last_line = lines[-1].split(",")
            return int(last_line[0]) + 1
    except Exception as e:
        print(f"[ATTENZIONE] Errore lettura ID esperimento: {e}")
        import time
        return int(time.time())

# Determina l'ID progressivo all'avvio
ID_ESPERIMENTO = get_next_experiment_id()

def aggregate_fit_metrics(replies: list, weight_key: str) -> MetricRecord:
    """Aggrega le metriche dai client e salva in CSV con notazione scientifica."""
    global latest_metrics, current_round
    current_round += 1
    count = len(replies)
    
    if count > 0:
        os.makedirs("results/clients", exist_ok=True)
        os.makedirs("results/server", exist_ok=True)
        
        sums = {k: 0.0 for k in ["acc", "loss", "energy", "banda", "flops_inf", "samples"]}
        seed_corrente = replies[0]["metrics"]["seed"]

        for reply in replies:
            m = reply["metrics"] 
            c_id = int(m["client_id"])
            
            # Accumulo per medie server
            sums["acc"] += m["accuracy"]
            sums["loss"] += m["loss"]
            sums["energy"] += m["energia"]
            sums["banda"] += m["banda"]
            sums["flops_inf"] += m["flops_inferenza"]
            sums["samples"] += m["num-examples"]
            
            # Scrittura CSV Client in NOTAZIONE SCIENTIFICA (.4e)
            client_file = f"results/clients/client_{c_id}.csv"
            file_exists = os.path.isfile(client_file)
            with open(client_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists: 
                    writer.writerow(CSV_COLUMNS)
                writer.writerow([
                    ID_ESPERIMENTO, 
                    int(m["seed"]), 
                    current_round, 
                    f"{m['accuracy']:.4e}", 
                    f"{m['loss']:.4e}", 
                    f"{m['energia']:.4e}", 
                    f"{m['banda']:.4e}", 
                    f"{m['flops_inferenza']:.4e}",
                    int(m["num-examples"])
                ])

        # Calcolo Medie aggregate per il Server
        latest_metrics["acc"] = sums["acc"] / count
        latest_metrics["loss"] = sums["loss"] / count
        latest_metrics["energia"] = sums["energy"] / count
        latest_metrics["banda"] = sums["banda"] / count
        latest_metrics["flops_inf"] = sums["flops_inf"] / count
        latest_metrics["samples"] = sums["samples"] / count

        # Scrittura CSV Server (Stesse colonne, valori medi) in NOTAZIONE SCIENTIFICA
        server_file = "results/server/server_aggregate.csv"
        file_exists = os.path.isfile(server_file)
        with open(server_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(CSV_COLUMNS)
            writer.writerow([
                ID_ESPERIMENTO, 
                int(seed_corrente), 
                current_round, 
                f"{latest_metrics['acc']:.4e}", 
                f"{latest_metrics['loss']:.4e}", 
                f"{latest_metrics['energia']:.4e}", 
                f"{latest_metrics['banda']:.4e}", 
                f"{latest_metrics['flops_inf']:.4e}",
                int(latest_metrics["samples"])
            ])

    return MetricRecord({})

@app.main()
def main(grid: Grid, context: Context) -> None:
    os.makedirs("results/clients", exist_ok=True)
    os.makedirs("results/server", exist_ok=True)

    # Recupero configurazioni dal contesto
    current_seed = context.run_config.get("seed", 42)
    set_all_seeds(current_seed)
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    lr = context.run_config["learning-rate"]

    # Rilevamento dispositivo (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Server avviato su dispositivo: {device} ---")

    # 1. Carichiamo il Proxy Dataset per la Distillazione (500 campioni da test split)
    # Passiamo il seed per garantire la riproducibilità dello slicing
    proxy_loader = load_proxy_dataset(num_samples=500, seed=current_seed)

    # 2. Inizializziamo il modello globale (Student)
    global_model = SmallNet()
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Usiamo la DistillationStrategy custom invece di FedAvg
    # Passiamo il proxy_loader e il device alla strategia
    strategy = DistillationStrategy(
        proxy_loader=proxy_loader,
        device=device,
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=aggregate_fit_metrics,
    )

    # Avvio della simulazione Flower
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Valutazione centralizzata con report IoT arricchito."""
    global latest_metrics
    
    model = SmallNet()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    # Inseriamo tutto nel MetricRecord finale per i log a video
    return MetricRecord({
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "avg_energy_joule": float(latest_metrics["energia"]),
        "avg_banda_mb": float(latest_metrics["banda"]),
        "avg_flops_inferenza": float(latest_metrics["flops_inf"]),
    })