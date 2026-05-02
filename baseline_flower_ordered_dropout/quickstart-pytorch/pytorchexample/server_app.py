"""pytorchexample: A Flower / PyTorch app."""

import torch
import os
import csv
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test, set_all_seeds
import math

# Variabili globali per il monitoraggio
latest_metrics = {"energia": 0.0, "banda": 0.0, "flops_inf": 0.0, "acc": 0.0, "loss": 0.0}
current_round = 0
CSV_COLUMNS = ["id_esperimento", "seed", "round", "accuracy", "loss", "energia(J)", "banda(MB)", "flops_inferenza", "samples"]

def get_next_experiment_id():
    server_file = "results/server/server_aggregate.csv"
    if not os.path.exists(server_file):
        return 1
    try:
        with open(server_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2:  # Solo header o file vuoto
                return 1
            last_line = lines[-1].split(",")
            return int(last_line[0]) + 1
    except Exception as e:
        print(f"[ATTENZIONE] Errore lettura ID esperimento: {e}")
        import time
        return int(time.time())

ID_ESPERIMENTO = get_next_experiment_id()

def aggregate_fit_metrics(results: list) -> dict:
    """Aggrega le metriche dai client e salva in CSV con notazione scientifica."""
    global latest_metrics, current_round
    current_round += 1
    count = len(results)
    
    if count > 0:
        os.makedirs("results/clients", exist_ok=True)
        os.makedirs("results/server", exist_ok=True)
        
        sums = {k: 0.0 for k in ["acc", "loss", "energy", "banda", "flops_inf", "samples"]}
        
        # results è una lista di tuple: (num_examples, metrics)
        _, first_metrics = results[0]
        seed_corrente = first_metrics.get("seed", 42)

        CLIENT_CSV_COLUMNS = ["id_esperimento", "seed", "round", "p_profile", "accuracy", "loss", "energia(J)", "banda(MB)", "flops_inferenza", "samples"]
        SERVER_CSV_COLUMNS = ["id_esperimento", "seed", "round", "accuracy", "loss", "energia(J)", "banda(MB)", "flops_inferenza", "samples"]

        for _, m in results:
            c_id = int(m["client_id"])
            p_val = float(m.get("p_profile", 1.0))
            
            # Accumulo per medie server
            sums["acc"] += m["accuracy"]
            
            # Gestione sicura dei NaN per non far saltare la media del server
            loss_val = m["loss"]
            if not math.isnan(loss_val):
                sums["loss"] += loss_val
                
            sums["energy"] += m["energia"]
            sums["banda"] += m["banda"]
            sums["flops_inf"] += m["flops_inferenza"]
            sums["samples"] += m["num-examples"]
            
            client_file = f"results/clients/client_{c_id}.csv"
            file_exists = os.path.isfile(client_file)
            with open(client_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists: 
                    writer.writerow(CLIENT_CSV_COLUMNS)
                writer.writerow([
                    ID_ESPERIMENTO, 
                    int(m["seed"]), 
                    current_round, 
                    f"{p_val:.2f}", 
                    f"{m['accuracy']:.4e}", 
                    f"{loss_val:.4e}", 
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

        # Scrittura CSV Server (Usa SERVER_CSV_COLUMNS)
        server_file = "results/server/server_aggregate.csv"
        file_exists = os.path.isfile(server_file)
        with open(server_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(SERVER_CSV_COLUMNS)
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

    return {"accuracy": latest_metrics["acc"]}

def global_evaluate(server_round: int, parameters_ndarrays, config) -> tuple:
    """Valutazione centralizzata per l'Ordered Dropout su p diversi."""
    from collections import OrderedDict
    
    model = Net()
    
    # QUI IL FIX: parameters_ndarrays è già la lista che ci serve, non serve parameters_to_ndarrays
    params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
    
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    
    results = {}
    loss_p1 = 0.0
    for p in [0.25, 0.5, 0.75, 1.0]:
        loss, acc = test(model, test_dataloader, device, p=p)
        results[f"acc_p_{p}"] = float(acc)
        results[f"loss_p_{p}"] = float(loss)
        if p == 1.0:
            loss_p1 = float(loss)
            
    print(f"Round {server_round} - Acc Full (p=1.0): {results['acc_p_1.0']:.4f} | Acc Pruned (p=0.25): {results['acc_p_0.25']:.4f}")
    
    return loss_p1, results

def server_fn(context: Context) -> ServerAppComponents:
    """Costruisce e restituisce i componenti della ServerApp."""
    current_seed = context.run_config.get("seed", 42)
    set_all_seeds(current_seed)
    
    num_rounds = context.run_config.get("num-server-rounds", 15)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.5)
    lr = context.run_config.get("learning-rate", 0.01)

    # Inizializziamo il modello per ricavare i parametri iniziali
    initial_model = Net()
    ndarrays = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # Configuriamo la strategia standard FedAvg
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=fraction_evaluate,
        on_fit_config_fn=lambda server_round: {"lr": lr},
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_fn=global_evaluate,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)