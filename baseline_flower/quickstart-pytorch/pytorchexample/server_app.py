"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import get_model_iot_metrics
from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.task import set_all_seeds

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. Legge il seed dal terminale (se manca usa 42)
    current_seed = context.run_config.get("seed", 42)
    
    # Imposta i seed globali
    set_all_seeds(current_seed)

    # 2. Legge gli altri parametri dal TOML (context.run_config li contiene già)
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    lr = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    # Posso calcolare le metriche in server_app perchè l'hardware è omogeneo e la rete è uguale per tutti

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Ottengo le metriche sul numero di parametri e FLOPS per inferenza del modello
    num_params, flops_inferenza = get_model_iot_metrics()
    
    # 1. Calcolo Banda (MB)
    # 32 bit sono 4 byte, quel 2 significa che l'operazione viene fatta 2 volte (upload + download)
    # 1024 * 1024 è per convertire da byte a megabyte
    banda_mb = (num_params * 4 * 2) / (1024 * 1024)

    # 2. Calcolo Consumo Energetico Stimato
    NUM_CLIENTS = 10 
    LOCAL_EPOCHS = 1
    IMMAGINI_TOTALI_TRAIN = 60000 * 0.8 # MNIST ha 60k immagini, l'80% è training
    
    # ========================================================================
    # CALCOLO DEL CARICO DATI MEDIO E GIUSTIFICAZIONE MATEMATICA
    # Perché questo calcolo funziona anche nello scenario Non-IID?
    # Il consumo energetico di un nodo è direttamente proporzionale al numero
    # di immagini che elabora (Consumo = Numero_Immagini * Costo_Fisso_Immagine).
    # Per la proprietà distributiva della moltiplicazione, fare la media dei consumi:
    #   (Img_Nodo1 * Costo + Img_Nodo2 * Costo + ...) / Numero_Client
    # equivale matematicamente a raccogliere il "Costo" a fattor comune:
    #   Costo * (Immagini_Totali / Numero_Client)
    # ========================================================================
    immagini_medie_nodo = IMMAGINI_TOTALI_TRAIN / NUM_CLIENTS
    
    # ========================================================================
    # STIMA DEI FLOPS TOTALI DI TRAINING PER ROUND
    # La formula calcola il costo computazionale complessivo per un nodo.
    # 1. flops_inferenza: Costo in FLOPS per singola immagine.
    # 2. Moltiplicatore 3: Approssimazione standard in letteratura per l'addestramento 
    #    (Forward Pass = 1x FLOPs, Backward Pass/Backpropagation = ~2x FLOPs).
    # 3. immagini_medie: Numero di campioni nel dataset locale del nodo.
    # 4. LOCAL_EPOCHS: Numero di volte che il nodo itera sull'intero dataset locale.
    # ========================================================================
    flops_totali_nodo = flops_inferenza * 3 * immagini_medie_nodo * LOCAL_EPOCHS
    
    # ========================================================================
    # SPIEGAZIONE DELLE COSTANTI ENERGETICHE
    # Dalla tabella di Mark Horowitz (2014) fonte -> https://ieeexplore.ieee.org/document/6757323
    # Ragionando con float a 32 bit, un'addizione costa 0.9 pJ e una moltiplicazione 3.7 pJ. 
    # Il costo logico totale è quindi di ~4.6 pJ per un'operazione MAC (Multiply-Accumulate).
    # A questo dobbiamo sommare l'overhead dovuto al recupero dei dati (pesi e attivazioni) 
    # dalla memoria cache locale: leggere da una SRAM da 8KB costa infatti circa 10 pJ.
    # Non si considera la memoria RAM (DRAM, >1300 pJ) perché l'hardware sfrutta il data reuse, 
    # rendendo l'accesso alla SRAM il costo dominante per singola operazione matematica.
    # Sommando il costo computazionale e quello di memoria locale (4.6 + 10 pJ), si ottiene 
    # un valore di ~15 pJ. Per convenzione nella letteratura TinyML/Edge AI, si approssima questo
    # costo all'ordine di grandezza di 10 picoJoule (1e-11 J), che rappresenta un compromesso 
    # realistico per modellare un'operazione FP32 su dispositivi IoT ottimizzati.
    # ========================================================================
    JOULE_PER_FLOP = 1e-11  # 10 picoJoule per operazione matematica
    JOULE_PER_MB = 0.05     # 50 milliJoule per MB trasmesso (Wi-Fi)
    
    energia_calcolo = flops_totali_nodo * JOULE_PER_FLOP
    energia_comunicazione = banda_mb * JOULE_PER_MB
    energia_totale_joule = energia_calcolo + energia_comunicazione

    # Return the evaluation metrics
    return MetricRecord({
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "flops_inferenza": float(flops_inferenza),
        "banda_nodo_mb": float(banda_mb),
        "energia_nodo_joule": float(energia_totale_joule)
    })
