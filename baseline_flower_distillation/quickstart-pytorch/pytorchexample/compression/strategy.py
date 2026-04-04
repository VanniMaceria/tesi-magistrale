import torch
from flwr.serverapp.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from pytorchexample.task import SmallNet
from pytorchexample.compression.distiller import train_distillation

class DistillationStrategy(FedAvg):
    def __init__(self, proxy_loader, device, **kwargs):
        super().__init__(**kwargs)
        self.proxy_loader = proxy_loader
        self.device = device

    # Ogni classe che eredita FedAvg può sovrascrivere questo metodo per implementare la propria logica di aggregazione dei modelli
    # Viene chiamato ad ogni round
    def aggregate_fit(self, server_round, results, failures):
        """
        Invece di fare la media dei pesi (FedAvg), usa i modelli SmallNet dei client 
        come Teacher per addestrare il nuovo modello globale SmallNet tramite distillazione.
        """
        if not results:
            return None, {}

        # 1. Ricostruiamo i modelli "Teacher" (SmallNet) dai pesi ricevuti dai client
        teacher_models = []
        for _, fit_res in results:
            # Convertiamo i parametri Flower (byte) in array NumPy
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            
            # Creiamo un'istanza di SmallNet
            t_model = SmallNet().to(self.device)
            
            # Carichiamo i pesi nel modello
            params_dict = zip(t_model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            t_model.load_state_dict(state_dict)
            
            t_model.eval() # Fondamentale: i Teacher devono essere in modalità valutazione
            teacher_models.append(t_model)

        # 2. Prepariamo lo "Student" (il modello globale SmallNet che verrà distribuito)
        # Partiamo da una nuova SmallNet
        student_model = SmallNet().to(self.device)

        # 3. Eseguiamo la Knowledge Distillation lato Server
        # Questa funzione userà il proxy_loader per far 'parlare' i modelli
        print(f"\n[SERVER] Round {server_round}: Avvio Distillazione su {len(teacher_models)} modelli SmallNet...")
        
        updated_student = train_distillation(
            student_model=student_model,
            teacher_models=teacher_models,
            proxy_loader=self.proxy_loader,
            device=self.device,
            epochs=1,
            lr=0.01
        )

        # 4. Convertiamo il modello Student aggiornato in parametri Flower per la distribuzione
        new_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in updated_student.state_dict().items()]
        )

        # 5. Aggreghiamo le metriche IoT (energia, banda, etc.) inviate dai client
        metrics_aggregated = {}
        if self.train_metrics_aggr_fn:
            metrics_aggregated = self.train_metrics_aggr_fn(results)

        return new_parameters, metrics_aggregated