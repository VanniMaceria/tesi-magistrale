import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_kd_loss(y_student, y_teacher, temperature=3.0):
    """
    Calcola la Knowledge Distillation Loss tra l'output dello Student 
    e la media degli output dei Teacher.
    La funzione di loss implementa la divergenza di Kullback-Leibler applicata a distribuzioni
    di probabilità 'ammorbidite' tramite un parametro di temperatura T (T troppo piccolo rende le distribuzioni
    troppo sicure, al contrario T rende la distribuzione piatta). 
    Questo permette di catturare la geometria dello spazio delle classi definita dai modelli 
    Teacher (i client), trasferendo allo Student non solo l'etichetta corretta, ma anche le 
    correlazioni tra le classi non predette.
    """
    # Si usa il Softmax con Temperatura per "ammorbidire" le probabilità
    # e catturare meglio la conoscenza dei Teacher
    soft_teacher = F.softmax(y_teacher / temperature, dim=1)
    soft_student = F.log_softmax(y_student / temperature, dim=1)
    
    # Kullback-Leibler Divergence: misura quanto due distribuzioni sono diverse
    loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
    return loss

def train_distillation(student_model, teacher_models, proxy_loader, device, epochs=1, lr=0.01):
    """
    Esegue l'addestramento dello Student sul Server usando il Proxy Dataset.
    """
    student_model.train()  # imposta il modello in modalità training, ma il training non inizia ancora
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in proxy_loader:
            images = batch["img"].to(device)
            optimizer.zero_grad()
            
            # 1. Otteniamo le predizioni dello Student
            outputs_student = student_model(images)
            
            # 2. Otteniamo le predizioni medie di tutti i Teacher (i client)
            # Portiamo i modelli in eval mode per non sprecare memoria/tempo
            with torch.no_grad():
                all_teacher_logits = []
                for t_model in teacher_models:
                    t_model.to(device)
                    t_model.eval()
                    all_teacher_logits.append(t_model(images))
                
                # Facciamo la media dei logit ricevuti dai client
                avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)
            
            # 3. Calcolo della Loss e aggiornamento
            loss = calculate_kd_loss(outputs_student, avg_teacher_logits)
            loss.backward()
            optimizer.step()
            
    return student_model