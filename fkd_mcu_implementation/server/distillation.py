import torch
import torch.nn as nn
import torch.optim as optim
from net import INPUT_SIZE, evaluate_model

class KnowledgeDistillationManager:
    def __init__(self, temperature=3.0, alpha=0.5, epochs=3, lr=0.001):
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr

    def perform_knowledge_distillation_multi(self, client_models_dict, server_model, proxy_loader):
        print(f"\n[Server] Avvio Knowledge Distillation Multi-Teacher ({len(client_models_dict)} client collegati) -> Student: Server...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        server_model.to(device)
        for client_id, model in client_models_dict.items():
            model.to(device)
            model.eval()  # I modelli dei client fanno da Teacher (bloccati)
            
        server_model.train()  # Il server fa da Student
        
        optimizer = optim.Adam(server_model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs): 
            running_loss = 0.0
            for inputs, labels in proxy_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(-1, INPUT_SIZE)
                
                optimizer.zero_grad()
                
                # Logit dello Student (Server)
                student_logits = server_model(inputs)
                
                # Calcolo dei logit medi da tutti i teacher (clienti attivi)
                with torch.no_grad():
                    teacher_logits_list = [m(inputs) for m in client_models_dict.values()]
                    mean_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
                    
                hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
                
                soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
                soft_teacher = nn.functional.softmax(mean_teacher_logits / self.temperature, dim=1)
                soft_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (self.temperature ** 2)
                
                loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            print(f"[Distillation] Epoca {epoch+1}/{self.epochs} - Loss Media Server: {running_loss/len(proxy_loader):.4f}")
            
        print("[Server] Distillazione Multi-Teacher completata!")
        
        # Calcolo accuracy finale del modello globale (server) sul proxy dataset
        global_accuracy = evaluate_model(server_model, proxy_loader)
        print(f"\n >>> [ACCURACY MODELLO GLOBALE SERVER] -> {global_accuracy:.2f}% <<<\n")
        
        return server_model