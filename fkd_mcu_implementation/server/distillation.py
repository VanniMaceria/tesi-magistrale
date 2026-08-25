import torch
import torch.nn as nn
import torch.nn.functional as F
from net import INPUT_SIZE

class KnowledgeDistillationManager:
    def __init__(self, temperature=3.0, epochs=3, lr=0.01):
        self.temperature = temperature
        self.epochs = epochs
        self.lr = lr

    def calculate_kd_loss(self, y_student, y_teacher):
        soft_teacher = F.softmax(y_teacher / self.temperature, dim=1)
        soft_student = F.log_softmax(y_student / self.temperature, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (self.temperature ** 2)
        return loss

    def perform_knowledge_distillation_multi(self, client_models_dict, server_model, proxy_loader):
        print(f"\n[Server] Avvio Knowledge Distillation Multi-Teacher ({len(client_models_dict)} client collegati) -> Student: Server...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        server_model.to(device)
        server_model.train()
        
        teacher_models = []
        for model in client_models_dict.values():
            model.to(device)
            model.eval()
            teacher_models.append(model)
            
        optimizer = torch.optim.Adam(server_model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in proxy_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch["img"]
                images = images.to(device).view(-1, INPUT_SIZE)
                
                optimizer.zero_grad()
                
                outputs_student = server_model(images)
                
                with torch.no_grad():
                    all_teacher_logits = [t(images) for t in teacher_models]
                    avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)
                
                loss = self.calculate_kd_loss(outputs_student, avg_teacher_logits)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            print(f"[Distillation] Epoca {epoch+1}/{self.epochs} - Loss Media Server: {running_loss/len(proxy_loader):.4f}")
            
        print("[Server] Distillazione Multi-Teacher completata!")
        return server_model