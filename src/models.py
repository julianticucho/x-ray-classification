import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import torchvision.models as torchvision_models

class ModelFactory:
    """Fábrica para crear modelos con diferentes configuraciones."""
    
    def __init__(self):
        pass
    
    def get_available_configurations(self):
        """Retorna las configuraciones de modelo disponibles."""
        return {
            'mlp': self.create_mlp,
            'densenet_binary': self.create_densenet_binary,
        }

    def get(self, model_name):
        """Retorna un modelo con la configuración especificada."""
        configs = self.get_available_configurations()
        if model_name not in configs:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(configs.keys())}")
        return configs[model_name]()
    
    def create_mlp(self):
        """Crea un modelo MLP con 14 clases."""
        return MLP(num_classes=14)
    
    def create_densenet_binary(self):
        """Crea un modelo DenseNet-121 para clasificación binaria."""
        return DenseNetBinary()
    

class BaseModel(nn.Module, ABC):
    """Clase base abstracta para modelos con método de entrenamiento."""
    
    def train_model(self, dataloader, val_dataloader=None, lr=0.001, device=None, num_epochs=10):
        """Entrena el modelo con el dataloader proporcionado."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
            
            if val_dataloader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = self(images)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                epoch_val_loss = val_loss / len(val_dataloader)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")
        
        return self


class MLP(BaseModel):
    """Perceptrón Multicapa de ejemplo."""
    
    def __init__(self, num_classes=15, input_size=224*224*3):
        """Inicializa el MLP con capas totalmente conectadas."""
        super(MLP, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Pasa la entrada a través de las capas del MLP."""
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class DenseNetBinary(BaseModel):
    """DenseNet-121 para clasificación binaria de Neumonía."""
    
    def __init__(self):
        super(DenseNetBinary, self).__init__()
        self.model = torchvision_models.densenet121(weights=torchvision_models.DenseNet121_Weights.DEFAULT)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 1)
        self.criterion = None
        self.optimizer = None
        
    def forward(self, x):
        return self.model(x)
        
    def train_model(self, dataloader, val_dataloader=None, lr=0.001, device=None, num_epochs=30, patience=7):
        """Sobrescribe train_model para calcular pesos, usar Weighted BCE, scheduler, y early stopping."""
        import copy
        
        print("Calculando pesos de clases (w_pos, w_neg) a partir del dataloader...")
        num_pos = 0
        num_neg = 0
        for _, labels in dataloader:
            pos = labels.sum().item()
            num_pos += pos
            num_neg += (labels.shape[0] - pos)
            
        total = num_pos + num_neg
        w_pos = num_neg / total if total > 0 else 0
        w_neg = num_pos / total if total > 0 else 0
        print(f"Pesos calculados: w_pos={w_pos:.4f}, w_neg={w_neg:.4f}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
            
        w_pos_tensor = torch.tensor(w_pos, dtype=torch.float32).to(device)
        w_neg_tensor = torch.tensor(w_neg, dtype=torch.float32).to(device)
        
        def weighted_loss(logits, targets):
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            weight_matrix = targets * w_pos_tensor + (1 - targets) * w_neg_tensor
            return (bce * weight_matrix).mean()
            
        self.criterion = weighted_loss
        
        # training logic specific to DenseNetBinary
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.state_dict())
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
            
            if val_dataloader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = self(images)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        
                epoch_val_loss = val_loss / len(val_dataloader)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")
                
                scheduler.step(epoch_val_loss)
                
                if epoch_val_loss < best_val_loss:
                    print(f"Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...")
                    best_val_loss = epoch_val_loss
                    best_model_wts = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"EarlyStopping counter: {patience_counter} out of {patience}")
                    
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
                    
        if val_dataloader is not None:
            print(f"Training complete. Best val loss: {best_val_loss:.4f}")
            self.load_state_dict(best_model_wts)
            
        return self
    
