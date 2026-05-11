import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm

class ModelFactory:
    """Fábrica para crear modelos con diferentes configuraciones."""
    
    def __init__(self):
        pass
    
    def get_available_configurations(self):
        """Retorna las configuraciones de modelo disponibles."""
        return {
            'mlp': self.create_mlp,
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
    

class BaseModel(nn.Module, ABC):
    """Clase base abstracta para modelos con método de entrenamiento."""
    
    def train_model(self, dataloader, lr=0.001, device=None, num_epochs=10):
        """Entrena el modelo con el dataloader proporcionado."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
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
    
