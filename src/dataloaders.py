import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.datasets import XRayDataset


class DataLoaderFactory:
    """Fábrica para crear DataLoaders con diferentes configuraciones."""
    
    def __init__(self, labels_df, img_dir, num_workers=1):
        """Inicializa la fábrica con el DataFrame de etiquetas y directorio(s) de imágenes."""
        self.labels_df = labels_df
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.num_workers = num_workers
    
    def get_available_configurations(self):
        """Retorna las configuraciones de DataLoader disponibles."""
        return {
            'basic': self.create_basic,
        }

    def get(self, config_name):
        """Retorna un DataLoader con la configuración especificada."""
        configs = self.get_available_configurations()
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        return configs[config_name]()
    
    def create_basic(self):
        """Crea un DataLoader con configuración básica (resize 224x224, batch_size=16)."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = XRayDataset(self.labels_df, self.img_dir, transform=transform)
        return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=self.num_workers)

    

