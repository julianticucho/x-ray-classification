import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.datasets import XRayDataset, XRayBinaryDataset


class DataLoaderFactory:
    """Fábrica para crear DataLoaders con diferentes configuraciones."""
    
    def __init__(self, labels_df, img_dir, num_workers=1, is_train=False):
        """Inicializa la fábrica con el DataFrame de etiquetas y directorio(s) de imágenes."""
        self.labels_df = labels_df
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.num_workers = num_workers
        self.is_train = is_train
    
    def get_available_configurations(self):
        """Retorna las configuraciones de DataLoader disponibles."""
        return {
            'basic': self.create_basic,
            'binary': self.create_binary,
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

    def create_binary(self):
        """Crea un DataLoader binario con resize, normalize y random flip (si es train)."""
        transform_list = [
            transforms.Resize((224, 224))
        ]
        if self.is_train:
            transform_list.append(transforms.RandomHorizontalFlip())
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform = transforms.Compose(transform_list)
        dataset = XRayBinaryDataset(self.labels_df, self.img_dir, transform=transform)
        return DataLoader(dataset, batch_size=16, shuffle=self.is_train, num_workers=self.num_workers)

    

