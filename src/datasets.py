import torch
from torch.utils.data import Dataset
from PIL import Image
import os


ALL_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]


class XRayDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas multi-label."""
    
    def __init__(self, labels_df, img_dir, transform=None):
        """Inicializa el dataset con DataFrame de etiquetas y directorio(es) de imágenes."""
        self.labels_df = labels_df
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.transform = transform
        
    def __len__(self):
        """Retorna el número de muestras en el dataset."""
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """Retorna la imagen y el vector de etiquetas para el índice dado."""
        img_name = self.labels_df.iloc[idx]['Image Index']
        img_path = None
        for directory in self.img_dir:
            path = os.path.join(directory, img_name)
            if os.path.exists(path):
                img_path = path
                break
        if img_path is None:
            raise FileNotFoundError(f"Imagen {img_name} no encontrada en ningún directorio")
        image = Image.open(img_path).convert('RGB')
        labels_str = self.labels_df.iloc[idx]['Finding Labels']
        label_vector = torch.zeros(len(ALL_LABELS))
        if labels_str != 'No Finding':
            for label in labels_str.split('|'):
                if label in ALL_LABELS:
                    label_vector[ALL_LABELS.index(label)] = 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_vector

class XRayBinaryDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Pneumonía)."""
    
    def __init__(self, labels_df, img_dir, transform=None):
        """Inicializa el dataset con DataFrame de etiquetas y directorio(es) de imágenes."""
        self.labels_df = labels_df
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.transform = transform
        
    def __len__(self):
        """Retorna el número de muestras en el dataset."""
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """Retorna la imagen y la etiqueta binaria para el índice dado."""
        img_name = self.labels_df.iloc[idx]['Image Index']
        img_path = None
        for directory in self.img_dir:
            path = os.path.join(directory, img_name)
            if os.path.exists(path):
                img_path = path
                break
        if img_path is None:
            raise FileNotFoundError(f"Imagen {img_name} no encontrada en ningún directorio")
        image = Image.open(img_path).convert('RGB')
        
        labels_str = self.labels_df.iloc[idx]['Finding Labels']
        label = 0
        if labels_str != 'No Finding':
            if 'Pneumonia' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
