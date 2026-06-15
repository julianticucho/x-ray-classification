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


class XRayEffusionDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Effusion)."""
    
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
            if 'Effusion' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayAtelectasisDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Atelectasis)."""
    
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
            if 'Atelectasis' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayConsolidationDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Consolidation)."""
    
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
            if 'Consolidation' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayInfiltrationDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Infiltration)."""
    
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
            if 'Infiltration' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayPneumothoraxDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Pneumothorax)."""
    
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
            if 'Pneumothorax' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayEdemaDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Edema)."""
    
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
            if 'Edema' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayEmphysemaDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Emphysema)."""
    
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
            if 'Emphysema' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayFibrosisDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Fibrosis)."""
    
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
            if 'Fibrosis' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayPleuralThickeningDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Pleural_Thickening)."""
    
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
            if 'Pleural_Thickening' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayCardiomegalyDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Cardiomegaly)."""
    
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
            if 'Cardiomegaly' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayNoduleDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Nodule)."""
    
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
            if 'Nodule' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    

class XRayMassDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Mass)."""
    
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
            if 'Mass' in labels_str.split('|'):
                label = 1
                
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor
    
    
class XRayHerniaDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Hernia)."""
    
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
            if 'Hernia' in labels_str.split('|'):
                label = 1

        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


class XRayGenderDataset(Dataset):
    """Dataset de radiografías de tórax con etiquetas binarias (Género: M=1, F=0)."""

    def __init__(self, labels_df, img_dir, transform=None):
        self.labels_df = labels_df
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
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

        label = 1 if self.labels_df.iloc[idx]['Patient Gender'] == 'F' else 0
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

