import torch
import os
import pandas as pd
from src.preprocessing import PreprocessingConfigFactory
from src.dataloaders import DataLoaderFactory
from src.models import ModelFactory

def save_model(
    filepath,
    model, 
    preprocessing_name,
    dataloader_name, 
    model_name,
    labels_dir,
    img_dir, 
):
    """Guarda el modelo y su configuración en un archivo."""
    save_dict = {
        'state_dict': model.state_dict(),
        'preprocessing_name': preprocessing_name,
        'dataloader_name': dataloader_name,
        'model_name': model_name,
        'labels_dir': labels_dir,
        'img_dir': img_dir
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(save_dict, filepath)
    print(f"Model '{model_name}' saved to {filepath}")


def load_model(filepath):
    """Carga un modelo desde un archivo guardado."""
    save_dict = torch.load(filepath)
    model_factory = ModelFactory()
    model = model_factory.get(save_dict['model_name'])
    model.load_state_dict(save_dict['state_dict'])
    return model

def load_test_dataloader(filepath):
    """Carga el DataLoader de test desde un archivo de modelo guardado."""
    save_dict = torch.load(filepath)
    labels_df = pd.read_csv(save_dict['labels_dir'])
    pp_factory = PreprocessingConfigFactory(labels_df)
    _, test_df = pp_factory.get(save_dict['preprocessing_name'])
    dl_factory = DataLoaderFactory(test_df, save_dict['img_dir'])
    return dl_factory.get(save_dict['dataloader_name'])


    