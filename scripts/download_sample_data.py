"""Script para descargar datos de muestra de Kaggle."""
import os
import kagglehub

if __name__ == '__main__':
    os.environ["KAGGLEHUB_CACHE"] = "."
    path = kagglehub.dataset_download("nih-chest-xrays/sample")
    print("Path to dataset files:", path)