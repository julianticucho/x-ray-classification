import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import load_model, load_test_dataloader
from src.diagnostics import plot_confusion_matrix_binary

def plot_confusion_and_save(model_path, output_path, threshold=0.5):
    """Plots the confusion matrix and saves it to a file."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado")
        sys.exit(1)
    model = load_model(model_path)
    dataloader_test = load_test_dataloader(model_path)
    fig = plot_confusion_matrix_binary(model, dataloader_test, threshold=threshold)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"Matriz de confusión guardada en: {output_path}")

if __name__ == '__main__':
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_binary_binary_low_res_densenet_binary_1epoch.pdf'
    # )
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth', 
    #     output_path='results/confusion/full_data_binary_binary_low_res_densenet_binary_15epochs.pdf'
    # )
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_binary_binary_densenet_binary_1epoch.pdf'
    # )
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth', 
    #     output_path='results/confusion/full_data_binary_binary_densenet_binary_15epochs.pdf'
    # )
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pdf',
    #     threshold=0.5
    # )
    plot_confusion_and_save(
        model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth',
        output_path='results/confusion/full_data_effusion_effusion_densenet_binary_15epochs.pdf',
        threshold=0.5
    )

