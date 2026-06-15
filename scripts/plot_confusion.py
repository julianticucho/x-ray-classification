import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import load_model, load_test_dataloader
from src.diagnostics import plot_confusion_matrix_binary

def plot_confusion_and_save(model_path, output_path, threshold=0.5, title='Confusion Matrix'):
    """Plots the confusion matrix and saves it to a file."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado")
        sys.exit(1)
    model = load_model(model_path)
    dataloader_test = load_test_dataloader(model_path)
    fig = plot_confusion_matrix_binary(model, dataloader_test, threshold=threshold, title=title)
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
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth',
    #     output_path='results/confusion/full_data_effusion_effusion_densenet_binary_15epochs.pdf',
    #     threshold=0.5
    # )

    # RESULTADOS PREGUNTA 3
    # # example
    # plot_confusion_and_save(
    #     model_path='results/models/sample_data_binary_binary_densenet_binary.pth', 
    #     output_path='results/confusion/example.png'
    # )
    # # pneumonia
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_binary_binary_low_res_densenet_binary_1epoch.png',
    #     title='Pneumonia'
    # )
    # # effusion
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_effusion_effusion_low_res_densenet_binary_1epoch.png',
    #     title='Effusion'
    # )
    # # atelectasis
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_atelectasis_atelectasis_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_atelectasis_atelectasis_low_res_densenet_binary_1epoch.png',
    #     title='Atelectasis'
    # )
    # # consolidation
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_consolidation_consolidation_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_consolidation_consolidation_low_res_densenet_binary_1epoch.png',
    #     title='Consolidation'
    # )
    # # infiltration
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_infiltration_infiltration_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_infiltration_infiltration_low_res_densenet_binary_1epoch.png',
    #     title='Infiltration'
    # )
    # # pneumothorax
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_pneumothorax_pneumothorax_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_pneumothorax_pneumothorax_low_res_densenet_binary_1epoch.png',
    #     title='Pneumothorax'
    # ) 
    # # edema
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_edema_edema_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_edema_edema_low_res_densenet_binary_1epoch.png',
    #     title='Edema'
    # )
    # # emphysema
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_emphysema_emphysema_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_emphysema_emphysema_low_res_densenet_binary_1epoch.png',
    #     title='Emphysema'
    # )
    # # fibrosis
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_fibrosis_fibrosis_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_fibrosis_fibrosis_low_res_densenet_binary_1epoch.png',
    #     title='Fibrosis'
    # )
    # # pleural_thickening
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_pleural_thickening_pleural_thickening_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_pleural_thickening_pleural_thickening_low_res_densenet_binary_1epoch.png',
    #     title='Pleural Thickening'
    # )
    # # cardiomegaly
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_cardiomegaly_cardiomegaly_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_cardiomegaly_cardiomegaly_low_res_densenet_binary_1epoch.png',
    #     title='Cardiomegaly'
    # )
    # # nodule
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_nodule_nodule_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_nodule_nodule_low_res_densenet_binary_1epoch.png',
    #     title='Nodule'
    # )
    # # mass
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_mass_mass_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_mass_mass_low_res_densenet_binary_1epoch.png',
    #     title='Mass'
    # )
    # # hernia
    # plot_confusion_and_save(
    #     model_path='results/models/full_data_hernia_hernia_low_res_densenet_binary_1epoch.pth', 
    #     output_path='results/confusion/full_data_hernia_hernia_low_res_densenet_binary_1epoch.png',
    #     title='Hernia'
    # )





