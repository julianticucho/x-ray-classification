import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import load_model, load_test_dataloader
from src.diagnostics import plot_roc_curve, plot_multiple_roc_curves

def plot_roc_and_save(model_path, output_path):
    """Plots the ROC curve and saves it to a file."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado")
        sys.exit(1)
    model = load_model(model_path)
    dataloader_test = load_test_dataloader(model_path)
    fig = plot_roc_curve(model, dataloader_test)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"Curva ROC guardada en: {output_path}")

def plot_multiple_roc_and_save(model_paths_labels, output_path, title='ROC Curves'):
    """Plots multiple ROC curves on one figure and saves it."""
    models_dataloaders = []
    for model_path, label in model_paths_labels:
        if not os.path.exists(model_path):
            print(f"Error: {model_path} no encontrado")
            sys.exit(1)
        model = load_model(model_path)
        dataloader_test = load_test_dataloader(model_path)
        models_dataloaders.append((model, dataloader_test, label))
    fig = plot_multiple_roc_curves(models_dataloaders, title=title, figsize=(10, 8))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"ROC múltiple guardada en: {output_path}")

if __name__ == '__main__':
    # plot_roc_and_save(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth',
    #     output_path='results/roc/full_data_binary_binary_low_res_densenet_binary_1epoch.pdf'
    # )
    # plot_roc_and_save(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth',
    #     output_path='results/roc/full_data_binary_binary_low_res_densenet_binary_15epochs.pdf'
    # )
    # plot_roc_and_save(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth',
    #     output_path='results/roc/full_data_binary_binary_densenet_binary_1epoch.pdf'
    # )
    # plot_roc_and_save(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth',
    #     output_path='results/roc/full_data_binary_binary_densenet_binary_15epochs.pdf'
    # )
    # plot_roc_and_save(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth',
    #     output_path='results/roc/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pdf'
    # )
    # plot_roc_and_save(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth',
    #     output_path='results/roc/full_data_effusion_effusion_densenet_binary_15epochs.pdf'
    # )

    # PREGUNTA 3
    # models_dir = 'results/models'
    # modelos_pregunta3 = [
    #     ('full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 'Pneumonia'),
    #     ('full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 'Effusion'),
    #     ('full_data_atelectasis_atelectasis_low_res_densenet_binary_1epoch.pth', 'Atelectasis'),
    #     ('full_data_consolidation_consolidation_low_res_densenet_binary_1epoch.pth', 'Consolidation'),
    #     ('full_data_infiltration_infiltration_low_res_densenet_binary_1epoch.pth', 'Infiltration'),
    #     ('full_data_pneumothorax_pneumothorax_low_res_densenet_binary_1epoch.pth', 'Pneumothorax'),
    #     ('full_data_edema_edema_low_res_densenet_binary_1epoch.pth', 'Edema'),
    #     ('full_data_emphysema_emphysema_low_res_densenet_binary_1epoch.pth', 'Emphysema'),
    #     ('full_data_fibrosis_fibrosis_low_res_densenet_binary_1epoch.pth', 'Fibrosis'),
    #     ('full_data_pleural_thickening_pleural_thickening_low_res_densenet_binary_1epoch.pth', 'Pleural Thickening'),
    #     ('full_data_cardiomegaly_cardiomegaly_low_res_densenet_binary_1epoch.pth', 'Cardiomegaly'),
    #     ('full_data_nodule_nodule_low_res_densenet_binary_1epoch.pth', 'Nodule'),
    #     ('full_data_mass_mass_low_res_densenet_binary_1epoch.pth', 'Mass'),
    #     ('full_data_hernia_hernia_low_res_densenet_binary_1epoch.pth', 'Hernia'),
    # ]
    # paths_labels = [(os.path.join(models_dir, name), label) for name, label in modelos_pregunta3]
    # plot_multiple_roc_and_save(
    #     paths_labels,
    #     output_path='results/roc/pregunta3_roc_all.png',
    #     title='Curva ROC modelos binarios'
    # )

    # PREGUNTA 2
    models_dir = 'results/models'
    modelos_pregunta2 = [
        ('full_data_gender_gender_low_res_densenet_binary_1epoch.pth', 'Gender'),
    ]
    paths_labels = [(os.path.join(models_dir, name), label) for name, label in modelos_pregunta2]
    plot_multiple_roc_and_save(
        paths_labels,
        output_path='results/roc/pregunta2_roc_all.png',
        title='Curva ROC clasificación por sexo'
    )


    
    