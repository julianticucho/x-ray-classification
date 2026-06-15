import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import load_model, load_test_dataloader
from src.diagnostics import plot_cam_overlay

def generate_batch_cams(
    model_path, 
    num_images=16, 
    mode='random', 
    output_dir='results/cam', 
    output_prefix='cam', 
    seed=42
    ):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado")
        sys.exit(1)
    model = load_model(model_path)
    dataloader_test = load_test_dataloader(model_path)
    dataset = dataloader_test.dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    
    selected_indices = []
    
    if mode == 'random':
        print(f"Seleccionando {num_images} imágenes al azar...")
        selected_indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    else:
        print(f"Buscando {num_images} imágenes con condición '{mode.upper()}'...")
        search_indices = list(range(len(dataset)))
        random.shuffle(search_indices)
        
        for idx in search_indices:
            img, lbl = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            true_label = int(lbl.item())
            
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
                pred_label = 1 if prob >= 0.5 else 0
                
            if mode == 'tp' and true_label == 1 and pred_label == 1:
                selected_indices.append(idx)
            elif mode == 'fp' and true_label == 0 and pred_label == 1:
                selected_indices.append(idx)
            elif mode == 'fn' and true_label == 1 and pred_label == 0:
                selected_indices.append(idx)
            elif mode == 'tn' and true_label == 0 and pred_label == 0:
                selected_indices.append(idx)
                
            if len(selected_indices) >= num_images:
                break
                
        print(f"Se encontraron {len(selected_indices)} imágenes de tipo '{mode.upper()}'.")
        if len(selected_indices) == 0:
            print("No se encontraron imágenes que cumplan la condición. Terminando.")
            return

    print(f"Generando CAMs...")
    
    for idx in selected_indices:
        image_tensor, lbl = dataset[idx]
        image_tensor_batch = image_tensor.unsqueeze(0).to(device)
        true_label = int(lbl.item())
        img_name = dataset.labels_df.iloc[idx]['Image Index']
        with torch.no_grad():
            output = model(image_tensor_batch)
            prob = torch.sigmoid(output).item()
            
        pred_label = 1 if prob >= 0.5 else 0
        
        if true_label == 1 and pred_label == 1:
            category = "True Positive"
        elif true_label == 0 and pred_label == 0:
            category = "True Negative"
        elif true_label == 0 and pred_label == 1:
            category = "False Positive"
        else:
            category = "False Negative"
            
        print(f"Procesando {img_name}: {category} (Prob: {prob:.4f})")
        fig = plot_cam_overlay(model, image_tensor_batch)
        fig.suptitle(f"CAM: {category} ({prob:.2f} prob)\nImage: {img_name}", fontsize=14)
        plt.tight_layout()
        base_name = os.path.splitext(img_name)[0]
        mode_prefix = f"{mode}_" if mode != 'random' else ""
        output_path = os.path.join(output_dir, f"{output_prefix}_{mode_prefix}{base_name}.png")
        fig.savefig(output_path)
        plt.close(fig) 
        
    print(f"Todas las imágenes han sido guardadas en {output_dir}/")
    return


if __name__ == '__main__':

    # # full_data_binary_binary_low_res_densenet_binary_1epoch
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_1epoch',
    #     seed=1
    # )

    # # full_data_binary_binary_low_res_densenet_binary_15epochs
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_low_res_densenet_binary_15epochs',
    #     seed=1
    # )

    # # full_data_binary_binary_densenet_binary_1epoch
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_1epoch',
    #     seed=1
    # )

    # # full_data_binary_binary_densenet_binary_15epochs
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_binary_binary_densenet_binary_15epochs',
    #     seed=1
    # )

    # full_data_effusion_effusion_low_res_densenet_binary_1epoch
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_low_res_densenet_binary_1epoch',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_low_res_densenet_binary_1epoch',
    #     seed=1
    # )

    # full_data_effusion_effusion_densenet_binary_15epochs
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='fn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_densenet_binary_15epochs',
    #     seed=1
    # )
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth', 
    #     num_images=10, 
    #     mode='tn', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_densenet_binary_15epochs',
    #     seed=1
    # )

    # PREGUNTA 3
    # generate_batch_cams(
    #     model_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth', 
    #     num_images=30, 
    #     mode='tp', 
    #     output_dir='results/cam', 
    #     output_prefix='full_data_effusion_effusion_low_res_densenet_binary_1epoch',
    #     seed=1
    # )

    # PREGUNTA 2
    generate_batch_cams(
        model_path='results/models/full_data_gender_gender_low_res_densenet_binary_1epoch.pth', 
        num_images=10, 
        mode='tp', 
        output_dir='results/cam', 
        output_prefix='full_data_gender_gender_low_res_densenet_binary_1epoch',
        seed=1
    )
    generate_batch_cams(
        model_path='results/models/full_data_gender_gender_low_res_densenet_binary_1epoch.pth', 
        num_images=10, 
        mode='tn', 
        output_dir='results/cam', 
        output_prefix='full_data_gender_gender_low_res_densenet_binary_1epoch',
        seed=1
    )
