import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm import tqdm

def plot_roc_curve(model, dataloader, device=None):
    """Genera la curva ROC y devuelve la figura."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Calculating Probabilities"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def plot_confusion_matrix_binary(model, dataloader, threshold=0.5, device=None):
    """Genera la matriz de confusion para un modelo binario y devuelve la figura."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Calculating Predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return fig

def plot_cam_overlay(model, image_tensor, original_image=None, device=None):
    """Genera el CAM para una imagen y lo superpone usando Matplotlib."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        cam = model.generate_cam(image_tensor)
    cam = cam.cpu().numpy()[0] # [H, W]
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    if original_image is not None:
        h, w = np.array(original_image).shape[:2]
    else:
        h, w = image_tensor.shape[2], image_tensor.shape[3]
    cam_resized = F.interpolate(cam_tensor, size=(h, w), mode='bilinear', align_corners=False)
    cam_resized = cam_resized.squeeze().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if original_image is not None:
        img_show = original_image
        axes[0].imshow(img_show, cmap='gray')
        axes[1].imshow(img_show, cmap='gray')
    else:
        img_show = image_tensor.cpu().squeeze().numpy()
        img_show = np.transpose(img_show, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_show = std * img_show + mean
        img_show = np.clip(img_show, 0, 1)
        img_gray = np.dot(img_show[...,:3], [0.2989, 0.5870, 0.1140])
        axes[0].imshow(img_gray, cmap='gray')
        axes[1].imshow(img_gray, cmap='gray')
        
    im = axes[1].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[0].axis('off')
    axes[0].set_title("Input")
    axes[1].axis('off')
    axes[1].set_title("CAM")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
