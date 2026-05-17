import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage import load_model, load_test_dataloader, load_train_dataloader
from src.diagnostics import plot_confusion_matrix_binary

if __name__ == '__main__':
    model_path = 'results/models/sample_data_binary_binary_densenet_binary.pth'
    output_path = 'results/confusion/confusion_example.pdf'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    dataloader_test = load_test_dataloader(model_path)
    
    print("Generating Confusion Matrix...")
    fig = plot_confusion_matrix_binary(model, dataloader_test, threshold=0.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"Confusion Matrix saved to {output_path}")
