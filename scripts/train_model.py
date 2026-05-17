import pandas as pd
import sys
import os
from src import PreprocessingConfigFactory, DataLoaderFactory, ModelFactory
from src.storage import save_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_model(
    labels_dir, # ruta al archivo CSV con las etiquetas
    img_dir, # ruta a la carpeta con las imágenes
    preprocessing_name, # nombre del preprocesamiento a usar (ej: 'example')
    dataloader_name, # nombre del dataloader a usar (ej: 'basic')
    model_name, # nombre del modelo a usar (ej: 'mlp')
    output_path, # ruta donde guardar el modelo entrenado
    num_epochs=10, # número de épocas para entrenar
    lr=0.001, # learning rate
    num_workers=1, # número de workers para el dataloader
):
    """Entrena un modelo con las configuraciones especificadas y lo guarda."""
    labels_df = pd.read_csv(labels_dir)
    pp_factory = PreprocessingConfigFactory(labels_df)

    train_df, val_df, test_df = pp_factory.get(preprocessing_name)
    print(f"Training with {len(train_df)} samples, Validating with {len(val_df)} samples")

    dl_factory_train = DataLoaderFactory(train_df, img_dir, num_workers=num_workers, is_train=True)
    train_dataloader = dl_factory_train.get(dataloader_name)

    dl_factory_val = DataLoaderFactory(val_df, img_dir, num_workers=num_workers, is_train=False)
    val_dataloader = dl_factory_val.get(dataloader_name)

    model_factory = ModelFactory()
    model = model_factory.get(model_name)
    model.train_model(train_dataloader, val_dataloader=val_dataloader, num_epochs=num_epochs, lr=lr)
    save_model(output_path, model, preprocessing_name, dataloader_name, model_name, labels_dir, img_dir)


if __name__ == '__main__':
    train_model(
        labels_dir='datasets/nih-chest-xrays/sample/versions/4/sample_labels.csv',
        img_dir=['datasets/nih-chest-xrays/sample/versions/4/sample/images'],
        preprocessing_name='binary',
        dataloader_name='binary',
        model_name='densenet_binary',
        output_path='results/models/sample_data_binary_binary_densenet_binary.pth',
        num_epochs=30,
        lr=0.001,
        num_workers=11,
    )

    # full data example
    # train_model(
    #     labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
    #     img_dir=[
    #         'datasets/nih-chest-xrays/data/versions/3/images_001/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_002/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_003/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_004/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_005/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_006/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_007/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_008/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_009/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_010/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_011/images',
    #         'datasets/nih-chest-xrays/data/versions/3/images_012/images',
    #     ],
    #     preprocessing_name='binary',
    #     dataloader_name='binary',
    #     model_name='densenet_binary',
    #     output_path='results/models/binary_binary_densenet_binary.pth',
    #     num_epochs=2,
    #     lr=0.001,
    #     num_workers=11,
    # )

    # 'datasets/nih-chest-xrays/sample/versions/4/sample_labels.csv'
    # ['datasets/nih-chest-xrays/sample/versions/4/sample/images']
    #
    # 'datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv'
    # [
    # 'datasets/nih-chest-xrays/data/versions/3/images_001/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_002/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_003/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_004/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_005/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_006/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_007/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_008/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_009/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_010/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_011/images',
    # 'datasets/nih-chest-xrays/data/versions/3/images_012/images',
    # ]
