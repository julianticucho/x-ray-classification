import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import sys
import os
from src import PreprocessingConfigFactory, DataLoaderFactory, ModelFactory
from src.storage import save_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_model(
    labels_dir, 
    img_dir, 
    preprocessing_name, 
    dataloader_name, 
    model_name, 
    output_path,
    num_epochs=10, 
    lr=0.001, 
    num_workers=1, 
):
    """Entrena un modelo con las configuraciones especificadas y lo guarda."""
    labels_df = pd.read_csv(labels_dir)
    pp_factory = PreprocessingConfigFactory(labels_df)

    train_df, val_df, test_df = pp_factory.get(preprocessing_name)
    print(f"Entrenando con {len(train_df)} muestras, validando con {len(val_df)} muestras")

    dl_factory_train = DataLoaderFactory(train_df, img_dir, num_workers=num_workers, is_train=True)
    train_dataloader = dl_factory_train.get(dataloader_name)

    dl_factory_val = DataLoaderFactory(val_df, img_dir, num_workers=num_workers, is_train=False)
    val_dataloader = dl_factory_val.get(dataloader_name)

    model_factory = ModelFactory()
    model = model_factory.get(model_name)
    model.train_model(train_dataloader, val_dataloader=val_dataloader, num_epochs=num_epochs, lr=lr)

    save_model(
        output_path, model, preprocessing_name, 
        dataloader_name, model_name, labels_dir, 
        img_dir, train_losses=model.train_losses, 
        val_losses=model.val_losses
    )


if __name__ == '__main__':
    # train_model(
    #     labels_dir='datasets/nih-chest-xrays/sample/versions/4/sample_labels.csv',
    #     img_dir=['datasets/nih-chest-xrays/sample/versions/4/sample/images'],
    #     preprocessing_name='binary',
    #     dataloader_name='binary_low_res',
    #     model_name='densenet_binary',
    #     output_path='results/models/sample_data_binary_binary_low_res_densenet_binary.pth',
    #     num_epochs=1,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     dataloader_name='binary_low_res',
    #     model_name='densenet_binary',
    #     output_path='results/models/full_data_binary_binary_low_res_densenet_binary_1epoch.pth',
    #     num_epochs=1,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     dataloader_name='binary_low_res',
    #     model_name='densenet_binary',
    #     output_path='results/models/full_data_binary_binary_low_res_densenet_binary_15epochs.pth',
    #     num_epochs=15,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     output_path='results/models/full_data_binary_binary_densenet_binary_1epoch.pth',
    #     num_epochs=1,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     output_path='results/models/full_data_binary_binary_densenet_binary_15epochs.pth',
    #     num_epochs=15,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     preprocessing_name='effusion',
    #     dataloader_name='effusion_low_res',
    #     model_name='densenet_binary',
    #     output_path='results/models/full_data_effusion_effusion_low_res_densenet_binary_1epoch.pth',
    #     num_epochs=1,
    #     lr=0.001,
    #     num_workers=11,
    # )

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
    #     preprocessing_name='effusion',
    #     dataloader_name='effusion',
    #     model_name='densenet_binary',
    #     output_path='results/models/full_data_effusion_effusion_densenet_binary_15epochs.pth',
    #     num_epochs=15,
    #     lr=0.001,
    #     num_workers=11,
    # )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='atelectasis',
        dataloader_name='atelectasis_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_atelectasis_atelectasis_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='consolidation',
        dataloader_name='consolidation_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_consolidation_consolidation_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='infiltration',
        dataloader_name='infiltration_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_infiltration_infiltration_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='pneumothorax',
        dataloader_name='pneumothorax_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_pneumothorax_pneumothorax_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='edema',
        dataloader_name='edema_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_edema_edema_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='emphysema',
        dataloader_name='emphysema_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_emphysema_emphysema_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='fibrosis',
        dataloader_name='fibrosis_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_fibrosis_fibrosis_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='pleural_thickening',
        dataloader_name='pleural_thickening_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_pleural_thickening_pleural_thickening_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='cardiomegaly',
        dataloader_name='cardiomegaly_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_cardiomegaly_cardiomegaly_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='nodule',
        dataloader_name='nodule_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_nodule_nodule_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='mass',
        dataloader_name='mass_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_mass_mass_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )

    train_model(
        labels_dir='datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv',
        img_dir=[
            'datasets/nih-chest-xrays/data/versions/3/images_001/images',
            'datasets/nih-chest-xrays/data/versions/3/images_002/images',
            'datasets/nih-chest-xrays/data/versions/3/images_003/images',
            'datasets/nih-chest-xrays/data/versions/3/images_004/images',
            'datasets/nih-chest-xrays/data/versions/3/images_005/images',
            'datasets/nih-chest-xrays/data/versions/3/images_006/images',
            'datasets/nih-chest-xrays/data/versions/3/images_007/images',
            'datasets/nih-chest-xrays/data/versions/3/images_008/images',
            'datasets/nih-chest-xrays/data/versions/3/images_009/images',
            'datasets/nih-chest-xrays/data/versions/3/images_010/images',
            'datasets/nih-chest-xrays/data/versions/3/images_011/images',
            'datasets/nih-chest-xrays/data/versions/3/images_012/images',
        ],
        preprocessing_name='hernia',
        dataloader_name='hernia_low_res',
        model_name='densenet_binary',
        output_path='results/models/full_data_hernia_hernia_low_res_densenet_binary_15epochs.pth',
        num_epochs=1,
        lr=0.001,
        num_workers=11,
    )




    

