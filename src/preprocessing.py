import numpy as np

class PreprocessingConfigFactory:
    """Fábrica para configuraciones de preprocesamiento de datos."""
    
    def __init__(self, labels_df):
        """Inicializa la fábrica con el DataFrame de etiquetas."""
        self.labels_df = labels_df.copy()
    
    def get_available_configurations(self):
        """Retorna las configuraciones de preprocesamiento disponibles."""
        return {
            'example': self.config_example,
            'binary': self.config_binary,
            'effusion': self.config_effusion,
            'atelectasis': self.config_atelectasis,
            'consolidation': self.config_consolidation,
            'infiltration': self.config_infiltration,
            'pneumothorax': self.config_pneumothorax,
            'edema': self.config_edema,
            'emphysema': self.config_emphysema,
            'fibrosis': self.config_fibrosis,
            'pleural_thickening': self.config_pleural_thickening,
            'cardiomegaly': self.config_cardiomegaly,
            'nodule': self.config_nodule,
            'mass': self.config_mass,
            'hernia': self.config_hernia,
            'gender': self.config_gender
        }
    
    def get(self, config_name):
        """Retorna los DataFrames de train, val y test con la configuración especificada."""
        configs = self.get_available_configurations()
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        return configs[config_name]()
    
    def config_example(self):
        """Configuración de ejemplo: convierte edad y hace split aleatorio."""
        self.labels_df['Patient Age'] = self.labels_df['Patient Age'].apply(self._convert_age)
        train_df, val_df, test_df = self._random_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_binary(self):
        """Configuración binaria para detección de Neumonía."""
        self.labels_df['Pneumonia_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Pneumonia' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_effusion(self):
        """Configuración para detección de Effusion."""
        self.labels_df['Effusion_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Effusion' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_atelectasis(self):
        """Configuración para detección de Atelectasis."""
        self.labels_df['Atelectasis_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Atelectasis' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_consolidation(self):
        """Configuración para detección de Consolidación."""
        self.labels_df['Consolidation_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Consolidation' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_infiltration(self):
        """Configuración para detección de Infiltración."""
        self.labels_df['Infiltration_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Infiltration' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_pneumothorax(self):
        """Configuración para detección de Pneumothorax."""
        self.labels_df['Pneumothorax_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Pneumothorax' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_edema(self):
        """Configuración para detección de Edema."""
        self.labels_df['Edema_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Edema' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_emphysema(self):
        """Configuración para detección de Emfisema."""
        self.labels_df['Emphysema_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Emphysema' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_fibrosis(self):
        """Configuración para detección de Fibrosis."""
        self.labels_df['Fibrosis_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Fibrosis' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_pleural_thickening(self):
        """Configuración para detección de Pleural Thickening."""
        self.labels_df['Pleural_Thickening_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Pleural_Thickening' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_cardiomegaly(self):
        """Configuración para detección de Cardiomegaly."""
        self.labels_df['Cardiomegaly_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Cardiomegaly' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df
    
    def config_nodule(self):
        """Configuración para detección de Nodule."""
        self.labels_df['Nodule_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Nodule' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df

    def config_mass(self):
        """Configuración para detección de Mass."""
        self.labels_df['Mass_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Mass' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df

    def config_hernia(self):
        """Configuración para detección de Hernia."""
        self.labels_df['Hernia_Label'] = self.labels_df['Finding Labels'].apply(
            lambda x: 1 if 'Hernia' in x else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df

    def config_gender(self):
        """Configuración para clasificación binaria de género (F=1, M=0)."""
        self.labels_df['Gender_Label'] = self.labels_df['Patient Gender'].apply(
            lambda x: 1 if x == 'F' else 0
        )
        train_df, val_df, test_df = self._patient_split(val_ratio=0.1, test_ratio=0.2, seed=0)
        return train_df, val_df, test_df

    def _convert_age(self, age_str):
        """Convierte string de edad (Y/M/D) a años."""
        if isinstance(age_str, (int, float)):
            return float(age_str)
        if age_str.endswith('Y'):
            return int(age_str.replace('Y', ''))
        elif age_str.endswith('M'):
            return int(age_str.replace('M', '')) / 12
        elif age_str.endswith('D'):
            return int(age_str.replace('D', '')) / 365
        return int(age_str)
    
    def _random_split(self, val_ratio=0.1, test_ratio=0.2, seed=0):
        """Divide el DataFrame en train, val y test de forma aleatoria."""
        shuffled_df = self.labels_df.sample(frac=1, random_state=seed)
        val_idx = int(len(shuffled_df) * (1 - test_ratio - val_ratio))
        test_idx = int(len(shuffled_df) * (1 - test_ratio))

        train_df = shuffled_df.iloc[:val_idx]
        val_df = shuffled_df.iloc[val_idx:test_idx]
        test_df = shuffled_df.iloc[test_idx:]

        return train_df, val_df, test_df

    def _patient_split(self, val_ratio=0.1, test_ratio=0.2, seed=0):
        """Divide el DataFrame a nivel de paciente."""
        np.random.seed(seed)
        unique_patients = self.labels_df['Patient ID'].unique()
        np.random.shuffle(unique_patients)
        
        num_patients = len(unique_patients)
        val_idx = int(num_patients * (1 - test_ratio - val_ratio))
        test_idx = int(num_patients * (1 - test_ratio))
        
        train_patients = unique_patients[:val_idx]
        val_patients = unique_patients[val_idx:test_idx]
        test_patients = unique_patients[test_idx:]
        
        train_df = self.labels_df[self.labels_df['Patient ID'].isin(train_patients)]
        val_df = self.labels_df[self.labels_df['Patient ID'].isin(val_patients)]
        test_df = self.labels_df[self.labels_df['Patient ID'].isin(test_patients)]
        
        return train_df, val_df, test_df
