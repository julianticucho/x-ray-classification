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
