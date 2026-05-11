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
        }
    
    def get(self, config_name):
        """Retorna el DataFrame de train y test con la configuración especificada."""
        configs = self.get_available_configurations()
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        return configs[config_name]()
    
    def config_example(self):
        """Configuración de ejemplo: convierte edad y hace split aleatorio."""
        self.labels_df['Patient Age'] = self.labels_df['Patient Age'].apply(self._convert_age)
        train_df, test_df = self._random_split(test_ratio=0.2, seed=0)
        return train_df, test_df
    
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
    
    def _random_split(self, test_ratio=0.2, seed=0):
        """Divide el DataFrame en train y test de forma aleatoria."""
        shuffled_df = self.labels_df.sample(frac=1, random_state=seed)
        split_idx = int(len(shuffled_df) * (1 - test_ratio))

        train_df = shuffled_df.iloc[:split_idx]
        test_df = shuffled_df.iloc[split_idx:]

        return train_df, test_df
