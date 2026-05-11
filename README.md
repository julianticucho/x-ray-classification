# Setup

## Instalación del Entorno Virtual (venv)

Para crear y activar el entorno virtual:

```bash
# crear el entorno virtual
python3 -m venv venv

# activar el entorno virtual
# en linux o mac
source venv/bin/activate
# en windows
venv\Scripts\activate

# instalar las dependencias
pip install -r requirements.txt
```

## Estructura del Repositorio

```
x-ray-classification/
├── datasets/           # almacenar los datasets (se crea al descargar datos)
├── notebooks/          # notebooks para análisis 
├── results/            # modelos entrenados, gráficos, etc...
├── scripts/            # scripts principales 
├── src/                # código fuente 
├── requirements.txt    # dependencias
└── venv/               # entorno virtual 
```

## Configuración de Kaggle

Para descargar los datasets necesitas una cuenta de Kaggle y configurar tus credenciales:

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com)
2. Ve a tu perfil → Settings → API Tokens
3. Haz clic en "Create Legacy API Key" para descargar `kaggle.json`
4. Coloca el archivo `kaggle.json` en:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<tu-usuario>\.kaggle\kaggle.json`

## Descarga de Datos

Para descargar los datos de ejemplo:

```bash
python -m scripts.download_sample_data
```

Para descargar los datos completos:

```bash
python -m scripts.download_full_data
```

## Factories 

- **DataLoaderFactory**: Crea DataLoaders con diferentes configuraciones de transformación y batch size.
- **ModelFactory**: Crea redes neuronales con diferentes arquitecturas.
- **PreprocessingConfigFactory**: Aplica diferentes configuraciones de preprocesamiento al DataFrame de etiquetas.

### Cómo agregar nuevas configuraciones

#### Agregar un nuevo preprocesamiento (`src/preprocessing.py`)

1. Crea un nuevo método en `PreprocessingConfigFactory`:
```python
def config_mi_preprocesamiento(self):
    """Descripción de tu preprocesamiento."""
    # Modifica self.labels_df según necesites
    train_df, test_df = self._random_split(test_ratio=0.2, seed=42)
    return train_df, test_df
```

2. Agrégalo a `get_available_configurations()`:
```python
def get_available_configurations(self):
    return {
        'example': self.config_example,
        'mi_preprocesamiento': self.config_mi_preprocesamiento,  # nuevo
    }
```

3. Úsalo en tu script:
```python
preprocessing_name='mi_preprocesamiento'
```

#### Agregar un nuevo modelo (`src/models.py`)

1. Crea tu clase de modelo heredando de `BaseModel`:
```python
class MiModelo(BaseModel):
    def __init__(self, num_classes=14):
        super(MiModelo, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        # Define tus capas aquí
        
    def forward(self, x):
        # Define el forward pass
        return x
```

2. Agrégalo al `ModelFactory`:
```python
def get_available_configurations(self):
    return {
        'mlp': self.create_mlp,
        'mi_modelo': self.create_mi_modelo,  # nuevo
    }

def create_mi_modelo(self):
    return MiModelo(num_classes=14)
```

3. Úsalo:
```python
model_name='mi_modelo'
```

#### Agregar un nuevo DataLoader (`src/dataloaders.py`)

1. Crea un método con tu transformación:
```python
def create_con_augmentation(self):
    """DataLoader con aumentación de datos."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = XRayDataset(self.labels_df, self.img_dir, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=self.num_workers)  # batch_size diferente
```

2. Agrégalo a `get_available_configurations()`:
```python
def get_available_configurations(self):
    return {
        'basic': self.create_basic,
        'con_augmentation': self.create_con_augmentation,  # nuevo
    }
```

3. Úsalo:
```python
dataloader_name='con_augmentation'
```

