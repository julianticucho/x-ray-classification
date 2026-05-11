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
chestx-classification/
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

