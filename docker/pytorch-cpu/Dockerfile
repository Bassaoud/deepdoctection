FROM python:3.9-slim
LABEL repository="deepdoctection-api"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y \
    git \
    poppler-utils \
    python3-pip \
    python3-opencv \
    tesseract-ocr \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Installation de la version spécifique de huggingface_hub
RUN pip install --no-cache-dir huggingface-hub==0.16.4

# Installation de FastAPI et dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de PyTorch CPU
RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu

# Installation de Detectron2
RUN python3 -m pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Installation de deepdoctection
RUN python3 -m pip install --no-cache-dir "deepdoctection[pt]"

# Correction : remplacer 'cached_download' par 'hf_hub_download' dans model.py
RUN sed -i 's/cached_download/hf_hub_download/g' /usr/local/lib/python3.9/site-packages/deepdoctection/extern/model.py

# Vérifier que le fichier a bien été modifié
RUN grep -n "hf_hub_download" /usr/local/lib/python3.9/site-packages/deepdoctection/extern/model.py

# Création du dossier de configuration
RUN mkdir -p /app/configs

# Création du fichier de configuration personnalisé
COPY configs/custom_config.yaml /app/configs/

# Création des dossiers nécessaires
RUN mkdir -p /app/uploads /app/training

# Copie du code de l'API
COPY ./api .

# Expose le port
EXPOSE 8000

# Commande pour démarrer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]