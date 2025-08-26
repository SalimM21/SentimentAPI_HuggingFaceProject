FROM python:3.10-slim

# Installer dépendances système nécessaires pour torch et transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Installer les dépendances Python, incluant torch CPU
RUN pip install --no-cache-dir torch==2.2.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .
COPY static ./static

# Exposer le port 7860 (obligatoire pour HF Spaces)
EXPOSE 7860

# Lancer FastAPI via Uvicorn (forme JSON recommandée)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
