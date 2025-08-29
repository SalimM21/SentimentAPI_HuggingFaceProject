# 🚀 Sentiment Analysis API (FastAPI + Hugging Face)

Cette application fournit une **API REST** et une **page web de test** pour analyser le sentiment (positif/négatif) d’un texte en utilisant un modèle Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`).

---

## 📂 Structure du projet

```
├── app/
│   └── main.py           # Application FastAPI
│── static/
│   └── index.html        # Page test frontend
├── requirements.txt      # Dépendances Python
├── Dockerfile            # Image Docker
└── README.md             # Documentation
```

---

## ⚙️ Installation locale

### 1. Créer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Lancer l’API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 Déploiement avec Docker

### 1. Construire l’image
```bash
docker build -t sentiment-api .
```

### 2. Lancer le conteneur
```bash
docker run -p 8000:8000 sentiment-api
```

L’API sera accessible sur :  
👉 [http://localhost:8000](http://localhost:8000)  

---

## 🌐 Déploiement sur Hugging Face Spaces

### 1. Créer un nouveau Space
- Aller sur [Hugging Face Spaces](https://huggingface.co/spaces)
- Créer un espace **type Docker**
- Uploader les fichiers du projet (`app/`, `requirements.txt`, `Dockerfile`, `README.md`)

### 2. Configurer le cache Hugging Face
Le code utilise un dossier **accessible en écriture** pour le cache :
```python
CACHE_DIR = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
```

Cela évite l’erreur `PermissionError: /.cache`.

### 3. Build & Run
Une fois le Space lancé, Hugging Face va automatiquement :
- Construire l’image Docker
- Démarrer l’API FastAPI
- Fournir un endpoint public

---

## 📡 Endpoints disponibles

### 🔹 Health Check
```
GET /health
```
**Réponse :**
```json
{ "status": "ok", "model": "distilbert-base-uncased-finetuned-sst-2-english" }
```

### 🔹 Prédiction de sentiment
```
POST /predict
```
**Entrée :**
```json
{ "text": "I love FastAPI and Hugging Face!" }
```

**Sortie :**
```json
{
  "label": "positive",
  "confidence": 0.9998,
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

---

## 🖼️ Page de test (Frontend)

L’application expose aussi une **page HTML minimaliste** :  
👉 [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

---

##  TODO

- [ ] Ajouter un modèle multilingue (par ex. `nlptown/bert-base-multilingual-uncased-sentiment`)  
- [ ] Ajouter une UI Gradio pour la démo  
- [ ] Intégrer des tests unitaires  

---

## Licence

Ce projet est sous licence MIT.  
Le modèle appartient à Hugging Face et ses contributeurs.
