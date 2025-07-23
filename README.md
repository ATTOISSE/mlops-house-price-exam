# 🏠 House Price Prediction MLOps

**Projet MLOps de prédiction de prix immobilier avec 3 modèles (Linear Regression, Random Forest, XGBoost), API FastAPI, frontend Streamlit et CI/CD.**

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Modèles](#modèles)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API Endpoints](#api-endpoints)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Structure du projet](#structure-du-projet)

## 🎯 Aperçu

Ce projet implémente une pipeline MLOps complète pour prédire le prix des maisons en utilisant le dataset [House Data de Kaggle](https://www.kaggle.com/datasets/shree1992/housedata). Il compare 3 algorithmes d'apprentissage automatique et fournit une API REST avec interface web.

## 🤖 Modèles

| Modèle | Type | Description |
|--------|------|-------------|
| **Linear Regression** | Régression linéaire | Modèle simple et rapide, bon pour la baseline |
| **Random Forest** | Ensemble | Robuste aux outliers, bon équilibre performance/interprétabilité |
| **XGBoost** | Gradient Boosting | Haute performance, optimisé pour la compétition |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API FastAPI   │    │   ML Models     │
│   Streamlit     │◄──►│   + Logging     │◄──►│   .pkl files    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│     Docker      │◄─────────────┘
                        │   + CI/CD       │
                        └─────────────────┘
```

## 🛠️ Installation

```bash
# Cloner le repository
git clone <repo-url>
cd house-price-mlops

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset depuis Kaggle
# Placer le fichier output.csv dans le dossier racine
```

## 🚀 Utilisation

### 1. Entraîner les modèles

```bash
python train_models.py
```

Génère 4 fichiers :
- `linear_reg_model.pkl`
- `random_forest_model.pkl` 
- `xgboost_model.pkl`
- `best_model.pkl`

### 2. Lancer l'API

```bash
python app.py
```

L'API sera disponible sur `http://localhost:8000`

### 3. Interface utilisateur

```bash
streamlit run frontend.py
```

Interface disponible sur `http://localhost:8501`

## 📡 API Endpoints

### `GET /`
Statut de l'API

### `GET /models`
Liste des modèles disponibles

```json
{
  "available_models": ["linear_regression", "random_forest", "xgboost"]
}
```

### `POST /predict`
Prédiction de prix

**Request body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1500,
  "sqft_lot": 5000,
  "floors": 1.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 1200,
  "sqft_basement": 300,
  "yr_built": 1990,
  "yr_renovated": 0,
  "model_name": "random_forest"
}
```

**Response:**
```json
{
  "predicted_price": 425000.50,
  "model_used": "random_forest"
}
```

## 🐳 Docker

### Build de l'image
```bash
docker build -t house-price-api .
```

### Lancer le container
```bash
docker run -p 8000:8000 house-price-api
```

## 🔄 CI/CD

Pipeline GitHub Actions automatique :

- ✅ **Tests** : Vérification de l'importation des modules
- 🐳 **Build** : Construction de l'image Docker
- 🧪 **Test container** : Test de l'API dans le container

Déclenché sur les push vers `main` et les pull requests.

## 📁 Structure du projet

```
house-price-mlops/
├── train_models.py          # Entraînement des modèles ML
├── app.py                   # API FastAPI
├── frontend.py              # Interface Streamlit
├── Dockerfile               # Configuration Docker
├── requirements.txt         # Dépendances Python
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # Pipeline CI/CD
├── *.pkl                   # Modèles entraînés
└── README.md               # Documentation
```

## 📊 Logging

L'API génère des logs structurés en JSON :

```json
{
  "timestamp": "2024-01-15T10:30:45",
  "model_used": "random_forest",
  "features": {...},
  "prediction": 425000.50,
  "duration": "0.045s"
}
```

## 🎯 Features du dataset

- **bedrooms** : Nombre de chambres
- **bathrooms** : Nombre de salles de bain
- **sqft_living** : Surface habitable (pieds carrés)
- **sqft_lot** : Surface du terrain
- **floors** : Nombre d'étages
- **waterfront** : Vue sur l'eau (0/1)
- **view** : Qualité de la vue (0-4)
- **condition** : État du bien (1-5)
- **sqft_above** : Surface au-dessus du sol
- **sqft_basement** : Surface du sous-sol
- **yr_built** : Année de construction
- **yr_renovated** : Année de rénovation

## 📈 Exemple d'utilisation

```bash
# Test avec curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1500,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1200,
    "sqft_basement": 300,
    "yr_built": 1990,
    "yr_renovated": 0,
    "model_name": "xgboost"
  }'
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT.
