import streamlit as st
import requests
import json
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.markdown("Prédisez le prix d'une maison en utilisant différents modèles ML")

# Configuration de l'API
API_URL = "http://localhost:8000"  # URL Docker
# Pour test local: API_URL = "http://localhost:8000"

@st.cache_data
def get_feature_names():
    """Récupère les noms des features depuis l'API"""
    try:
        response = requests.get(f"{API_URL}/features")
        if response.status_code == 200:
            return response.json()["features"]
        else:
            return []
    except:
        # Fallback: charger depuis le fichier local si disponible
        try:
            return joblib.load('models/feature_names.pkl')
        except:
            return [f"feature_{i}" for i in range(10)]  # Fallback générique

@st.cache_data
def get_available_models():
    """Récupère les modèles disponibles depuis l'API"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        else:
            return ["random_forest", "xgboost", "linear_regression"]
    except:
        return ["random_forest", "xgboost", "linear_regression"]

# Sidebar pour la sélection du modèle
st.sidebar.header("Configuration")
models = get_available_models()
selected_model = st.sidebar.selectbox("Choisir un modèle", models)

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Caractéristiques de la maison")
    
    # Récupérer les features
    feature_names = get_feature_names()
    
    # Bouton pour générer des données de test
    if st.button("🎲 Générer des données de test", type="secondary"):
        # Générer des valeurs réalistes pour une maison
        import random
        test_values = []
        for i, feature in enumerate(feature_names):
            feature_lower = feature.lower()
            if 'bedroom' in feature_lower or 'bed' in feature_lower:
                val = random.randint(2, 5)
            elif 'bathroom' in feature_lower or 'bath' in feature_lower:
                val = random.randint(1, 4)
            elif 'floor' in feature_lower:
                val = random.choice([1, 1.5, 2, 2.5, 3])
            elif 'sqft' in feature_lower or 'area' in feature_lower or 'size' in feature_lower:
                if 'lot' in feature_lower:
                    val = random.randint(2000, 15000)
                else:
                    val = random.randint(800, 4000)
            elif 'year' in feature_lower or 'built' in feature_lower:
                val = random.randint(1950, 2023)
            elif 'age' in feature_lower:
                val = random.randint(1, 70)
            elif 'grade' in feature_lower or 'condition' in feature_lower:
                val = random.randint(3, 12)
            elif 'view' in feature_lower:
                val = random.randint(0, 4)
            elif 'waterfront' in feature_lower:
                val = random.choice([0, 1])
            else:
                val = round(random.uniform(0.5, 10), 2)
            
            # Stocker dans session state
            st.session_state[f"feature_{i}"] = val
        
        st.success("🎉 Données de test générées ! Faites défiler pour voir les valeurs.")
    
    # Créer les inputs dynamiquement
    features_values = []
    
    # Organiser en colonnes pour un meilleur affichage
    cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 2
        with cols[col_idx]:
            # Utiliser la valeur du session_state si elle existe, sinon 0.0
            default_value = st.session_state.get(f"feature_{i}", 0.0)
            value = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=float(default_value),
                key=f"feature_{i}"
            )
            features_values.append(value)

with col2:
    st.header("Prédiction")
    
    if st.button("Prédire le prix", type="primary"):
        # Préparer la requête
        prediction_data = {
            "features": features_values,
            "model_name": selected_model
        }
        
        try:
            # Faire la prédiction
            with st.spinner("Prédiction en cours..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=prediction_data
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Afficher le résultat
                st.success("Prédiction réussie!")
                st.metric(
                    label="Prix prédit",
                    value=f"${result['prediction']:,.2f}"
                )
                
                st.info(f"Modèle utilisé: {result['model_used']}")
                st.caption(f"Timestamp: {result['timestamp']}")
                
            else:
                error_detail = response.json().get("detail", "Erreur inconnue")
                st.error(f"Erreur: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter à l'API. Vérifiez que le service est démarré.")
        except Exception as e:
            st.error(f"Erreur inattendue: {str(e)}")

# Section d'information
st.markdown("---")

# Exemples de données
with st.expander("📋 Exemples de valeurs typiques"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏠 Maison Standard")
        st.write("• Chambres: 3")
        st.write("• Salles de bain: 2")
        st.write("• Surface: 1500-2000 sqft")
        st.write("• Terrain: 5000-8000 sqft")
        st.write("• Étages: 1-2")
    
    with col2:
        st.subheader("🏘️ Maison Familiale")
        st.write("• Chambres: 4-5")
        st.write("• Salles de bain: 2-3")
        st.write("• Surface: 2000-3000 sqft")
        st.write("• Terrain: 8000-12000 sqft")
        st.write("• Étages: 2-2.5")
    
    with col3:
        st.subheader("🏛️ Maison de Luxe")
        st.write("• Chambres: 5+")
        st.write("• Salles de bain: 3+")
        st.write("• Surface: 3000+ sqft")
        st.write("• Terrain: 10000+ sqft")
        st.write("• Grade/Condition: 8+")

st.header("Information sur les modèles")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌲 Random Forest")
    st.write("Ensemble de décision trees, robuste aux outliers")

with col2:
    st.subheader("🚀 XGBoost")
    st.write("Gradient boosting optimisé, haute performance")

with col3:
    st.subheader("📈 Linear Regression")
    st.write("Modèle linéaire simple et interprétable")

# Health check de l'API
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API connectée")
    else:
        st.sidebar.error("❌ API non disponible")
except:
    st.sidebar.error("❌ API non disponible")