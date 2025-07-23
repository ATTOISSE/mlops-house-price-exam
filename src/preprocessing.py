import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path = '..data/data.csv'
def preprocess_data(input_file='data/data.csv', output_file='data/clean_data.csv'):
    """
    Prétraitement des données de maisons
    """
    try:
        # Charger les données
        logger.info("Chargement des données...")
        df = pd.read_csv(input_file)
        logger.info(f"Données chargées: {df.shape}")
        
        # Nettoyage des valeurs manquantes
        logger.info("Nettoyage des valeurs manquantes...")
        
        # Supprimer les colonnes avec trop de valeurs manquantes (>50%)
        missing_percent = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_percent[missing_percent > 50].index
        df = df.drop(columns=cols_to_drop)
        
        # Remplir les valeurs manquantes numériques par la médiane
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Remplir les valeurs manquantes catégorielles par le mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encodage des variables catégorielles
        logger.info("Encodage des variables catégorielles...")
        le = LabelEncoder()
        for col in categorical_cols:
            if col != 'price':  # Assumant que price est la target
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Supprimer les outliers (IQR method)
        logger.info("Suppression des outliers...")
        for col in numeric_cols:
            if col != 'price':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Sauvegarder les données nettoyées
        df.to_csv(output_file, index=False)
        logger.info(f"Données prétraitées sauvegardées: {output_file}")
        logger.info(f"Shape finale: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {e}")
        raise

if __name__ == "__main__":
    
    preprocess_data()