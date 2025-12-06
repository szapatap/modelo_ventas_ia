import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. PREPARACIÓN DE DATOS
# ==========================================

def load_data(uploaded_file):
    """
    Carga y limpia el dataset desde un archivo Excel o CSV.
    """
    try:
        # Detectar si es un objeto archivo de streamlit o una ruta string
        if isinstance(uploaded_file, str):
             file_name = uploaded_file
        else:
             file_name = uploaded_file.name

        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        return None, str(e)

    # Estandarización de columnas
    df.columns = [c.strip() for c in df.columns]
    
    # Conversión de fechas si existen
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Mes'] = df['Fecha'].dt.month_name()
    
    # Eliminar filas donde el Target o Zona sean nulos
    df = df.dropna(subset=['Zona Geográfica', '¿Adjudicado?', 'Cliente'])
    
    return df, None

# ==========================================
# 2. ENTRENAMIENTO DEL MODELO
# ==========================================

def train_model_logic(df):
    """
    Ejecuta el pipeline de entrenamiento:
    1. Target Encoding para Clientes.
    2. Label Encoding para Zonas.
    3. Random Forest.
    """
    # 1. Feature Engineering: Historial del Cliente (Target Encoding)
    # Calculamos la media de adjudicación por cliente
    client_stats = df.groupby('Cliente')['¿Adjudicado?'].mean().to_dict()
    global_win_rate = df['¿Adjudicado?'].mean()
    
    # Mapeamos esta tasa al dataframe para entrenar
    df['Client_Win_Rate'] = df['Cliente'].map(client_stats)
    
    # 2. Feature Engineering: Zona (Label Encoding)
    le_zona = LabelEncoder()
    df['Zona_Code'] = le_zona.fit_transform(df['Zona Geográfica'])
    
    # Selección de variables (Features y Target)
    X = df[['Zona_Code', 'Client_Win_Rate']]
    y = df['¿Adjudicado?']
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {'accuracy': acc, 'report': report}
    
    # Artefactos necesarios para producción
    artifacts = {
        'client_stats': client_stats,
        'global_win_rate': global_win_rate,
        'le_zona': le_zona,
        'unique_zones': list(le_zona.classes_),
        'unique_clients': list(client_stats.keys())
    }
    
    return model, metrics, artifacts

# ==========================================
# 3. GUARDADO Y CARGA
# ==========================================

def save_model_artifacts(model, artifacts, folder='models'):
    """Guarda el modelo entrenado en disco."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, os.path.join(folder, 'sales_model.pkl'))
    joblib.dump(artifacts, os.path.join(folder, 'artifacts.pkl'))

def load_model_artifacts(folder='models'):
    """Carga el modelo desde disco."""
    try:
        model = joblib.load(os.path.join(folder, 'sales_model.pkl'))
        artifacts = joblib.load(os.path.join(folder, 'artifacts.pkl'))
        return model, artifacts
    except:
        return None, None

# ==========================================
# 4. LÓGICA DE PREDICCIÓN
# ==========================================

def make_prediction(model, artifacts, client_name, zone, is_new_client):
    """
    Función pura de predicción.
    """
    le_zona = artifacts['le_zona']
    client_stats = artifacts['client_stats']
    global_avg = artifacts['global_win_rate']
    
    # Codificar Zona (Manejo de error si es zona nueva desconocida)
    try:
        zone_code = le_zona.transform([zone])[0]
    except ValueError:
        zone_code = 0 # Valor por defecto/seguro
        
    # Obtener tasa del cliente
    if is_new_client:
        client_rate = global_avg
    else:
        client_rate = client_stats.get(client_name, global_avg)
        
    # Input vector
    input_data = pd.DataFrame([[zone_code, client_rate]], columns=['Zona_Code', 'Client_Win_Rate'])
    
    # Predicción
    prob = model.predict_proba(input_data)[0][1]
    
    if prob > 0.65:
        label = "Alta Probabilidad"
    elif prob > 0.35:
        label = "Probabilidad Media"
    else:
        label = "Baja Probabilidad"
        
    return prob, label