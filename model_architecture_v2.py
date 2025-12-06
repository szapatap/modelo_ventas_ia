import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse as sp

# ==========================================
# 1. PREPARACIÓN DE DATOS (MEJORADO)
# ==========================================

def load_data(uploaded_file):
    """
    Carga y limpia el dataset desde un archivo Excel o CSV.
    Ahora también procesa la columna 'Solicitud'.
    """
    try:
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
    
    # Llenar valores NaN en 'Solicitud' con "Sin especificar"
    if 'Solicitud' in df.columns:
        df['Solicitud'] = df['Solicitud'].fillna('Sin especificar')
    
    # Eliminar filas donde el Target o Zona sean nulos
    df = df.dropna(subset=['Zona Geográfica', '¿Adjudicado?', 'Cliente'])
    
    return df, None


# ==========================================
# 2. ENTRENAMIENTO DEL MODELO (MEJORADO)
# ==========================================

def train_model_logic(df):
    """
    Pipeline de entrenamiento mejorado:
    1. Target Encoding para Clientes
    2. Label Encoding para Zonas
    3. TF-IDF Vectorization para Solicitudes
    4. Combinación de features
    5. Random Forest
    """
    
    # 1. Feature Engineering: Historial del Cliente
    client_stats = df.groupby('Cliente')['¿Adjudicado?'].mean().to_dict()
    global_win_rate = df['¿Adjudicado?'].mean()
    df['Client_Win_Rate'] = df['Cliente'].map(client_stats)
    
    # 2. Feature Engineering: Zona (Label Encoding)
    le_zona = LabelEncoder()
    df['Zona_Code'] = le_zona.fit_transform(df['Zona Geográfica'])
    
    # 3. Feature Engineering: Solicitud (TF-IDF Vectorization)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10,  # Limitar a los 10 términos más importantes
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2)  # Unigramas y bigramas
    )
    
    # Asegurar que 'Solicitud' existe
    if 'Solicitud' not in df.columns:
        df['Solicitud'] = 'Sin especificar'
    
    solicitud_tfidf = tfidf_vectorizer.fit_transform(df['Solicitud'].fillna('Sin especificar'))
    
    # 4. Combinar features (columnas numéricas + matriz TF-IDF)
    X_numeric = df[['Zona_Code', 'Client_Win_Rate']].values
    X_combined = sp.hstack([X_numeric, solicitud_tfidf]).toarray()
    
    y = df['¿Adjudicado?']
    
    # 5. División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    # 6. Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # Manejo de desbalance
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {'accuracy': acc, 'report': report}
    
    # 8. Artefactos necesarios para producción
    artifacts = {
        'client_stats': client_stats,
        'global_win_rate': global_win_rate,
        'le_zona': le_zona,
        'tfidf_vectorizer': tfidf_vectorizer,
        'unique_zones': list(le_zona.classes_),
        'unique_clients': list(client_stats.keys()),
        'unique_solicitudes': df['Solicitud'].unique().tolist()
    }
    
    return model, metrics, artifacts


# ==========================================
# 3. GUARDADO Y CARGA
# ==========================================

def save_model_artifacts(model, artifacts, folder='models'):
    """Guarda el modelo entrenado en disco."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    joblib.dump(model, os.path.join(folder, 'sales_model_v2.pkl'))
    joblib.dump(artifacts, os.path.join(folder, 'artifacts_v2.pkl'))


def load_model_artifacts(folder='models'):
    """Carga el modelo desde disco."""
    try:
        model = joblib.load(os.path.join(folder, 'sales_model_v2.pkl'))
        artifacts = joblib.load(os.path.join(folder, 'artifacts_v2.pkl'))
        return model, artifacts
    except:
        return None, None


# ==========================================
# 4. LÓGICA DE PREDICCIÓN (MEJORADA)
# ==========================================

def make_prediction(model, artifacts, client_name, zone, solicitud, is_new_client):
    """
    Función pura de predicción con la nueva variable.
    
    Args:
        model: Modelo entrenado
        artifacts: Diccionario de artefactos
        client_name: Nombre del cliente
        zone: Zona geográfica
        solicitud: Tipo de solicitud/requerimiento
        is_new_client: Boolean indicando si es cliente nuevo
    
    Returns:
        prob: Probabilidad de adjudicación (0-1)
        label: Categoría de probabilidad
    """
    
    le_zona = artifacts['le_zona']
    client_stats = artifacts['client_stats']
    global_avg = artifacts['global_win_rate']
    tfidf_vectorizer = artifacts['tfidf_vectorizer']
    
    # Codificar Zona
    try:
        zone_code = le_zona.transform([zone])[0]
    except ValueError:
        zone_code = 0
    
    # Obtener tasa del cliente
    if is_new_client:
        client_rate = global_avg
    else:
        client_rate = client_stats.get(client_name, global_avg)
    
    # Vectorizar solicitud
    solicitud_tfidf = tfidf_vectorizer.transform([solicitud]).toarray()
    
    # Construir vector de entrada
    input_numeric = np.array([[zone_code, client_rate]])
    input_data = np.hstack([input_numeric, solicitud_tfidf])
    
    # Predicción
    prob = model.predict_proba(input_data)[0][1]
    
    if prob > 0.65:
        label = "Alta Probabilidad ⭐"
    elif prob > 0.35:
        label = "Probabilidad Media 📊"
    else:
        label = "Baja Probabilidad ⚠️"
    
    return prob, label


# ==========================================
# 5. ANÁLISIS DE IMPORTANCIA DE FEATURES
# ==========================================

def get_feature_importance(model, tfidf_feature_names):
    """
    Extrae la importancia de features para interpretabilidad.
    """
    importances = model.feature_importances_
    
    feature_names = ['Zona_Code', 'Client_Win_Rate'] + tfidf_feature_names
    
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance
