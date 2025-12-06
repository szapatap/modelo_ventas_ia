"""
BACKEND - Sales Intelligence Platform v6.3
Procesamiento de datos y lógica del modelo ML
VERSIÓN v6.3 - Categoría incluida como feature
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os
from datetime import datetime

# ==========================================
# CONFIGURACIÓN
# ==========================================

MODEL_PATH = 'model.pkl'
ARTIFACTS_PATH = 'artifacts.json'

# ==========================================
# TEMA OSCURO
# ==========================================

def aplicar_tema_oscuro():
    """Aplica tema oscuro a la aplicación"""
    st.set_page_config(
        page_title="Sales Intelligence Platform",
        page_icon="📊",
        layout="wide"
    )
    
    tema_css = """
    <style>
    :root {
        --color-primary: #FF6B6B;
        --color-secondary: #4ECDC4;
        --color-success: #6BCB77;
        --color-warning: #FFD93D;
        --color-danger: #FF6B6B;
    }
    
    body {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stMetric {
        background-color: #161B22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363D;
    }
    
    .stDataFrame {
        background-color: #0D1117;
    }
    </style>
    """
    st.markdown(tema_css, unsafe_allow_html=True)

# ==========================================
# CARGA DE DATOS
# ==========================================

def load_data(file):
    """Carga datos desde archivo Excel o CSV"""
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return None, "Formato no soportado"
        
        # Validar columnas requeridas
        columnas_requeridas = [
            'Cliente', 
            'Zona Geográfica', 
            'Usuario_Interno', 
            'Categoria_Producto',  # ✅ REQUERIDA en v6.3
            '¿Adjudicado?'
        ]
        
        for col in columnas_requeridas:
            if col not in df.columns:
                return None, f"Falta columna: {col}"
        
        return df, None
    
    except Exception as e:
        return None, str(e)

# ==========================================
# ENTRENAMIENTO DEL MODELO - v6.3
# ==========================================

def train_model_logic(df):
    """Entrena el modelo ML con Categoría incluida - v6.3"""
    try:
        # ✅ FEATURES INCLUYENDO CATEGORÍA
        features = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        target = '¿Adjudicado?'
        
        # Validar que existan todas las columnas
        for col in features + [target]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el dataset")
        
        # Preparar datos
        df_model = df[features + [target]].copy()
        
        # Eliminar filas con valores nulos
        df_model = df_model.dropna()
        
        # One-hot encoding para variables categóricas
        # ✅ INCLUYE Categoria_Producto
        df_encoded = pd.get_dummies(
            df_model,
            columns=['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto'],
            drop_first=True
        )
        
        # Separar features y target
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        # Entrenar modelo
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Calcular accuracy
        accuracy = model.score(X, y)
        
        # ✅ ESTADÍSTICAS POR CATEGORÍA
        stats_categoria = []
        for cat in sorted(df['Categoria_Producto'].unique()):
            df_cat = df[df['Categoria_Producto'] == cat]
            ganadas = len(df_cat[df_cat['¿Adjudicado?'] == 1])
            total = len(df_cat)
            tasa = (ganadas / total) if total > 0 else 0
            
            stats_categoria.append({
                'Categoria': cat,
                'Total': total,
                'Ganadas': ganadas,
                'Tasa_Conversion': tasa
            })
        
        # Guardar artifacts para predicción
        # ✅ INCLUYE feature_names para alineación
        artifacts = {
            'feature_names': list(X.columns),
            'unique_clients': sorted(df['Cliente'].unique().tolist()),
            'unique_zones': sorted(df['Zona Geográfica'].unique().tolist()),
            'categorias': sorted(df['Categoria_Producto'].unique().tolist()),  # ✅
            'unique_usuarios': sorted(df['Usuario_Interno'].unique().tolist()),
            'stats_categoria': stats_categoria,
            'model_accuracy': accuracy
        }
        
        metrics = {
            'accuracy': accuracy,
            'n_samples': len(df),
            'n_features': len(X.columns)
        }
        
        return model, metrics, artifacts
    
    except Exception as e:
        raise ValueError(f"Error en entrenamiento: {str(e)}")

# ==========================================
# GUARDADO Y CARGA DE MODELOS
# ==========================================

def save_model_artifacts(model, artifacts):
    """Guarda modelo y artifacts"""
    try:
        # Guardar modelo
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        # Guardar artifacts
        with open(ARTIFACTS_PATH, 'w') as f:
            json.dump(artifacts, f)
        
        return True
    
    except Exception as e:
        print(f"Error guardando modelo: {str(e)}")
        return False

def load_model_artifacts():
    """Carga modelo y artifacts guardados"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACTS_PATH):
            return None, None
        
        # Cargar modelo
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Cargar artifacts
        with open(ARTIFACTS_PATH, 'r') as f:
            artifacts = json.load(f)
        
        return model, artifacts
    
    except Exception as e:
        print(f"Error cargando modelo: {str(e)}")
        return None, None

# ==========================================
# PREDICCIÓN - v6.3
# ==========================================

def make_prediction(model, artifacts, client, zona, categoria, usuario, is_new=False):
    """Realiza predicción CON Categoría incluida - v6.3"""
    try:
        # ✅ CREAR DATAFRAME CON CATEGORÍA
        prediction_data = pd.DataFrame({
            'Cliente': [client],
            'Zona Geográfica': [zona],
            'Usuario_Interno': [usuario],
            'Categoria_Producto': [categoria]  # ✅ INCLUIDA
        })
        
        # ✅ ONE-HOT ENCODING INCLUYENDO CATEGORÍA
        prediction_encoded = pd.get_dummies(
            prediction_data,
            columns=['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        )
        
        # ✅ ALINEAR FEATURES CON EL MODELO
        # Agregar columnas faltantes con valor 0
        for feature in artifacts['feature_names']:
            if feature not in prediction_encoded.columns:
                prediction_encoded[feature] = 0
        
        # Seleccionar features en orden correcto
        X_pred = prediction_encoded[artifacts['feature_names']]
        
        # ✅ HACER PREDICCIÓN
        prob = model.predict_proba(X_pred)[0][1]
        
        # Determinar label
        label = "Probable" if prob > 0.5 else "Improbable"
        
        # ✅ OBTENER RECOMENDACIÓN BASADA EN CATEGORÍA
        stats_cat = next(
            (s for s in artifacts['stats_categoria'] if s['Categoria'] == categoria),
            None
        )
        
        if stats_cat:
            categoria_tasa = stats_cat['Tasa_Conversion']
            categoria_ventas = stats_cat['Ganadas']
            
            # Recomendación personalizada por probabilidad
            if prob > 0.7:
                recomendacion = (
                    f"🟢 Alta probabilidad de cierre. "
                    f"{categoria}: {categoria_tasa:.0%} tasa histórica "
                    f"({categoria_ventas} ventas)"
                )
            elif prob > 0.5:
                recomendacion = (
                    f"🟡 Probabilidad media. "
                    f"{categoria}: {categoria_tasa:.0%} tasa histórica "
                    f"({categoria_ventas} ventas)"
                )
            else:
                recomendacion = (
                    f"🔴 Baja probabilidad. Considerar estrategia diferente. "
                    f"{categoria}: {categoria_tasa:.0%} tasa histórica "
                    f"({categoria_ventas} ventas)"
                )
        else:
            recomendacion = "⚠️ Categoría no encontrada en datos de entrenamiento"
        
        # Determinar color según probabilidad
        if prob > 0.7:
            color = '#6BCB77'  # Verde
        elif prob > 0.5:
            color = '#FFD93D'  # Amarillo
        else:
            color = '#FF6B6B'  # Rojo
        
        return prob, label, recomendacion, color
    
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return 0, "Error", f"Error al realizar predicción: {str(e)}", '#FF6B6B'

# ==========================================
# VISUALIZACIONES
# ==========================================

def grafico_barras_horizontal(df, x, y, titulo, color='#FF6B6B'):
    """Gráfico de barras horizontal"""
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation='h',
        title=titulo,
        color_discrete_sequence=[color],
        height=400
    )
    
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#111823',
        font_color='#FFFFFF',
        showlegend=False,
        title_font_size=16,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        yaxis=dict(showgrid=False),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def grafico_pie_categorias(df, categoria_col, titulo, colores=None):
    """Gráfico de pastel"""
    fig = px.pie(
        df,
        names=categoria_col,
        values='Tasa' if 'Tasa' in df.columns else None,
        title=titulo,
        color_discrete_sequence=colores,
        height=400
    )
    
    fig.update_layout(
        paper_bgcolor='#0E1117',
        font_color='#FFFFFF',
        title_font_size=16,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def mostrar_kpi_grande(titulo, valor, color='#4D96FF', icon='📊'):
    """Muestra KPI grande"""
    st.markdown(f"""
    <div style="background-color: #161B22; padding: 20px; border-radius: 10px; 
    border-left: 4px solid {color}; text-align: center;">
        <p style="color: #888; margin: 0; font-size: 12px;">{icon} {titulo}</p>
        <p style="color: #FFF; margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{valor}</p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color):
    """Muestra tarjeta de predicción"""
    st.markdown(f"""
    <div style="background-color: #161B22; padding: 20px; border-radius: 10px; 
    border-left: 4px solid {color}; margin: 20px 0;">
        <h3 style="color: {color}; margin-top: 0;">Predicción de Adjudicación</h3>
        <p style="color: #FFF; font-size: 14px; margin: 10px 0;">
            <strong>Categoría:</strong> {categoria}
        </p>
        <p style="color: #FFF; font-size: 14px; margin: 10px 0;">
            <strong>Probabilidad:</strong> <span style="color: {color}; font-size: 24px; font-weight: bold;">{prob:.1%}</span>
        </p>
        <p style="color: #AAA; font-size: 13px; margin: 10px 0;">
            {recomendacion}
        </p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_crosssell(categoria, tasa, ventas):
    """Muestra oportunidad de cross-sell"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"📦 **{categoria}**")
    
    with col2:
        st.metric("Tasa", f"{tasa:.0%}")
    
    with col3:
        st.metric("Ventas", ventas)

# ==========================================
# ANÁLISIS Y REPORTES
# ==========================================

def generar_resumen_entrenamiento(df, model, artifacts, metrics):
    """Genera resumen del entrenamiento"""
    resumen = {
        'timestamp': datetime.now().isoformat(),
        'n_registros': len(df),
        'n_clientes': df['Cliente'].nunique(),
        'n_zonas': df['Zona Geográfica'].nunique(),
        'n_categorias': df['Categoria_Producto'].nunique(),
        'n_vendedores': df['Usuario_Interno'].nunique(),
        'accuracy': metrics['accuracy'],
        'n_features': metrics['n_features'],
        'tasa_adjudicacion': df['¿Adjudicado?'].mean()
    }
    
    return resumen

# ==========================================
# UTILIDADES
# ==========================================

def limpiar_archivo(ruta):
    """Elimina archivo si existe"""
    if os.path.exists(ruta):
        try:
            os.remove(ruta)
            return True
        except:
            return False
    return True
