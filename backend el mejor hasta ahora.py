"""
BACKEND - Sales Intelligence Platform v6.5.1
VERSIÓN v6.5.1 - Con aplicar_tema_oscuro() incluida
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
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
# TEMA OSCURO - v6.5.1 ✨
# ==========================================

def aplicar_tema_oscuro():
    """Aplica tema oscuro a la aplicación Streamlit - v6.5.1"""
    st.set_page_config(
        page_title="Sales Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    tema_css = """
    <style>
    :root {
        --color-primary: #FF6B6B;
        --color-secondary: #4ECDC4;
        --color-success: #6BCB77;
        --color-warning: #FFD93D;
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
# MAPEO DE CATEGORÍAS - v6.5.1
# ==========================================

CATEGORIA_KEYWORDS = {
    'Válvulas y Actuadores': [
        'válvula', 'valvula', 'actuador', 
        'jamesbury', 'neles', 'metso', 'newco', 'walworth',
        'bola', 'compuerta', 'mariposa', 'angulares', 'reguladora',
        'gv', 'btf', 'bv'
    ],
    'Instrumentos de Medición': [
        'indicador', 'medidor', 'medidores', 
        'transmisor', 'sensor', 'sonda',
        'magnetrol', 'jerguson', 'westlock',
        'nivel', 'presión', 'temperatura', 'flujo',
        'rtd', 'manometro', 'termometro', 'rotametro',
        'switch', 'interruptor', 'analizador',
        'iq70', 'iq90', 'iqs20', 'báscula', 'merrick',
        'wedgmeter', 'varec', 'modulevel', 'teltru',
        'instrumentacion', 'instrumentación', 'instrumentos'
    ],
    'Controles Eléctricos': [
        'arrancador', 'variador', 'inversor',
        'abb', 'siemens', 'plc',
        'controlador', 'módulo', 'modulo',
        'fuente', 'electronico', 'electromagnético',
        'electrónico', 'motor', 'motorizado',
        'ac500', 'sm1000',
        'elect', 'thermostat', 'chromalox',
        'moxa', 'protector', 'transientes'
    ],
    'Equipos de Automatización': [
        'automatización', 'automatizar',
        'sistema de control', 'autoclaves',
        'horno', 'unitronics', 'fieldlogger', 'novus',
        'generador', 'hipoclorito', 'cromatógrafo',
        'merrick', 'clorador'
    ],
    'Accesorios y Repuestos': [
        'repuesto', 'repuestos',
        'accesorio', 'accesorios',
        'empaque', 'empaques',
        'sello', 'sellos',
        'oring', 'brida', 'tubing', 'manifold',
        'kit', 'cabezote', 'tornillo',
        'disco', 'ruptura', 'correa', 'banda',
        'alambre', 'bellofram', 'tarjeta'
    ],
    'Servicios y Consultoría': [
        'servicio', 'servicios',
        'mantenimiento',
        'calibración', 'calibrar',
        'consultoría',
        'montaje', 'instalación',
        'reparación',
        'asistencia técnica',
        'toma muestras'
    ]
}

COLORES_CATEGORIAS = {
    'Válvulas y Actuadores': '#FF6B6B',
    'Instrumentos de Medición': '#4ECDC4',
    'Controles Eléctricos': '#FFD93D',
    'Equipos de Automatización': '#6BCB77',
    'Accesorios y Repuestos': '#A78BFA',
    'Servicios y Consultoría': '#FB7185',
    'Otros': '#9CA3AF'
}

COLORES_ZONAS = {
    'Bogotá': '#FF6B6B',
    'Medellín': '#4ECDC4',
    'Cali': '#FFD93D',
    'Barranquilla': '#6BCB77',
    'Cartagena': '#A78BFA'
}

def extraer_categoria(solicitud):
    """Extrae categoría de la solicitud - v6.5.1"""
    if pd.isna(solicitud):
        return 'Sin Categoría'
    
    solicitud_lower = str(solicitud).lower().strip()
    
    for categoria, keywords in CATEGORIA_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in solicitud_lower:
                return categoria
    
    return 'Otros'

# ==========================================
# MAPEO DE COLUMNAS - v6.5.1
# ==========================================

COLUMN_MAPPING = {
    'Cliente': ['Cliente', 'client', 'customer', 'CLIENTE'],
    'Zona Geográfica': ['Zona Geográfica', 'Zona', 'zona', 'Region', 'región', 'ZONA', 'region'],
    'Usuario_Interno': ['Usuario_Interno', 'Usuario interno', 'usuario interno', 'Vendedor', 'vendedor', 
                        'User', 'usuario', 'USUARIO', 'User_ID', 'user_id', 'Salesperson', 'salesperson',
                        'Usuario_interno', 'usuario_interno'],
    'Solicitud': ['Solicitud', 'Solicitud del Cliente', 'solicitud', 'Request', 'Requerimiento'],
    '¿Adjudicado?': ['¿Adjudicado?', 'Adjudicado', 'adjudicado', 'Adjudicada', 'won', 
                     'Won', 'ganada', 'Ganada', 'is_won', 'Sale_Status']
}

def normalizar_columnas(df):
    """Normaliza nombres de columnas - v6.5.1"""
    df_normalized = df.copy()
    renaming_dict = {}
    
    for required_col, alternatives in COLUMN_MAPPING.items():
        encontrada = False
        
        if required_col in df_normalized.columns:
            encontrada = True
            continue
        
        if not encontrada:
            for alt_name in alternatives:
                if alt_name in df_normalized.columns:
                    renaming_dict[alt_name] = required_col
                    encontrada = True
                    break
        
        if not encontrada:
            raise ValueError(
                f"❌ No se encontró columna para '{required_col}'\n"
                f"Alternativas: {', '.join(alternatives)}\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
    
    if renaming_dict:
        df_normalized = df_normalized.rename(columns=renaming_dict)
    
    return df_normalized

def limpiar_valores(df):
    """Limpia valores y extrae categoría - v6.5.1"""
    df_limpio = df.copy()
    
    df_limpio['Cliente'] = df_limpio['Cliente'].astype(str).str.strip()
    df_limpio['Zona Geográfica'] = df_limpio['Zona Geográfica'].astype(str).str.strip()
    df_limpio['Usuario_Interno'] = df_limpio['Usuario_Interno'].astype(str).str.strip()
    df_limpio['Solicitud'] = df_limpio['Solicitud'].astype(str).str.strip()
    
    df_limpio['Categoria_Producto'] = df_limpio['Solicitud'].apply(extraer_categoria)
    
    df_limpio = df_limpio.dropna(subset=['Cliente', 'Solicitud'])
    
    return df_limpio

def ordenar_lista_segura(lista):
    """Ordena lista de forma segura - v6.5.1"""
    try:
        lista_str = [str(x).strip() for x in lista if pd.notna(x) 
                     and str(x).strip() not in ['nan', 'None', 'NaN', 'Sin Categoría']]
        return sorted(list(set(lista_str)))
    except Exception as e:
        return [str(x).strip() for x in lista if pd.notna(x)]

# ==========================================
# CARGA DE DATOS - v6.5.1
# ==========================================

def load_data(file):
    """Carga datos Excel/CSV - v6.5.1"""
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return None, "❌ Formato no soportado. Usa Excel (.xlsx) o CSV (.csv)"
        
        try:
            df = normalizar_columnas(df)
        except ValueError as e:
            return None, str(e)
        
        try:
            df = limpiar_valores(df)
        except Exception as e:
            return None, f"❌ Error limpiando datos: {str(e)}"
        
        columnas_requeridas = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 
                              'Categoria_Producto', '¿Adjudicado?']
        
        for col in columnas_requeridas:
            if col not in df.columns:
                return None, f"❌ Falta columna: {col}"
        
        try:
            df['¿Adjudicado?'] = df['¿Adjudicado?'].astype(int)
        except:
            df['¿Adjudicado?'] = df['¿Adjudicado?'].map(
                {1: 1, 0: 0, 'Sí': 1, 'SI': 1, 'Yes': 1, 
                 'No': 0, 'NO': 0, True: 1, False: 0}
            ).fillna(0).astype(int)
        
        return df, None
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==========================================
# ENTRENAMIENTO DEL MODELO - v6.5.1
# ==========================================

def train_model_logic(df):
    """Entrena modelo ML - v6.5.1"""
    try:
        features = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        target = '¿Adjudicado?'
        
        for col in features + [target]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada")
        
        df_model = df[features + [target]].copy()
        
        df_model['Categoria_Producto'] = df_model['Categoria_Producto'].astype(str).str.strip()
        df_model['Cliente'] = df_model['Cliente'].astype(str).str.strip()
        df_model['Zona Geográfica'] = df_model['Zona Geográfica'].astype(str).str.strip()
        df_model['Usuario_Interno'] = df_model['Usuario_Interno'].astype(str).str.strip()
        
        df_model = df_model.dropna()
        
        if len(df_model) == 0:
            raise ValueError("Dataset vacío")
        
        df_encoded = pd.get_dummies(
            df_model,
            columns=['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto'],
            drop_first=True
        )
        
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        
        stats_categoria = []
        categorias_ordenadas = ordenar_lista_segura(df['Categoria_Producto'].unique())
        
        for cat in categorias_ordenadas:
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
        
        artifacts = {
            'feature_names': list(X.columns),
            'unique_clients': ordenar_lista_segura(df['Cliente'].unique()),
            'unique_zones': ordenar_lista_segura(df['Zona Geográfica'].unique()),
            'categorias': categorias_ordenadas,
            'unique_usuarios': ordenar_lista_segura(df['Usuario_Interno'].unique()),
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
        raise ValueError(f"❌ Error: {str(e)}")

# ==========================================
# GUARDADO Y CARGA DE MODELOS - v6.5.1
# ==========================================

def save_model_artifacts(model, artifacts):
    """Guarda modelo y artifacts"""
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ARTIFACTS_PATH, 'w') as f:
            json.dump(artifacts, f)
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def load_model_artifacts():
    """Carga modelo y artifacts"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACTS_PATH):
            return None, None
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ARTIFACTS_PATH, 'r') as f:
            artifacts = json.load(f)
        return model, artifacts
    except:
        return None, None

# ==========================================
# PREDICCIÓN - v6.5.1
# ==========================================

def make_prediction(model, artifacts, client, zona, categoria, usuario, is_new=False):
    """Realiza predicción - v6.5.1"""
    try:
        client = str(client).strip()
        zona = str(zona).strip()
        categoria = str(categoria).strip()
        usuario = str(usuario).strip()
        
        prediction_data = pd.DataFrame({
            'Cliente': [client],
            'Zona Geográfica': [zona],
            'Usuario_Interno': [usuario],
            'Categoria_Producto': [categoria]
        })
        
        prediction_encoded = pd.get_dummies(
            prediction_data,
            columns=['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        )
        
        for feature in artifacts['feature_names']:
            if feature not in prediction_encoded.columns:
                prediction_encoded[feature] = 0
        
        X_pred = prediction_encoded[artifacts['feature_names']]
        
        prob = model.predict_proba(X_pred)[0][1]
        label = "Probable" if prob > 0.5 else "Improbable"
        
        stats_cat = next(
            (s for s in artifacts['stats_categoria'] if s['Categoria'] == categoria),
            None
        )
        
        if stats_cat:
            categoria_tasa = stats_cat['Tasa_Conversion']
            categoria_ventas = stats_cat['Ganadas']
            
            if prob > 0.7:
                recomendacion = f"🟢 {categoria}: {categoria_tasa:.0%} ({categoria_ventas} ventas)"
            elif prob > 0.5:
                recomendacion = f"🟡 {categoria}: {categoria_tasa:.0%} ({categoria_ventas} ventas)"
            else:
                recomendacion = f"🔴 {categoria}: {categoria_tasa:.0%} ({categoria_ventas} ventas)"
        else:
            recomendacion = "⚠️ Categoría no encontrada"
        
        color = '#6BCB77' if prob > 0.7 else '#FFD93D' if prob > 0.5 else '#FF6B6B'
        
        return prob, label, recomendacion, color
    
    except Exception as e:
        return 0, "Error", f"❌ Error: {str(e)}", '#FF6B6B'

# ==========================================
# VISUALIZACIONES - v6.5.1
# ==========================================

def grafico_barras_horizontal(df, x, y, titulo, color='#FF6B6B'):
    """Gráfico barras horizontal - v6.5.1"""
    fig = px.bar(df, x=x, y=y, orientation='h', title=f'<b>{titulo}</b>',
                 color_discrete_sequence=[color], height=400, text=x)
    
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#111823',
        font=dict(color='#FFFFFF', size=12), showlegend=False,
        title_font_size=16, xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        yaxis=dict(showgrid=False), margin=dict(l=100, r=50, t=80, b=50),
        hovermode='y unified'
    )
    fig.update_traces(textposition='outside', textfont_size=11)
    return fig

def grafico_barras_vertical(df, x, y, titulo, color='#4ECDC4'):
    """Gráfico barras vertical - v6.5.1"""
    fig = px.bar(df, x=x, y=y, title=f'<b>{titulo}</b>',
                 color_discrete_sequence=[color], height=400, text=y)
    
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#111823',
        font=dict(color='#FFFFFF', size=12), showlegend=False,
        title_font_size=16, xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        margin=dict(l=50, r=50, t=80, b=80), hovermode='x unified'
    )
    fig.update_traces(textposition='outside', textfont_size=11)
    return fig

def grafico_pie_categorias(df, categoria_col, titulo, colores=None):
    """Gráfico pastel - v6.5.1"""
    if colores is None:
        colores = list(COLORES_CATEGORIAS.values())
    
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    fig = px.pie(df, names=categoria_col,
                 values=df.iloc[:, 0] if len(df.columns) > 1 else None,
                 title=f'<b>{titulo}</b>', color_discrete_sequence=colores, height=450)
    
    fig.update_layout(
        paper_bgcolor='#0E1117', font=dict(color='#FFFFFF', size=12),
        title_font_size=16, margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True, legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0)',
                                     bordercolor='#31333D', borderwidth=1)
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value}<extra></extra>',
                     textposition='auto', textinfo='label+percent')
    return fig

def grafico_donut(df, labels, values, titulo, colores=None):
    """Gráfico dona - v6.5.1"""
    if colores is None:
        colores = list(COLORES_CATEGORIAS.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                 marker=dict(colors=colores),
                                 hovertemplate='<b>%{label}</b><br>%{value}<extra></extra>')])
    
    fig.update_layout(
        title_text=f'<b>{titulo}</b>', paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', size=12), title_font_size=16,
        margin=dict(l=50, r=50, t=80, b=50), height=450
    )
    return fig

def grafico_linea_tendencia(df, x, y, titulo, color='#4ECDC4'):
    """Gráfico línea - v6.5.1"""
    fig = px.line(df, x=x, y=y, title=f'<b>{titulo}</b>',
                  color_discrete_sequence=[color], height=400, markers=True)
    
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#111823',
        font=dict(color='#FFFFFF', size=12), showlegend=False,
        title_font_size=16, xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        margin=dict(l=50, r=50, t=80, b=50), hovermode='x unified'
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8),
                     hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>')
    return fig

def mostrar_kpi_grande(titulo, valor, color='#4D96FF', icon='📊'):
    """Muestra KPI grande - v6.5.1"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 25px; border-radius: 12px; border-left: 5px solid {color};
                text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <p style="color: #888; margin: 0; font-size: 13px; text-transform: uppercase;">
            {icon} {titulo}</p>
        <p style="color: {color}; margin: 12px 0 0 0; font-size: 36px; font-weight: bold;">
            {valor}</p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color):
    """Muestra tarjeta predicción - v6.5.1"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 25px; border-radius: 12px; border-left: 5px solid {color};
                margin: 20px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <h3 style="color: {color}; margin-top: 0;">🎯 Predicción</h3>
        <p style="color: #FFFFFF; font-size: 14px;"><strong>Categoría:</strong> {categoria}</p>
        <p style="color: #FFFFFF; font-size: 14px;">
            <strong>Probabilidad:</strong> <span style="color: {color}; font-size: 28px;">
                {prob:.1%}</span></p>
        <p style="color: #AAA; font-size: 13px;">{recomendacion}</p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_crosssell(categoria, tasa, ventas, color='#4ECDC4'):
    """Muestra tarjeta cross-sell - v6.5.1"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 10px; border-left: 4px solid {color};
                margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FFFFFF; font-weight: 600;">📦 {categoria}</span>
            <div style="display: flex; gap: 20px;">
                <div style="text-align: center;">
                    <span style="color: {color}; font-size: 18px; font-weight: bold;">
                        {tasa:.0%}</span>
                    <div style="color: #888; font-size: 11px;">Tasa</div>
                </div>
                <div style="text-align: center;">
                    <span style="color: {color}; font-size: 18px; font-weight: bold;">
                        {ventas}</span>
                    <div style="color: #888; font-size: 11px;">Ventas</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# REPORTES - v6.5.1
# ==========================================

def generar_resumen_entrenamiento(df, model, artifacts, metrics):
    """Genera resumen - v6.5.1"""
    return {
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

def limpiar_archivo(ruta):
    """Limpia archivo - v6.5.1"""
    if os.path.exists(ruta):
        try:
            os.remove(ruta)
            return True
        except:
            return False
    return True
