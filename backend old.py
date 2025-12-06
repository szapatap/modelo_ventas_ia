"""
BACKEND CONSOLIDADO - Sales Intelligence Platform v3
Contiene: Clasificador + Modelo ML + Utilidades de Dashboard
VERSIÓN CORREGIDA - Fix para KeyError Categoria_Producto
"""

import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import scipy.sparse as sp

# ==========================================
# PARTE 1: CLASIFICADOR DE SOLICITUDES
# ==========================================

CATEGORIAS = {
    'Válvulas y Actuadores': {
        'keywords': [
            'válvula', 'actuador', 'mariposa', 'angular', 'control',
            'jamesbury', 'neles', 'metso', 'ruptura', 'solenoide',
            'neumático', 'hidráulico', 'proporcional'
        ],
        'emoji': '🔧',
        'color': '#FF6B6B'
    },
    'Sistemas de Control': {
        'keywords': [
            'switch', 'unitronics', 'arrancador', 'electronica', 'medidor',
            'flujo', 'presión', 'nivel', 'temperatura', 'controlador',
            'abb', 'siemens', 'plc', 'automatización', 'variador'
        ],
        'emoji': '⚙️',
        'color': '#4ECDC4'
    },
    'Equipos de Laboratorio e Instrumentación': {
        'keywords': [
            'instrumentación', 'indicador', 'westlock', 'digi', 'magnetico',
            'medidor', 'agua', 'laboratorio', 'sensor', 'transmisor',
            'calibrador', 'registrador', 'instrumento'
        ],
        'emoji': '🔬',
        'color': '#95E1D3'
    },
    'Producto General': {
        'keywords': ['general', 'otros', 'sin especificar'],
        'emoji': '📦',
        'color': '#A0A0A0'
    }
}

def similitud_cadena(a, b):
    """Calcula similitud entre dos cadenas (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def clasificar_solicitud(solicitud_texto):
    """
    Clasifica una solicitud a una categoría de producto.
    Retorna: (categoria, emoji, color, confianza)
    """
    if not solicitud_texto or solicitud_texto.strip().lower() in ['sin especificar', 'nan', '']:
        return 'Producto General', '📦', '#A0A0A0', 0.5
    
    solicitud_lower = solicitud_texto.lower()
    puntuaciones = {}
    
    for categoria, config in CATEGORIAS.items():
        coincidencias = sum(1 for keyword in config['keywords'] if keyword in solicitud_lower)
        if coincidencias > 0:
            puntuaciones[categoria] = coincidencias
    
    if puntuaciones:
        categoria_ganadora = max(puntuaciones, key=puntuaciones.get)
        confianza = min(0.95, 0.6 + (puntuaciones[categoria_ganadora] * 0.15))
        config = CATEGORIAS[categoria_ganadora]
        return categoria_ganadora, config['emoji'], config['color'], confianza
    
    mejor_similitud = 0
    categoria_similar = 'Producto General'
    
    for categoria, config in CATEGORIAS.items():
        if categoria != 'Producto General':
            keywords_str = ' '.join(config['keywords'])
            sim = similitud_cadena(solicitud_lower, keywords_str)
            if sim > mejor_similitud and sim > 0.3:
                mejor_similitud = sim
                categoria_similar = categoria
    
    if mejor_similitud > 0.3:
        config = CATEGORIAS[categoria_similar]
        return categoria_similar, config['emoji'], config['color'], mejor_similitud
    
    return 'Producto General', '📦', '#A0A0A0', 0.5

def clasificar_batch(df, columna_solicitud='Solicitud'):
    """Clasifica múltiples solicitudes"""
    resultados = []
    
    for solicitud in df[columna_solicitud].fillna('Sin especificar'):
        categoria, emoji, color, confianza = clasificar_solicitud(str(solicitud))
        resultados.append({
            'Categoria_Producto': categoria,
            'Emoji': emoji,
            'Color': color,
            'Confianza': confianza
        })
    
    resultado_df = pd.DataFrame(resultados)
    df_final = pd.concat([df, resultado_df], axis=1)
    return df_final

# ==========================================
# PARTE 2: MODELO ML Y DATOS
# ==========================================

def load_data(uploaded_file):
    """Carga, limpia y enriquece el dataset - VERSION CORREGIDA"""
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
    
    # Limpiar nombres de columnas
    df.columns = [c.strip() for c in df.columns]
    
    # Validar columnas requeridas
    required_cols = ['Cliente', 'Zona Geográfica', '¿Adjudicado?']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return None, f"Faltan columnas: {', '.join(missing_cols)}"
    
    # Procesar Fecha si existe
    if 'Fecha' in df.columns:
        try:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df['Año'] = df['Fecha'].dt.year
            df['Mes'] = df['Fecha'].dt.month
            df['Mes_Nombre'] = df['Fecha'].dt.month_name()
            df['Trimestre'] = df['Fecha'].dt.quarter
        except:
            pass
    
    # IMPORTANTE: Clasificar Solicitud si existe
    if 'Solicitud' in df.columns:
        df['Solicitud'] = df['Solicitud'].fillna('Sin especificar')
        df = clasificar_batch(df, 'Solicitud')
    else:
        # Si no existe columna Solicitud, crear categoría por defecto
        df['Categoria_Producto'] = 'Producto General'
        df['Emoji'] = '📦'
        df['Color'] = '#A0A0A0'
        df['Confianza'] = 0.5
    
    # Usuario interno
    if 'Usuario interno' in df.columns:
        df['Usuario_Interno'] = df['Usuario interno'].fillna('Sin asignar')
    elif 'Usuario_Interno' not in df.columns:
        df['Usuario_Interno'] = 'Sin asignar'
    
    # Convertir ¿Adjudicado? a binario
    if df['¿Adjudicado?'].dtype == 'object':
        df['¿Adjudicado?'] = df['¿Adjudicado?'].map({
            'Sí': 1, 'sí': 1, 'Si': 1, 'SI': 1, 'Yes': 1, 'yes': 1, '1': 1, 1: 1,
            'No': 0, 'no': 0, 'NO': 0, 'None': 0, '0': 0, 0: 0
        })
    
    # Eliminar filas con valores faltantes críticos
    df = df.dropna(subset=['Zona Geográfica', '¿Adjudicado?', 'Cliente', 'Usuario_Interno'])
    
    # Validar que tenemos datos
    if len(df) == 0:
        return None, "No hay datos válidos después de limpiar"
    
    return df, None

def train_model_logic(df):
    """Pipeline de entrenamiento con Gradient Boosting - VERSION CORREGIDA"""
    
    try:
        # Verificar que Categoria_Producto existe
        if 'Categoria_Producto' not in df.columns:
            return None, {"error": "Falta Categoria_Producto"}, {}
        
        # Feature Engineering
        client_stats = df.groupby('Cliente')['¿Adjudicado?'].mean().to_dict()
        global_win_rate = df['¿Adjudicado?'].mean()
        df['Client_Win_Rate'] = df['Cliente'].map(client_stats)
        
        le_zona = LabelEncoder()
        df['Zona_Code'] = le_zona.fit_transform(df['Zona Geográfica'])
        
        le_categoria = LabelEncoder()
        df['Categoria_Code'] = le_categoria.fit_transform(df['Categoria_Producto'])
        
        le_usuario = LabelEncoder()
        df['Usuario_Code'] = le_usuario.fit_transform(df['Usuario_Interno'])
        
        X_numeric = df[['Zona_Code', 'Client_Win_Rate', 'Categoria_Code', 'Usuario_Code']].values
        y = df['¿Adjudicado?']
        
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {'accuracy': acc, 'report': report, 'confusion_matrix': conf_matrix.tolist()}
        
        # Estadísticas
        stats_categoria = df.groupby('Categoria_Producto')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_categoria.columns = ['Categoria', 'Total', 'Ganadas', 'Tasa_Conversion']
        
        stats_zona = df.groupby('Zona Geográfica')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_zona.columns = ['Zona', 'Total', 'Ganadas', 'Tasa_Conversion']
        
        stats_usuario = df.groupby('Usuario_Interno')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_usuario.columns = ['Usuario', 'Total', 'Ganadas', 'Tasa_Conversion']
        
        artifacts = {
            'client_stats': client_stats,
            'global_win_rate': global_win_rate,
            'le_zona': le_zona,
            'le_categoria': le_categoria,
            'le_usuario': le_usuario,
            'unique_zones': list(le_zona.classes_),
            'unique_clients': list(client_stats.keys()),
            'unique_usuarios': list(le_usuario.classes_),
            'categorias': ['Válvulas y Actuadores', 'Sistemas de Control', 
                          'Equipos de Laboratorio e Instrumentación', 'Producto General'],
            'stats_categoria': stats_categoria.to_dict('records'),
            'stats_zona': stats_zona.to_dict('records'),
            'stats_usuario': stats_usuario.to_dict('records')
        }
        
        return model, metrics, artifacts
    
    except Exception as e:
        return None, {"error": str(e)}, {}

def save_model_artifacts(model, artifacts, folder='models'):
    """Guarda el modelo entrenado"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    joblib.dump(model, os.path.join(folder, 'sales_model.pkl'))
    joblib.dump(artifacts, os.path.join(folder, 'artifacts.pkl'))

def load_model_artifacts(folder='models'):
    """Carga el modelo desde disco"""
    try:
        model = joblib.load(os.path.join(folder, 'sales_model.pkl'))
        artifacts = joblib.load(os.path.join(folder, 'artifacts.pkl'))
        return model, artifacts
    except:
        return None, None

def make_prediction(model, artifacts, client_name, zone, categoria_producto, usuario='Sin asignar', is_new_client=False):
    """Predicción mejorada con recomendaciones"""
    
    le_zona = artifacts['le_zona']
    le_categoria = artifacts['le_categoria']
    le_usuario = artifacts['le_usuario']
    client_stats = artifacts['client_stats']
    global_avg = artifacts['global_win_rate']
    
    try:
        zone_code = le_zona.transform([zone])[0]
    except:
        zone_code = 0
    
    try:
        categoria_code = le_categoria.transform([categoria_producto])[0]
    except:
        categoria_code = 0
    
    try:
        usuario_code = le_usuario.transform([usuario])[0]
    except:
        usuario_code = 0
    
    if is_new_client:
        client_rate = global_avg
    else:
        client_rate = client_stats.get(client_name, global_avg)
    
    input_data = np.array([[zone_code, client_rate, categoria_code, usuario_code]])
    prob = model.predict_proba(input_data)[0][1]
    
    if prob > 0.70:
        label = "🔥 Muy Alta Probabilidad"
        recomendacion = "✅ Preparar propuesta formal inmediatamente"
        color = "#FF6B6B"
    elif prob > 0.55:
        label = "⭐ Alta Probabilidad"
        recomendacion = "📞 Contactar cliente para detalles técnicos"
        color = "#FFD93D"
    elif prob > 0.40:
        label = "📊 Probabilidad Media"
        recomendacion = "⏳ Seguimiento periódico, esperar señales"
        color = "#6BCB77"
    elif prob > 0.25:
        label = "⚠️  Baja Probabilidad"
        recomendacion = "🔍 Necesita investigación adicional"
        color = "#4D96FF"
    else:
        label = "❌ Muy Baja Probabilidad"
        recomendacion = "⛔ Enfocarse en otros leads"
        color = "#FF6B9D"
    
    return prob, label, recomendacion, color

# ==========================================
# PARTE 3: UTILIDADES DE DASHBOARD
# ==========================================

COLORES_VIBRANTES = {
    'rojo': '#FF6B6B',
    'naranja': '#FFA500',
    'amarillo': '#FFD93D',
    'verde': '#6BCB77',
    'verde_claro': '#95E1D3',
    'azul': '#4D96FF',
    'púrpura': '#BB6BD9',
    'rosa': '#FF6B9D',
    'gris': '#3A3F47',
    'gris_claro': '#5A6268'
}

TEMA_OSCURO = {
    'paper_bgcolor': '#0E1117',
    'plot_bgcolor': '#111823',
    'font_color': '#FFFFFF',
    'gridcolor': '#31333D',
    'margin': dict(l=50, r=50, t=50, b=50)
}

def aplicar_tema_oscuro():
    """Aplica tema oscuro a toda la app"""
    st.markdown("""
    <style>
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #4ECDC4;
            --background-color: #0E1117;
            --surface-color: #111823;
        }
        
        [data-testid="stAppViewContainer"] { background-color: #0E1117; }
        [data-testid="stHeader"] { background-color: #0E1117; }
        [data-testid="stSidebar"] { background-color: #111823; }
        [data-testid="stMarkdownContainer"] { color: #FFFFFF; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { color: #999; border-bottom: 2px solid transparent; }
        .stTabs [aria-selected="true"] [data-baseweb="tab"] { color: #FF6B6B; border-bottom-color: #FF6B6B; }
        .stMetric { background-color: #111823; padding: 15px; border-radius: 8px; }
        
        input { background-color: #1F2937 !important; color: #FFFFFF !important; border: 1px solid #374151 !important; }
        select { background-color: #1F2937 !important; color: #FFFFFF !important; border: 1px solid #374151 !important; }
    </style>
    """, unsafe_allow_html=True)

def mostrar_kpi_grande(titulo, valor, cambio_pct=None, color='#4D96FF', icon='📊'):
    """Muestra un KPI grande y atractivo"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20 0%, {color}05 100%); padding: 20px; border-radius: 10px; border-left: 4px solid {color};">
            <p style="color: #888; margin: 0; font-size: 14px;">{titulo}</p>
            <h2 style="color: {color}; margin: 10px 0; font-size: 32px;">{valor}</h2>
            {f'<p style="color: #6BCB77; margin: 0; font-size: 12px;">↑ {cambio_pct}% vs mes anterior</p>' if cambio_pct else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div style="font-size: 48px; text-align: center; padding: 10px;">{icon}</div>', unsafe_allow_html=True)

def mostrar_tarjeta_prediccion(probabilidad, categoria, recomendacion, color):
    """Tarjeta de resultado de predicción"""
    porcentaje = f"{probabilidad:.1%}"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}20 0%, {color}05 100%); border: 2px solid {color}; border-radius: 12px; padding: 30px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: #888; margin: 0; font-size: 12px; text-transform: uppercase;">Probabilidad de Cierre</p>
                <h1 style="color: {color}; margin: 10px 0; font-size: 48px;">{porcentaje}</h1>
                <p style="color: #AAA; margin: 0; font-size: 14px;">Categoría: <b>{categoria}</b></p>
            </div>
            <div style="font-size: 64px; opacity: 0.3;">
                {'🔥' if probabilidad > 0.70 else '⭐' if probabilidad > 0.55 else '📊' if probabilidad > 0.40 else '⚠️' if probabilidad > 0.25 else '❌'}
            </div>
        </div>
        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid {color}40;">
            <p style="color: #FFF; margin: 0; font-size: 14px;"><b>Recomendación:</b></p>
            <p style="color: #CCC; margin: 5px 0; font-size: 14px;">{recomendacion}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_crosssell(categoria, tasa_cierre, ventas_totales):
    """Tarjeta de oportunidad de cross-sell"""
    emoji_map = {
        'Válvulas y Actuadores': '🔧',
        'Sistemas de Control': '⚙️',
        'Equipos de Laboratorio e Instrumentación': '🔬',
        'Producto General': '📦'
    }
    
    emoji = emoji_map.get(categoria, '📦')
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #95E1D320 0%, #95E1D305 100%); border-left: 4px solid #95E1D3; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: #888; margin: 0; font-size: 12px;">{emoji} {categoria}</p>
                <h3 style="color: #95E1D3; margin: 5px 0; font-size: 18px;">{tasa_cierre:.0%} de cierre</h3>
                <p style="color: #666; margin: 0; font-size: 11px;">{int(ventas_totales)} ventas realizadas</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def crear_barra_progreso_prob(prob):
    """Crea una barra de progreso visual para probabilidad"""
    
    if prob > 0.70:
        color = '#FF6B6B'
        texto = '🔥 Muy Alta'
    elif prob > 0.55:
        color = '#FFD93D'
        texto = '⭐ Alta'
    elif prob > 0.40:
        color = '#6BCB77'
        texto = '📊 Media'
    elif prob > 0.25:
        color = '#4D96FF'
        texto = '⚠️  Baja'
    else:
        color = '#FF6B9D'
        texto = '❌ Muy Baja'
    
    st.markdown(f"""
    <div style="background: #111823; border-radius: 8px; padding: 10px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #CCC; font-size: 12px;">Probabilidad de Cierre</span>
            <span style="color: {color}; font-weight: bold;">{texto}</span>
        </div>
        <div style="background: #0E1117; border-radius: 4px; height: 20px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, {color}40, {color}); width: {prob*100}%; height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: flex-end; padding-right: 5px;">
                <span style="color: #FFF; font-size: 11px; font-weight: bold;">{prob:.0%}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def grafico_barras_horizontal(df, x, y, titulo, color='#FF6B6B'):
    """Gráfico de barras horizontal atractivo"""
    fig = px.bar(df, y=y, x=x, orientation='h', title=titulo, color=x, color_continuous_scale=['#0E1117', color])
    fig.update_layout(**TEMA_OSCURO, height=400, showlegend=False, title_font_size=18, title_font_color='#FFFFFF', hovermode='closest')
    fig.update_traces(marker_line_width=0)
    return fig

def grafico_pie_categorias(df, categoria_col, titulo, colores=None):
    """Gráfico de pie con colores personalizados"""
    data = df[categoria_col].value_counts().reset_index()
    data.columns = [categoria_col, 'count']
    
    if colores is None:
        colores = list(COLORES_VIBRANTES.values())[:len(data)]
    
    fig = go.Figure(data=[go.Pie(labels=data[categoria_col], values=data['count'], 
        marker=dict(colors=colores, line=dict(color='#0E1117', width=2)), textposition='inside', textinfo='label+percent')])
    
    fig.update_layout(**TEMA_OSCURO, height=400, title_font_size=18, title_text=titulo, title_font_color='#FFFFFF')
    return fig
