"""
APLICACIÓN STREAMLIT - Sales Intelligence Platform v3
Frontend profesional con Sidebar y tema oscuro
Importa toda la lógica de backend.py
"""

import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime, timedelta
import backend as be

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================

st.set_page_config(
    page_title="Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

be.aplicar_tema_oscuro()

# ==========================================
# GESTIÓN DE BD Y USUARIOS
# ==========================================

def init_db():
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
    (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, client_name TEXT,
    zone TEXT, categoria TEXT, prob REAL, label TEXT, recomendacion TEXT, 
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def user_auth(username, password, mode='login'):
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    pwd_hash = hashlib.sha256(str.encode(password)).hexdigest()
    
    if mode == 'login':
        c.execute('SELECT * FROM users WHERE username =? AND password = ?', (username, pwd_hash))
        return c.fetchall()
    elif mode == 'signup':
        try:
            c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, pwd_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    conn.close()

def save_prediction_db(user, client, zone, categoria, prob, label, recomendacion):
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions
    (user_id, client_name, zone, categoria, prob, label, recomendacion) 
    VALUES (?,?,?,?,?,?,?)''',
    (user, client, zone, categoria, prob, label, recomendacion))
    conn.commit()
    conn.close()

def get_history(user_id=None, days=30):
    conn = sqlite3.connect('sales_app.db')
    
    if user_id:
        df = pd.read_sql_query(
            f"SELECT * FROM predictions WHERE user_id = '{user_id}' AND timestamp > datetime('now', '-{days} days') ORDER BY timestamp DESC",
            conn
        )
    else:
        df = pd.read_sql_query(
            f"SELECT * FROM predictions WHERE timestamp > datetime('now', '-{days} days') ORDER BY timestamp DESC",
            conn
        )
    
    conn.close()
    return df

# ==========================================
# FUNCIONES DE ANÁLISIS
# ==========================================

def mostrar_dashboard_general(df):
    """Dashboard general con KPIs y gráficos principales"""
    
    st.markdown("### 📊 Dashboard de Inteligencia de Ventas")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_cotizaciones = len(df)
    ventas_ganadas = len(df[df['¿Adjudicado?'] == 1])
    tasa_conversion = df['¿Adjudicado?'].mean()
    
    with col1:
        be.mostrar_kpi_grande("Total de Cotizaciones", f"{total_cotizaciones:,}", color='#4D96FF', icon='📋')
    
    with col2:
        be.mostrar_kpi_grande("Ventas Ganadas", f"{ventas_ganadas:,}", color='#6BCB77', icon='✅')
    
    with col3:
        be.mostrar_kpi_grande("Tasa de Conversión", f"{tasa_conversion:.1%}", color='#FFD93D', icon='📈')
    
    with col4:
        unique_clientes = df['Cliente'].nunique()
        be.mostrar_kpi_grande("Clientes Únicos", f"{unique_clientes:,}", color='#FF6B6B', icon='👥')
    
    st.divider()
    
    # Gráficos principales
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Tasa por categoría
        stats_categoria = df.groupby('Categoria_Producto')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_categoria.columns = ['Categoria', 'Total', 'Ganadas', 'Tasa']
        
        colores_categorias = {
            'Válvulas y Actuadores': '#FF6B6B',
            'Sistemas de Control': '#4ECDC4',
            'Equipos de Laboratorio e Instrumentación': '#95E1D3',
            'Producto General': '#A0A0A0'
        }
        
        fig_cat = be.grafico_barras_horizontal(
            stats_categoria.sort_values('Tasa', ascending=True),
            'Tasa', 'Categoria',
            '📊 Tasa de Conversión por Categoría de Producto',
            '#FF6B6B'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_right:
        # Desempeño por zona
        stats_zona = df.groupby('Zona Geográfica')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_zona.columns = ['Zona', 'Total', 'Ganadas', 'Tasa']
        
        colores_zona = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#6BCB77']
        fig_zona = be.grafico_pie_categorias(stats_zona, 'Zona', '🌍 Distribución por Zona', colores_zona)
        st.plotly_chart(fig_zona, use_container_width=True)
    
    st.divider()
    
    # Top vendedores
    col_left, col_middle, col_right = st.columns(3)
    
    with col_left:
        st.markdown("### 🏆 Top 5 Vendedores")
        top_vendedores = df[df['¿Adjudicado?'] == 1]['Usuario_Interno'].value_counts().head(5)
        for i, (vendedor, cant) in enumerate(top_vendedores.items(), 1):
            st.markdown(f'<div style="margin: 8px 0;"><span style="color: #4D96FF; font-weight: bold;">{i}.</span> <span style="color: #FFF;">{vendedor}: <b style="color: #6BCB77;">{cant}</b></span></div>', unsafe_allow_html=True)
    
    with col_middle:
        st.markdown("### ⭐ Mejores Zonas")
        best_zonas = stats_zona.nlargest(5, 'Tasa')
        for i, row in best_zonas.iterrows():
            st.markdown(f'<div style="margin: 8px 0;"><span style="color: #FFD93D; font-weight: bold;">📍</span> <span style="color: #FFF;">{row["Zona"]}: <b style="color: #FF6B6B;">{row["Tasa"]:.0%}</b></span></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown("### 📦 Categorías Exitosas")
        best_cats = stats_categoria.nlargest(5, 'Tasa')
        for i, row in best_cats.iterrows():
            st.markdown(f'<div style="margin: 8px 0;"><span style="color: #6BCB77; font-weight: bold;">✓</span> <span style="color: #FFF;">{row["Categoria"][:25]}...: <b style="color: #FFD93D;">{row["Tasa"]:.0%}</b></span></div>', unsafe_allow_html=True)

def mostrar_analisis_usuarios(df):
    """Análisis de desempeño por usuario"""
    
    st.markdown("### 👤 Análisis de Vendedores")
    
    usuarios = df['Usuario_Interno'].unique()
    usuario_selected = st.selectbox("Selecciona un vendedor:", usuarios, key='usuario_select')
    
    df_usuario = df[df['Usuario_Interno'] == usuario_selected]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cotizaciones Totales", len(df_usuario))
    
    with col2:
        ventas_usuario = len(df_usuario[df_usuario['¿Adjudicado?'] == 1])
        st.metric("Ventas Ganadas", ventas_usuario)
    
    with col3:
        tasa_usuario = df_usuario['¿Adjudicado?'].mean()
        st.metric("Tasa de Cierre", f"{tasa_usuario:.1%}")

def mostrar_analisis_clientes(df):
    """Análisis de clientes y oportunidades de cross-sell"""
    
    st.markdown("### 👥 Análisis de Clientes & Cross-Sell")
    
    clientes = sorted(df['Cliente'].unique())
    cliente_selected = st.selectbox("Selecciona un cliente:", clientes, key='cliente_select')
    
    df_cliente = df[df['Cliente'] == cliente_selected]
    
    # Info del cliente
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cotizaciones", len(df_cliente))
    
    with col2:
        ventas = len(df_cliente[df_cliente['¿Adjudicado?'] == 1])
        st.metric("Ventas Ganadas", ventas)
    
    with col3:
        tasa = df_cliente['¿Adjudicado?'].mean()
        st.metric("Tasa de Cierre", f"{tasa:.1%}")
    
    with col4:
        zonas_cliente = df_cliente['Zona Geográfica'].nunique()
        st.metric("Zonas", zonas_cliente)
    
    st.divider()
    
    # Cross-sell
    st.markdown("#### 🎯 Oportunidades de Cross-Sell")
    
    categorias_compradas = df_cliente[df_cliente['¿Adjudicado?'] == 1]['Categoria_Producto'].unique()
    todas_categorias = df['Categoria_Producto'].unique()
    
    st.markdown(f"**Comprado actualmente:** {', '.join(categorias_compradas) if len(categorias_compradas) > 0 else 'Ninguna venta aún'}")
    
    categorias_oportunidad = [c for c in todas_categorias if c not in categorias_compradas]
    
    if categorias_oportunidad:
        st.markdown("**Oportunidades potenciales:**")
        for categoria in categorias_oportunidad:
            tasa_categoria = df[df['Categoria_Producto'] == categoria]['¿Adjudicado?'].mean()
            ventas_categoria = len(df[(df['Categoria_Producto'] == categoria) & (df['¿Adjudicado?'] == 1)])
            be.mostrar_tarjeta_crosssell(categoria, tasa_categoria, ventas_categoria)

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

def main():
    init_db()
    
    # ========== LOGIN SYSTEM ==========
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="color: #FF6B6B; font-size: 48px;">📊</h1>
                <h2 style="color: #FFF;">Sales Intelligence</h2>
                <p style="color: #AAA;">Predicción de Ventas con IA</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            menu_login = st.radio("", ["Iniciar Sesión", "Registrarse"], horizontal=True)
            
            user = st.text_input("👤 Usuario")
            pwd = st.text_input("🔐 Contraseña", type="password")
            
            if st.button("Continuar", use_container_width=True, type="primary"):
                if menu_login == "Iniciar Sesión":
                    if user_auth(user, pwd, 'login'):
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user
                        st.rerun()
                    else:
                        st.error("❌ Credenciales inválidas")
                else:
                    if user_auth(user, pwd, 'signup'):
                        st.success("✅ Usuario creado. Por favor, inicia sesión.")
                    else:
                        st.error("❌ El usuario ya existe")
        
        return
    
    # ========== APP DASHBOARD ==========
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #FF6B6B; margin: 0;">📊 Sales AI</h2>
            <p style="color: #888; margin: 5px 0; font-size: 12px;">Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown(f"**👤 {st.session_state['user']}**")
        
        menu = st.radio(
            "Menú Principal",
            ["📊 Dashboard", "🔮 Predicción", "👥 Clientes", "👤 Vendedores", 
             "📈 Análisis", "📝 Historial", "⚙️ Entrenamiento"],
            key='menu_principal'
        )
        
        st.divider()
        
        if st.button("🚪 Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # ========== CONTENIDO PRINCIPAL ==========
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="color: #FFF; margin: 0;">📊 Sistema Predictivo de Ventas v3</h1>
        <p style="color: #888; margin: 0; font-size: 12px;">Con Clasificación Automática</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar dataset si no existe
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if st.session_state['df'] is None and menu != "⚙️ Entrenamiento":
        st.warning("⚠️ Primero debes entrenar el modelo. Ve a la sección '⚙️ Entrenamiento'")
    
    # Menú de navegación
    if menu == "📊 Dashboard":
        if st.session_state['df'] is not None:
            mostrar_dashboard_general(st.session_state['df'])
        else:
            st.info("Carga datos para ver el dashboard")
    
    elif menu == "🔮 Predicción":
        if st.session_state['df'] is not None:
            st.markdown("### 🔮 Simulador de Predicción")
            
            model, artifacts = be.load_model_artifacts()
            
            if model is None:
                st.error("❌ Modelo no encontrado. Entrena primero.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    tipo_cliente = st.radio("Tipo de Cliente", ["Existente", "Nuevo"])
                    
                    if tipo_cliente == "Existente":
                        client = st.selectbox("Cliente", artifacts['unique_clients'])
                        is_new = False
                    else:
                        client = st.text_input("Nombre del Cliente")
                        is_new = True
                
                with col2:
                    zona = st.selectbox("Zona Geográfica", artifacts['unique_zones'])
                    categoria = st.selectbox("Categoría de Producto", artifacts['categorias'])
                
                usuario_sel = st.selectbox("Vendedor Asignado", artifacts['unique_usuarios'])
                
                if st.button("🚀 Predecir Adjudicación", use_container_width=True, type="primary"):
                    if client:
                        prob, label, recomendacion, color = be.make_prediction(
                            model, artifacts, client, zona, categoria, usuario_sel, is_new
                        )
                        
                        save_prediction_db(st.session_state['user'], client, zona, categoria, prob, label, recomendacion)
                        
                        be.mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color)
                        
                        st.success("✅ Predicción guardada en el historial")
        else:
            st.info("Carga datos para hacer predicciones")
    
    elif menu == "👥 Clientes":
        if st.session_state['df'] is not None:
            mostrar_analisis_clientes(st.session_state['df'])
        else:
            st.info("Carga datos para ver análisis de clientes")
    
    elif menu == "👤 Vendedores":
        if st.session_state['df'] is not None:
            mostrar_analisis_usuarios(st.session_state['df'])
        else:
            st.info("Carga datos para ver análisis de vendedores")
    
    elif menu == "📈 Análisis":
        if st.session_state['df'] is not None:
            st.markdown("### 📈 Análisis Avanzado")
            
            df_hist = get_history(days=30)
            
            if not df_hist.empty:
                st.markdown("#### Predicciones Recientes")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Predicciones", len(df_hist))
                
                with col2:
                    st.metric("Prob. Promedio", f"{df_hist['prob'].mean():.1%}")
                
                with col3:
                    alta_prob = len(df_hist[df_hist['prob'] > 0.65])
                    st.metric("Alta Probabilidad", alta_prob)
                
                st.divider()
                
                st.markdown("#### Histórico de Predicciones")
                st.dataframe(df_hist[['client_name', 'zone', 'categoria', 'prob', 'label', 'timestamp']], 
                           use_container_width=True, height=400)
            else:
                st.info("Sin predicciones aún")
        else:
            st.info("Carga datos para ver análisis")
    
    elif menu == "📝 Historial":
        st.markdown("### 📝 Historial de Predicciones")
        
        df_hist = get_history(user_id=st.session_state['user'], days=90)
        
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Sin predicciones en tu historial")
    
    elif menu == "⚙️ Entrenamiento":
        st.markdown("### ⚙️ Entrenar Modelo")
        
        file = st.file_uploader("📤 Sube tu dataset (Excel o CSV)", type=['xlsx', 'csv'])
        
        if file:
            with st.spinner("Cargando datos..."):
                df, err = be.load_data(file)
            
            if df is not None:
                st.success(f"✅ Dataset cargado: {len(df):,} registros")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Filas", f"{len(df):,}")
                
                with col2:
                    st.metric("Clientes", df['Cliente'].nunique())
                
                with col3:
                    st.metric("Zonas", df['Zona Geográfica'].nunique())
                
                with col4:
                    st.metric("Vendedores", df['Usuario_Interno'].nunique())
                
                st.divider()
                
                if st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary"):
                    with st.spinner("Entrenando modelo... esto puede tomar un momento"):
                        model, metrics, artifacts = be.train_model_logic(df)
                        be.save_model_artifacts(model, artifacts)
                        st.session_state['df'] = df
                    
                    st.success("✅ ¡Modelo entrenado exitosamente!")
                    st.metric("Accuracy del Modelo", f"{metrics['accuracy']:.2%}")
                    
                    st.divider()
                    st.markdown("#### 📊 Distribución por Categoría")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for i, cat in enumerate(artifacts['categorias']):
                        stats = next((s for s in artifacts['stats_categoria'] if s['Categoria'] == cat), None)
                        if stats:
                            with st.columns(4)[i]:
                                st.metric(
                                    f"{cat[:20]}...",
                                    f"{stats['Tasa_Conversion']:.0%}",
                                    f"{int(stats['Ganadas'])} ganadas"
                                )
            else:
                st.error(f"❌ Error: {err}")

if __name__ == '__main__':
    main()
