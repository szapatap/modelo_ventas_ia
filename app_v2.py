import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import plotly.express as px
from datetime import datetime

# Importamos nuestro módulo de arquitectura mejorado
import model_architecture_v2 as ma

# Configuración inicial
st.set_page_config(page_title="AI Sales Predictor v2", layout="wide", page_icon="📈")

# ==========================================
# GESTIÓN DE BD Y USUARIOS
# ==========================================

def init_db():
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
    (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, client_name TEXT,
    zone TEXT, solicitud TEXT, prob REAL, label TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
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

def save_prediction_db(user, client, zone, solicitud, prob, label):
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions(user_id, client_name, zone, solicitud, prob, label) VALUES (?,?,?,?,?,?)',
    (user, client, zone, solicitud, prob, label))
    conn.commit()
    conn.close()

def get_history(order='DESC'):
    conn = sqlite3.connect('sales_app.db')
    df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY timestamp {order}", conn)
    conn.close()
    return df

# ==========================================
# VISUALIZACIÓN (EDA)
# ==========================================

def show_eda(df):
    st.markdown("### 🔍 Análisis de Inteligencia de Negocios")
    
    # 1. KPIs
    total_ventas = len(df[df['¿Adjudicado?']==1])
    tasa_global = df['¿Adjudicado?'].mean()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cotizaciones", len(df))
    c2.metric("Ventas Ganadas", total_ventas)
    c3.metric("Tasa de Conversión Global", f"{tasa_global:.1%}")
    
    st.divider()
    
    # 2. Gráficos Principales
    col1, col2 = st.columns(2)
    
    with col1:
        # Probabilidad por Zona
        prob_zona = df.groupby('Zona Geográfica')['¿Adjudicado?'].mean().reset_index()
        fig_z = px.bar(prob_zona, x='Zona Geográfica', y='¿Adjudicado?', color='¿Adjudicado?',
        title="Probabilidad de Éxito por Zona", color_continuous_scale='Viridis')
        st.plotly_chart(fig_z, use_container_width=True)
    
    with col2:
        # Ventas por Solicitud (Top 10)
        if 'Solicitud' in df.columns:
            top_solicitud = df[df['¿Adjudicado?']==1]['Solicitud'].value_counts().head(10).reset_index()
            top_solicitud.columns = ['Solicitud', 'Ventas']
            fig_s = px.pie(top_solicitud, names='Solicitud', values='Ventas', title="Top 10 Solicitudes (Market Share)")
            st.plotly_chart(fig_s, use_container_width=True)
    
    # 3. Línea Temporal
    if 'Fecha' in df.columns:
        temporal = df.groupby(df['Fecha'].dt.to_period('M'))['¿Adjudicado?'].mean().reset_index()
        temporal['Fecha'] = temporal['Fecha'].astype(str)
        fig_t = px.area(temporal, x='Fecha', y='¿Adjudicado?', title="Tendencia Histórica de Adjudicación",
        line_shape='spline')
        st.plotly_chart(fig_t, use_container_width=True)
    
    # 4. Análisis por Solicitud
    st.markdown("#### 📊 Tasa de Conversión por Tipo de Solicitud")
    if 'Solicitud' in df.columns:
        solicitud_stats = df.groupby('Solicitud').agg({
            '¿Adjudicado?': ['count', 'sum', 'mean']
        }).reset_index()
        solicitud_stats.columns = ['Solicitud', 'Total', 'Ganadas', 'Tasa Conversión']
        solicitud_stats = solicitud_stats.sort_values('Tasa Conversión', ascending=False)
        st.dataframe(solicitud_stats, use_container_width=True)

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

def main():
    init_db()
    
    # --- LOGIN SYSTEM ---
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=100)
        st.sidebar.title("Acceso")
        menu_login = st.sidebar.selectbox("Opción", ["Iniciar Sesión", "Registrarse"])
        
        user = st.sidebar.text_input("Usuario")
        pwd = st.sidebar.text_input("Contraseña", type="password")
        
        if st.sidebar.button("Entrar/Registrar"):
            if menu_login == "Iniciar Sesión":
                if user_auth(user, pwd, 'login'):
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = user
                    st.rerun()
                else:
                    st.sidebar.error("Credenciales inválidas")
            else:
                if user_auth(user, pwd, 'signup'):
                    st.sidebar.success("Usuario creado. Logueate.")
                else:
                    st.sidebar.error("El usuario ya existe")
        return # Detiene ejecución si no hay login
    
    # --- APP DASHBOARD ---
    st.sidebar.success(f"Hola, {st.session_state['user']}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    st.title("🚀 Sistema Predictivo de Ventas v2")
    st.markdown("**Modelo mejorado con análisis de Solicitud/Requerimiento del cliente**")
    
    tabs = st.tabs(["📂 Datos & Modelo", "📊 Análisis (EDA)", "🔮 Predicciones", "📝 Historial"])
    
    # 1. PESTAÑA DATOS Y ENTRENAMIENTO
    with tabs[0]:
        st.subheader("Entrenamiento del Modelo Mejorado")
        st.info("✨ Este modelo ahora considera: Zona Geográfica + Historial Cliente + Tipo de Solicitud")
        
        file = st.file_uploader("Cargar Dataset Histórico", type=['xlsx', 'csv'])
        
        if file:
            df, err = ma.load_data(file)
            if df is not None:
                st.write("**Vista previa del dataset:**")
                st.dataframe(df.head(5))
                
                st.write(f"**Resumen:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Filas", len(df))
                col2.metric("Zonas únicas", df['Zona Geográfica'].nunique())
                col3.metric("Solicitudes únicas", df['Solicitud'].nunique() if 'Solicitud' in df.columns else 0)
                
                if st.button("Entrenar Modelo con estos datos", type="primary"):
                    with st.spinner("Entrenando... esto puede tomar un momento"):
                        model, metrics, artifacts = ma.train_model_logic(df)
                        ma.save_model_artifacts(model, artifacts)
                        
                        st.success(f"✅ Modelo actualizado!")
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        
                        # Mostrar reporte de clasificación
                        st.write("**Reporte de Clasificación:**")
                        report_df = pd.DataFrame(metrics['report']).T
                        st.dataframe(report_df, use_container_width=True)
            else:
                st.error(f"Error: {err}")
    
    # 2. PESTAÑA EDA
    with tabs[1]:
        st.subheader("📊 Análisis Exploratorio de Datos")
        eda_file = st.file_uploader("Cargar datos para visualizar", type=['xlsx', 'csv'], key='eda')
        
        if eda_file:
            df_eda, _ = ma.load_data(eda_file)
            if df_eda is not None:
                show_eda(df_eda)
        else:
            st.info("Sube un archivo para ver las gráficas.")
    
    # 3. PESTAÑA PREDICCIÓN
    with tabs[2]:
        st.subheader("🔮 Simulador de Negocios")
        
        model, artifacts = ma.load_model_artifacts()
        
        if model is None:
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Datos & Modelo'.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                tipo = st.radio("Estado del Cliente", ["Existente", "Nuevo Prospecto"])
                
                if tipo == "Existente":
                    cli = st.selectbox("Buscar Cliente", artifacts['unique_clients'])
                    is_new = False
                else:
                    cli = st.text_input("Nombre del Cliente Nuevo")
                    is_new = True
            
            with col2:
                zona = st.selectbox("Zona Geográfica", artifacts['unique_zones'])
                solicitud = st.selectbox(
                    "Tipo de Solicitud/Requerimiento",
                    artifacts['unique_solicitudes'],
                    help="Selecciona el tipo de requerimiento técnico del cliente"
                )
            
            if st.button("Predecir Adjudicación", type="primary", use_container_width=True):
                if not cli:
                    st.error("Falta el nombre del cliente")
                else:
                    prob, label = ma.make_prediction(model, artifacts, cli, zona, solicitud, is_new)
                    
                    st.markdown("---")
                    col_res, col_gauge = st.columns([2, 1])
                    
                    with col_res:
                        st.write(f"### Resultado: {label}")
                        
                        # Mostrar probabilidad con color
                        if prob > 0.65:
                            st.success(f"Probabilidad estimada: **{prob:.1%}**")
                        elif prob > 0.35:
                            st.warning(f"Probabilidad estimada: **{prob:.1%}**")
                        else:
                            st.error(f"Probabilidad estimada: **{prob:.1%}**")
                        
                        # Detalles de la predicción
                        st.markdown("#### 📋 Detalles de la Predicción")
                        det_col1, det_col2, det_col3 = st.columns(3)
                        det_col1.info(f"**Cliente:** {cli}\n\n{'(Nuevo)' if is_new else '(Histórico)'}")
                        det_col2.info(f"**Zona:** {zona}")
                        det_col3.info(f"**Solicitud:** {solicitud}")
                    
                    # Guardar predicción
                    save_prediction_db(st.session_state['user'], cli, zona, solicitud, prob, label)
                    st.toast("✅ Predicción guardada en base de datos")
    
    # 4. PESTAÑA HISTORIAL
    with tabs[3]:
        st.subheader("📝 Registro de Actividad")
        
        col_order, col_filter = st.columns([2, 2])
        
        with col_order:
            orden = st.selectbox("Ordenar por", ["Más recientes (DESC)", "Más antiguos (ASC)"])
        
        with col_filter:
            sql_order = "DESC" if "DESC" in orden else "ASC"
        
        df_hist = get_history(sql_order)
        
        if not df_hist.empty:
            # Mostrar estadísticas del historial
            st.markdown("#### 📊 Resumen de Predicciones")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            stat_col1.metric("Total de Predicciones", len(df_hist))
            stat_col2.metric("Prob. Promedio", f"{df_hist['prob'].mean():.1%}")
            
            # Contar predicciones por categoría
            alta = len(df_hist[df_hist['label'].str.contains('Alta')])
            stat_col3.metric("Predicciones con Alta Prob.", alta)
            
            st.markdown("---")
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("No hay predicciones registradas aún.")

if __name__ == '__main__':
    main()
