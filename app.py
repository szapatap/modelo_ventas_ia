import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import plotly.express as px
from datetime import datetime

# Importamos nuestro módulo de arquitectura
import model_architecture as ma

# Configuración inicial
st.set_page_config(page_title="AI Sales Predictor", layout="wide", page_icon="📈")

# ==========================================
# GESTIÓN DE BD Y USUARIOS
# ==========================================

def init_db():
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, client_name TEXT, 
                  zone TEXT, prob REAL, label TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
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

def save_prediction_db(user, client, zone, prob, label):
    conn = sqlite3.connect('sales_app.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions(user_id, client_name, zone, prob, label) VALUES (?,?,?,?,?)',
              (user, client, zone, prob, label))
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
        # Ventas por Vendedor (Top 10)
        if 'Vendedor' in df.columns:
            top_vend = df[df['¿Adjudicado?']==1]['Vendedor'].value_counts().head(10).reset_index()
            top_vend.columns = ['Vendedor', 'Ventas']
            fig_v = px.pie(top_vend, names='Vendedor', values='Ventas', title="Top 10 Vendedores (Market Share)")
            st.plotly_chart(fig_v, use_container_width=True)

    # 3. Línea Temporal
    if 'Fecha' in df.columns:
        temporal = df.groupby(df['Fecha'].dt.to_period('M'))['¿Adjudicado?'].mean().reset_index()
        temporal['Fecha'] = temporal['Fecha'].astype(str)
        fig_t = px.area(temporal, x='Fecha', y='¿Adjudicado?', title="Tendencia Histórica de Adjudicación",
                        line_shape='spline')
        st.plotly_chart(fig_t, use_container_width=True)

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
        
    st.title("🚀 Sistema Predictivo de Ventas")
    
    tabs = st.tabs(["📂 Datos & Modelo", "📊 Análisis (EDA)", "🔮 Predicciones", "📝 Historial"])

    # 1. PESTAÑA DATOS Y ENTRENAMIENTO
    with tabs[0]:
        st.subheader("Entrenamiento del Modelo")
        file = st.file_uploader("Cargar Dataset Histórico", type=['xlsx', 'csv'])
        
        if file:
            df, err = ma.load_data(file)
            if df is not None:
                st.dataframe(df.head(3))
                if st.button("Entrenar Modelo con estos datos", type="primary"):
                    with st.spinner("Entrenando..."):
                        model, metrics, artifacts = ma.train_model_logic(df)
                        ma.save_model_artifacts(model, artifacts)
                    st.success(f"Modelo actualizado. Accuracy: {metrics['accuracy']:.2%}")
            else:
                st.error(f"Error: {err}")

    # 2. PESTAÑA EDA
    with tabs[1]:
        # Cargar datos solo para visualizar
        eda_file = st.file_uploader("Cargar datos para visualizar", type=['xlsx', 'csv'], key='eda')
        if eda_file:
            df_eda, _ = ma.load_data(eda_file)
            if df_eda is not None:
                show_eda(df_eda)
        else:
            st.info("Sube un archivo para ver las gráficas.")

    # 3. PESTAÑA PREDICCIÓN
    with tabs[2]:
        st.subheader("Simulador de Negocios")
        
        model, artifacts = ma.load_model_artifacts()
        
        if model is None:
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 1.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                tipo = st.radio("Estado del Cliente", ["Existente", "Nuevo Prospecto"])
                if tipo == "Existente":
                    cli = st.selectbox("Buscar Cliente", artifacts['unique_clients'])
                    is_new = False
                else:
                    cli = st.text_input("Nombre del Cliente Nuevo")
                    is_new = True
            
            with c2:
                # Solo permite seleccionar Zona como se pidió
                zona = st.selectbox("Zona Geográfica", artifacts['unique_zones'])
            
            if st.button("Predecir Adjudicación", type="primary"):
                if not cli:
                    st.error("Falta el nombre del cliente")
                else:
                    prob, label = ma.make_prediction(model, artifacts, cli, zona, is_new)
                    
                    st.markdown("---")
                    col_res, col_gauge = st.columns([2,1])
                    
                    with col_res:
                        st.write(f"### Resultado: {label}")
                        if prob > 0.5:
                            st.success(f"Probabilidad estimada: **{prob:.1%}**")
                        else:
                            st.error(f"Probabilidad estimada: **{prob:.1%}**")
                        
                    save_prediction_db(st.session_state['user'], cli, zona, prob, label)
                    st.toast("Predicción guardada")

    # 4. PESTAÑA HISTORIAL
    with tabs[3]:
        st.subheader("Registro de Actividad")
        orden = st.selectbox("Ordenar por", ["Más recientes (DESC)", "Más antiguos (ASC)"])
        sql_order = "DESC" if "DESC" in orden else "ASC"
        
        df_hist = get_history(sql_order)
        st.dataframe(df_hist, use_container_width=True)

if __name__ == '__main__':
    main()