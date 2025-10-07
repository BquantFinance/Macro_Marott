import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data as web
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="FRED Análisis Económico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #06b6d4;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem !important;
    }
    
    h2 {
        color: #06b6d4;
        font-weight: 700;
        border-bottom: 2px solid #06b6d4;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #38bdf8;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        color: #94a3b8;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    }
    
    .collaboration-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 50%, #1e293b 100%);
        border-top: 2px solid #06b6d4;
        padding: 15px 20px;
        text-align: center;
        z-index: 999;
        box-shadow: 0 -4px 20px rgba(6, 182, 212, 0.2);
    }
    
    .collaboration-footer p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 5px 0;
    }
    
    .collaboration-footer a {
        color: #06b6d4;
        text-decoration: none;
        font-weight: 600;
        margin: 0 15px;
    }
    
    .collaboration-footer a:hover {
        color: #38bdf8;
        text-decoration: underline;
    }
    
    .collaboration-footer .collab-text {
        font-size: 1rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Logo styling */
    img {
        border-radius: 10px;
        background: white;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_series(series_dict, start_date, end_date):
    """Obtener múltiples series de FRED y combinarlas"""
    dfs = {}
    errors = []
    
    for name, series_id in series_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            if df is not None and not df.empty:
                dfs[name] = df.iloc[:, 0]
        except Exception as e:
            errors.append(f"{name}")
    
    if errors and len(errors) < len(series_dict):
        st.info(f"Algunas series no disponibles: {', '.join(errors[:5])}")
    
    if dfs:
        combined = pd.DataFrame(dfs)
        combined = combined.dropna(how='all')
        return combined
    return None

def safe_value(value, format_str="{:.2f}"):
    """Formatear valor de forma segura"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

def create_line_chart(df, title, y_title, colors=None):
    """Crear gráfico de líneas"""
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    if colors is None:
        colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6']
    
    for i, col in enumerate(df.columns):
        if df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col, mode='lines',
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate='<b>%{fullData.name}</b><br>%{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Fecha", yaxis_title=y_title, hovermode='x unified',
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
        height=500
    )
    return fig

def create_area_chart(df, title, y_title, color='#06b6d4'):
    """Crear gráfico de área"""
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    for col in df.columns:
        if df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col, mode='lines',
                fill='tozeroy', line=dict(width=2, color=color),
                fillcolor='rgba(6, 182, 212, 0.3)'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Fecha", yaxis_title=y_title,
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        height=500
    )
    return fig

def create_dual_axis_chart(df, col1, col2, title, y1_title, y2_title):
    """Crear gráfico de doble eje"""
    if df is None or df.empty or col1 not in df.columns or col2 not in df.columns:
        return create_empty_chart(title)
    
    valid_data = df[[col1, col2]].dropna()
    if valid_data.empty:
        return create_empty_chart(title)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=valid_data.index, y=valid_data[col1], name=col1,
                  line=dict(color='#06b6d4', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=valid_data.index, y=valid_data[col2], name=col2,
                  line=dict(color='#ec4899', width=3)),
        secondary_y=True
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        hovermode='x unified', plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        height=500,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1)
    )
    
    fig.update_yaxes(title_text=y1_title, secondary_y=False, showgrid=True, gridcolor='#334155')
    fig.update_yaxes(title_text=y2_title, secondary_y=True, showgrid=False)
    
    return fig

def create_empty_chart(title):
    """Crear gráfico vacío"""
    fig = go.Figure()
    fig.add_annotation(
        text="No hay datos disponibles", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color='#94a3b8')
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        height=500, xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig

def main():
    # Header con logo
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        try:
            st.image("marot.avif", width=120)
        except:
            pass
    
    with col2:
        st.markdown("<h1>📊 Análisis Económico FRED</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Plataforma de Visualización de Datos Económicos de la Reserva Federal</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        date_range = st.date_input(
            "Seleccionar Rango de Fechas",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        st.markdown("---")
        st.markdown("### 📌 Fuentes de Datos")
        st.markdown("• Federal Reserve Economic Data (FRED)")
        st.markdown("• Departamento del Tesoro de EE.UU.")
        st.markdown("• Oficina de Estadísticas Laborales")
        st.markdown("• S&P Dow Jones Indices")
        
        st.markdown("---")
        st.markdown("### 🔄 Última Actualización")
        st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Política Monetaria",
        "📈 Inflación", 
        "👥 Mercado Laboral",
        "🏠 Sector Inmobiliario",
        "📊 Curvas de Rendimiento",
        "💼 Economía & Consumo"
    ])
    
    # TAB 1: POLÍTICA MONETARIA
    with tab1:
        st.markdown("## 💰 Panel de Política Monetaria")
        
        monetary_series = {
            'FEDFUNDS': 'FEDFUNDS',
            'DGS2': 'DGS2',
            'DGS10': 'DGS10',
            'DGS30': 'DGS30',
            'T10Y2Y': 'T10Y2Y',
            'T10Y3M': 'T10Y3M',
            'T5YIE': 'T5YIE',
            'T5YIFR': 'T5YIFR'
        }
        
        with st.spinner('Cargando datos...'):
            data = fetch_multiple_series(monetary_series, start_date, end_date)
        
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            prev = data.iloc[-30] if len(data) > 30 else data.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                val = latest.get('FEDFUNDS')
                prev_val = prev.get('FEDFUNDS')
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Fed Funds", safe_value(val, "{:.2f}%"), f"{change:+.2f}%" if change != 0 else "")
            
            with col2:
                val = latest.get('DGS10')
                prev_val = prev.get('DGS10')
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Tesoro 10A", safe_value(val, "{:.2f}%"), f"{change:+.2f}%" if change != 0 else "")
            
            with col3:
                val = latest.get('T10Y2Y')
                st.metric("Spread 10A-2A", safe_value(val, "{:.2f}%"), 
                         "Invertida" if (val and val < 0) else "Normal")
            
            with col4:
                val = latest.get('T5YIE')
                st.metric("Inflación Esperada 5A", safe_value(val, "{:.2f}%"))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['FEDFUNDS', 'DGS2', 'DGS10', 'DGS30']].dropna(how='all'),
                    "📊 Tasas de Interés", "Tasa (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                spread_data = pd.DataFrame({
                    'T10Y2Y': data.get('T10Y2Y', pd.Series()),
                    'T10Y3M': data.get('T10Y3M', pd.Series())
                }).dropna(how='all')
                fig = create_line_chart(spread_data, "📉 Spreads de Rendimiento", "Spread (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Expectativas de inflación
            st.markdown("### Expectativas de Inflación")
            inflation_exp = data[['T5YIE', 'T5YIFR']].dropna(how='all')
            fig = create_line_chart(inflation_exp, "🔮 Expectativas de Inflación", "Tasa (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: INFLACIÓN
    with tab2:
        st.markdown("## 📈 Panel de Inflación")
        
        inflation_series = {
            'CPIAUCSL': 'CPIAUCSL',
            'CPILFESL': 'CPILFESL',
            'CORESTICKM159SFRBATL': 'CORESTICKM159SFRBATL',
            'PCEPI': 'PCEPI',
            'PCEPILFE': 'PCEPILFE',
            'PCETRIM12M159SFRBDAL': 'PCETRIM12M159SFRBDAL'
        }
        
        with st.spinner('Cargando datos de inflación...'):
            data = fetch_multiple_series(inflation_series, start_date, end_date)
        
        if data is not None and not data.empty:
            yoy = data.pct_change(12) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            latest_yoy = yoy.iloc[-1]
            
            with col1:
                val = latest_yoy.get('CPIAUCSL')
                st.metric("IPC YoY", safe_value(val, "{:.2f}%"))
            
            with col2:
                val = latest_yoy.get('CPILFESL')
                st.metric("IPC Subyacente YoY", safe_value(val, "{:.2f}%"))
            
            with col3:
                val = latest_yoy.get('PCEPI')
                st.metric("PCE YoY", safe_value(val, "{:.2f}%"))
            
            with col4:
                val = latest_yoy.get('PCEPILFE')
                st.metric("PCE Subyacente YoY", safe_value(val, "{:.2f}%"))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    yoy[['CPIAUCSL', 'CPILFESL', 'CORESTICKM159SFRBATL']].dropna(how='all'),
                    "📊 Inflación IPC (YoY)", "Cambio YoY (%)",
                    colors=['#f59e0b', '#ef4444', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    yoy[['PCEPI', 'PCEPILFE', 'PCETRIM12M159SFRBDAL']].dropna(how='all'),
                    "📈 Inflación PCE (YoY)", "Cambio YoY (%)",
                    colors=['#8b5cf6', '#a78bfa', '#c4b5fd']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: MERCADO LABORAL
    with tab3:
        st.markdown("## 👥 Panel del Mercado Laboral")
        
        labor_series = {
            'UNRATE': 'UNRATE',
            'PAYEMS': 'PAYEMS',
            'CES0500000003': 'CES0500000003',
            'CIVPART': 'CIVPART',
            'ICSA': 'ICSA',
            'JTSJOL': 'JTSJOL',
            'JTSQUL': 'JTSQUL',
            'JTSLDL': 'JTSLDL'
        }
        
        with st.spinner('Cargando datos laborales...'):
            data = fetch_multiple_series(labor_series, start_date, end_date)
        
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else data.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                val = latest.get('UNRATE')
                prev_val = prev.get('UNRATE')
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Desempleo", safe_value(val, "{:.1f}%"), 
                         f"{change:+.1f}%" if change != 0 else "", delta_color="inverse")
            
            with col2:
                val = latest.get('PAYEMS')
                prev_val = prev.get('PAYEMS')
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Nóminas", safe_value(val, "{:.0f}K"), f"{change:+.0f}K" if change != 0 else "")
            
            with col3:
                val = latest.get('CES0500000003')
                st.metric("Salario por Hora", safe_value(val, "${:.2f}"))
            
            with col4:
                val = latest.get('JTSJOL')
                st.metric("Vacantes (JOLTS)", safe_value(val, "{:.0f}K"))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_area_chart(
                    data[['UNRATE']].dropna(),
                    "📉 Tasa de Desempleo", "Tasa (%)", color='#8b5cf6'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['PAYEMS', 'JTSJOL']].dropna(how='all'),
                    "📊 Nóminas y Vacantes", "Miles",
                    colors=['#10b981', '#06b6d4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # JOLTS Data
            st.markdown("### Datos JOLTS (Job Openings and Labor Turnover Survey)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['JTSQUL']].dropna(),
                    "🚪 Renuncias JOLTS", "Miles", colors=['#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['JTSLDL']].dropna(),
                    "📉 Despidos JOLTS", "Miles", colors=['#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: SECTOR INMOBILIARIO
    with tab4:
        st.markdown("## 🏠 Panel del Sector Inmobiliario")
        
        re_series = {
            'CSUSHPISA': 'CSUSHPISA',
            'USSTHPI': 'USSTHPI',
            'MSPNHSUS': 'MSPNHSUS',
            'HOUST': 'HOUST',
            'HOUST1F': 'HOUST1F',
            'PERMIT': 'PERMIT',
            'PERMIT1': 'PERMIT1',
            'HSN1FNSA': 'HSN1FNSA',
            'EXHOSLUSM495S': 'EXHOSLUSM495S',
            'MORTGAGE30US': 'MORTGAGE30US',
            'PRRESCON': 'PRRESCON',
            'USCONS': 'USCONS'
        }
        
        with st.spinner('Cargando datos inmobiliarios...'):
            data = fetch_multiple_series(re_series, start_date, end_date)
        
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            prev_year = data.iloc[-12] if len(data) > 12 else data.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                val = latest.get('CSUSHPISA')
                prev_val = prev_year.get('CSUSHPISA')
                if val and prev_val and not pd.isna(val) and not pd.isna(prev_val):
                    yoy = ((val / prev_val) - 1) * 100
                    st.metric("Case-Shiller", safe_value(val), f"{yoy:+.2f}% YoY")
                else:
                    st.metric("Case-Shiller", "N/A")
            
            with col2:
                val = latest.get('MORTGAGE30US')
                st.metric("Hipoteca 30A", safe_value(val, "{:.2f}%"))
            
            with col3:
                val = latest.get('HOUST')
                st.metric("Inicios Construcción", safe_value(val, "{:.0f}K"))
            
            with col4:
                val = latest.get('HSN1FNSA')
                st.metric("Ventas Nuevas", safe_value(val, "{:.0f}K"))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['CSUSHPISA', 'USSTHPI']].dropna(how='all'),
                    "🏘️ Índices de Precios de Vivienda", "Índice",
                    colors=['#10b981', '#06b6d4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'HOUST' in data.columns and 'MORTGAGE30US' in data.columns:
                    fig = create_dual_axis_chart(
                        data, 'HOUST', 'MORTGAGE30US',
                        "🏗️ Construcción vs Hipotecas",
                        "Inicios (K)", "Tasa (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Actividad de construcción
            st.markdown("### Actividad de Construcción y Permisos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['HOUST', 'HOUST1F', 'PERMIT', 'PERMIT1']].dropna(how='all'),
                    "🏗️ Inicios y Permisos de Construcción", "Miles",
                    colors=['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['HSN1FNSA', 'EXHOSLUSM495S']].dropna(how='all'),
                    "🏡 Ventas de Viviendas", "Miles",
                    colors=['#10b981', '#3b82f6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Construcción y empleo
            st.markdown("### Gasto en Construcción y Empleo")
            
            if 'PRRESCON' in data.columns and 'USCONS' in data.columns:
                fig = create_dual_axis_chart(
                    data, 'PRRESCON', 'USCONS',
                    "💼 Gasto en Construcción vs Empleo",
                    "Gasto (Millones $)", "Empleados (K)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: CURVAS DE RENDIMIENTO
    with tab5:
        st.markdown("## 📊 Curvas de Rendimiento del Tesoro")
        
        yield_series = {
            'DGS1MO': 'DGS1MO', 'DGS3MO': 'DGS3MO', 'DGS6MO': 'DGS6MO',
            'DGS1': 'DGS1', 'DGS2': 'DGS2', 'DGS3': 'DGS3',
            'DGS5': 'DGS5', 'DGS7': 'DGS7', 'DGS10': 'DGS10',
            'DGS20': 'DGS20', 'DGS30': 'DGS30'
        }
        
        maturity_labels = {
            'DGS1MO': '1M', 'DGS3MO': '3M', 'DGS6MO': '6M',
            'DGS1': '1A', 'DGS2': '2A', 'DGS3': '3A',
            'DGS5': '5A', 'DGS7': '7A', 'DGS10': '10A',
            'DGS20': '20A', 'DGS30': '30A'
        }
        
        with st.spinner('Cargando curva de rendimiento...'):
            data = fetch_multiple_series(yield_series, start_date, end_date)
        
        if data is not None and not data.empty:
            st.markdown("### Curva de Rendimiento Actual")
            
            latest_yields = data.iloc[-1].dropna()
            
            if not latest_yields.empty:
                maturities = [maturity_labels.get(col, col) for col in latest_yields.index]
                yields = list(latest_yields.values)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=maturities, y=yields,
                    mode='lines+markers',
                    line=dict(color='#06b6d4', width=4),
                    marker=dict(size=10, color='#06b6d4'),
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text="📈 Curva de Rendimiento (Actual)", 
                              font=dict(size=24, color='#06b6d4', family='Arial Black')),
                    xaxis_title="Vencimiento", yaxis_title="Rendimiento (%)",
                    plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
                    font=dict(color='#e2e8f0', size=14),
                    xaxis=dict(showgrid=True, gridcolor='#334155'),
                    yaxis=dict(showgrid=True, gridcolor='#334155'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Evolución de Rendimientos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['DGS2', 'DGS10', 'DGS30']].dropna(how='all'),
                    "📊 Rendimientos Principales", "Rendimiento (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                spreads = pd.DataFrame({
                    '10A-2A': data.get('DGS10', pd.Series()) - data.get('DGS2', pd.Series()),
                    '10A-3M': data.get('DGS10', pd.Series()) - data.get('DGS3MO', pd.Series())
                }).dropna(how='all')
                fig = create_line_chart(spreads, "📉 Spreads de Rendimiento", "Spread (%)")
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: ECONOMÍA & CONSUMO
    with tab6:
        st.markdown("## 💼 Panel de Economía y Consumo")
        
        econ_series = {
            'PCE': 'PCE',
            'PCES': 'PCES',
            'DGDSRC1': 'DGDSRC1',
            'RSXFS': 'RSXFS',
            'RSAFS': 'RSAFS',
            'DGORDER': 'DGORDER',
            'UMCSENT': 'UMCSENT',
            'MICH': 'MICH',
            'RPI': 'RPI'
        }
        
        with st.spinner('Cargando datos económicos...'):
            data = fetch_multiple_series(econ_series, start_date, end_date)
        
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                val = latest.get('PCE')
                st.metric("Gasto Personal", safe_value(val, "{:.0f}B"))
            
            with col2:
                val = latest.get('UMCSENT')
                st.metric("Confianza Consumidor", safe_value(val, "{:.1f}"))
            
            with col3:
                val = latest.get('DGORDER')
                st.metric("Pedidos Bienes Duraderos", safe_value(val, "{:.0f}M"))
            
            with col4:
                val = latest.get('RPI')
                st.metric("Ingreso Personal Real", safe_value(val, "{:.0f}B"))
            
            st.markdown("---")
            
            # Consumo
            st.markdown("### Consumo Personal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['PCE', 'PCES', 'DGDSRC1']].dropna(how='all'),
                    "💰 Gasto de Consumo Personal", "Miles de Millones $",
                    colors=['#06b6d4', '#8b5cf6', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['RSXFS', 'RSAFS']].dropna(how='all'),
                    "🛒 Ventas Minoristas", "Millones $",
                    colors=['#10b981', '#3b82f6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confianza y expectativas
            st.markdown("### Confianza del Consumidor")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['UMCSENT', 'MICH']].dropna(how='all'),
                    "😊 Índice de Confianza del Consumidor", "Índice",
                    colors=['#f59e0b', '#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'DGORDER' in data.columns:
                    fig = create_area_chart(
                        data[['DGORDER']].dropna(),
                        "📦 Pedidos de Bienes Duraderos", "Millones $",
                        color='#8b5cf6'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="collaboration-footer">
        <p class="collab-text">Desarrollado en colaboración con</p>
        <p>
            <a href="https://marotstrategies.com" target="_blank">🎯 Marot Strategies</a>
            <span style="color: #475569;">|</span>
            <a href="https://bquantfinance.com" target="_blank">📊 bquant</a>
        </p>
        <p style="font-size: 0.8rem; color: #64748b; margin-top: 5px;">
            Plataforma de Análisis Económico Profesional © 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logo adicional en sidebar
    with st.sidebar:
        st.markdown("---")
        try:
            st.image("marot.avif", use_column_width=True)
        except:
            pass

if __name__ == "__main__":
    main()
