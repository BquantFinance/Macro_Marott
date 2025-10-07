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
        transition: color 0.3s;
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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_data(series_id, start_date, end_date):
    """Obtener datos de FRED usando pandas_datareader"""
    try:
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        df = df.dropna()  # Eliminar valores NaN
        return df
    except Exception as e:
        st.error(f"Error al obtener {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_series(series_dict, start_date, end_date):
    """Obtener múltiples series de FRED y combinarlas"""
    dfs = {}
    errors = []
    
    for name, series_id in series_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            if df is not None and not df.empty:
                dfs[name] = df.iloc[:, 0]  # Tomar la primera columna
        except Exception as e:
            errors.append(f"{name} ({series_id}): {str(e)}")
    
    if errors:
        st.warning(f"No se pudieron obtener algunas series: {', '.join(errors)}")
    
    if dfs:
        combined = pd.DataFrame(dfs)
        combined = combined.dropna(how='all')  # Eliminar filas con todos NaN
        return combined
    return None

def safe_metric_value(value, format_str="{:.2f}"):
    """Formatear valor de métrica de forma segura"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

def create_line_chart(df, title, y_title, colors=None):
    """Crear un gráfico de líneas"""
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    
    if colors is None:
        colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    for i, col in enumerate(df.columns):
        if df[col].notna().any():  # Solo graficar si hay datos
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate='<b>%{fullData.name}</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Fecha",
        yaxis_title=y_title,
        hovermode='x unified',
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155', linecolor='#475569'),
        yaxis=dict(showgrid=True, gridcolor='#334155', linecolor='#475569'),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
        height=500
    )
    
    return fig

def create_area_chart(df, title, y_title, color='#06b6d4'):
    """Crear un gráfico de área"""
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    
    for col in df.columns:
        if df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode='lines',
                fill='tozeroy',
                line=dict(width=2, color=color),
                fillcolor='rgba(6, 182, 212, 0.3)',
                hovertemplate='<b>%{fullData.name}</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Fecha",
        yaxis_title=y_title,
        hovermode='x unified',
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155', linecolor='#475569'),
        yaxis=dict(showgrid=True, gridcolor='#334155', linecolor='#475569'),
        height=500
    )
    
    return fig

def create_dual_axis_chart(df, col1, col2, title, y1_title, y2_title):
    """Crear un gráfico de doble eje"""
    if df is None or df.empty or col1 not in df.columns or col2 not in df.columns:
        return create_empty_chart(title)
    
    # Filtrar datos válidos
    valid_data = df[[col1, col2]].dropna()
    
    if valid_data.empty:
        return create_empty_chart(title)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=valid_data.index, 
            y=valid_data[col1], 
            name=col1,
            line=dict(color='#06b6d4', width=3),
            hovertemplate=f'<b>{col1}</b><br>%{{y:.2f}}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=valid_data.index, 
            y=valid_data[col2], 
            name=col2,
            line=dict(color='#ec4899', width=3),
            hovertemplate=f'<b>{col2}</b><br>%{{y:.2f}}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        hovermode='x unified',
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(showgrid=True, gridcolor='#334155', linecolor='#475569'),
        height=500,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1)
    )
    
    fig.update_yaxes(title_text=y1_title, secondary_y=False, showgrid=True, gridcolor='#334155')
    fig.update_yaxes(title_text=y2_title, secondary_y=True, showgrid=False)
    
    return fig

def create_empty_chart(title):
    """Crear un gráfico vacío con mensaje"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="No hay datos disponibles",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color='#94a3b8')
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig

def main():
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
        st.markdown("• Índices S&P Dow Jones")
        
        st.markdown("---")
        st.markdown("### 🔄 Última Actualización")
        st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Política Monetaria",
        "📈 Métricas de Inflación", 
        "👥 Mercado Laboral",
        "🏠 Sector Inmobiliario",
        "📊 Curvas de Rendimiento"
    ])
    
    with tab1:
        st.markdown("## 💰 Panel de Política Monetaria")
        
        monetary_series = {
            'FEDFUNDS': 'FEDFUNDS',
            'DGS2': 'DGS2',
            'DGS10': 'DGS10',
            'DGS30': 'DGS30'
        }
        
        with st.spinner('Cargando datos de política monetaria...'):
            monetary_data = fetch_multiple_series(monetary_series, start_date, end_date)
        
        if monetary_data is not None and not monetary_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            latest = monetary_data.iloc[-1]
            prev = monetary_data.iloc[-30] if len(monetary_data) > 30 else monetary_data.iloc[0]
            
            with col1:
                val = latest.get('FEDFUNDS', None)
                prev_val = prev.get('FEDFUNDS', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Tasa Fed Funds", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col2:
                val = latest.get('DGS2', None)
                prev_val = prev.get('DGS2', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Tesoro 2 Años", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col3:
                val = latest.get('DGS10', None)
                prev_val = prev.get('DGS10', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Tesoro 10 Años", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col4:
                dgs10 = latest.get('DGS10', None)
                dgs2 = latest.get('DGS2', None)
                if dgs10 and dgs2 and not pd.isna(dgs10) and not pd.isna(dgs2):
                    spread = dgs10 - dgs2
                    st.metric("Spread 10A-2A", 
                             f"{spread:.2f}%",
                             "Invertida" if spread < 0 else "Normal",
                             delta_color="inverse")
                else:
                    st.metric("Spread 10A-2A", "N/A")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                plot_data = monetary_data[['FEDFUNDS', 'DGS2', 'DGS10']].dropna(how='all')
                fig1 = create_line_chart(
                    plot_data,
                    "📊 Tasas de Interés a lo Largo del Tiempo",
                    "Tasa (%)"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if 'DGS10' in monetary_data.columns and 'DGS2' in monetary_data.columns:
                    spread_df = pd.DataFrame({
                        'Spread 10A-2A': monetary_data['DGS10'] - monetary_data['DGS2']
                    }).dropna()
                    fig2 = create_area_chart(
                        spread_df,
                        "📉 Spread de Curva de Rendimiento (10A-2A)",
                        "Spread (%)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos de política monetaria")
    
    with tab2:
        st.markdown("## 📈 Panel de Métricas de Inflación")
        
        inflation_series = {
            'CPIAUCSL': 'CPIAUCSL',
            'CPILFESL': 'CPILFESL',
            'PCEPI': 'PCEPI',
            'PCEPILFE': 'PCEPILFE'
        }
        
        with st.spinner('Cargando datos de inflación...'):
            inflation_data = fetch_multiple_series(inflation_series, start_date, end_date)
        
        if inflation_data is not None and not inflation_data.empty:
            inflation_yoy = inflation_data.pct_change(12) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            latest_yoy = inflation_yoy.iloc[-1]
            prev_yoy = inflation_yoy.iloc[-13] if len(inflation_yoy) > 13 else inflation_yoy.iloc[0]
            
            with col1:
                val = latest_yoy.get('CPIAUCSL', None)
                prev_val = prev_yoy.get('CPIAUCSL', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("IPC Interanual", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col2:
                val = latest_yoy.get('CPILFESL', None)
                prev_val = prev_yoy.get('CPILFESL', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("IPC Subyacente Interanual", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col3:
                val = latest_yoy.get('PCEPI', None)
                prev_val = prev_yoy.get('PCEPI', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("PCE Interanual", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col4:
                val = latest_yoy.get('PCEPILFE', None)
                prev_val = prev_yoy.get('PCEPILFE', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("PCE Subyacente Interanual", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    inflation_data.dropna(how='all'),
                    "📊 Índices de Inflación (Nivel)",
                    "Nivel del Índice"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_line_chart(
                    inflation_yoy.dropna(how='all'),
                    "📈 Tasa de Inflación (% Cambio Interanual)",
                    "Cambio Interanual (%)",
                    colors=['#f59e0b', '#ef4444', '#ec4899', '#8b5cf6']
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos de inflación")
    
    with tab3:
        st.markdown("## 👥 Panel del Mercado Laboral")
        
        labor_series = {
            'UNRATE': 'UNRATE',
            'PAYEMS': 'PAYEMS',
            'CIVPART': 'CIVPART',
            'ICSA': 'ICSA'
        }
        
        with st.spinner('Cargando datos del mercado laboral...'):
            labor_data = fetch_multiple_series(labor_series, start_date, end_date)
        
        if labor_data is not None and not labor_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            latest = labor_data.iloc[-1]
            prev = labor_data.iloc[-2] if len(labor_data) > 1 else labor_data.iloc[0]
            
            with col1:
                val = latest.get('UNRATE', None)
                prev_val = prev.get('UNRATE', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Tasa de Desempleo", 
                         safe_metric_value(val, "{:.1f}%"),
                         f"{change:+.1f}%" if change != 0 else "",
                         delta_color="inverse")
            
            with col2:
                val = latest.get('PAYEMS', None)
                prev_val = prev.get('PAYEMS', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Nóminas No Agrícolas", 
                         safe_metric_value(val, "{:.0f}K"),
                         f"{change:+.0f}K" if change != 0 else "")
            
            with col3:
                val = latest.get('CIVPART', None)
                prev_val = prev.get('CIVPART', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Participación Laboral", 
                         safe_metric_value(val, "{:.1f}%"),
                         f"{change:+.1f}%" if change != 0 else "")
            
            with col4:
                val = latest.get('ICSA', None)
                prev_val = prev.get('ICSA', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Solicitudes Iniciales", 
                         safe_metric_value(val, "{:.0f}K"),
                         f"{change:+.0f}K" if change != 0 else "",
                         delta_color="inverse")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'UNRATE' in labor_data.columns:
                    fig1 = create_area_chart(
                        labor_data[['UNRATE']].dropna(),
                        "📉 Tasa de Desempleo",
                        "Tasa (%)",
                        color='#8b5cf6'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if 'PAYEMS' in labor_data.columns:
                    fig2 = create_line_chart(
                        labor_data[['PAYEMS']].dropna(),
                        "📊 Nóminas No Agrícolas",
                        "Miles",
                        colors=['#10b981']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos del mercado laboral")
    
    with tab4:
        st.markdown("## 🏠 Panel del Sector Inmobiliario")
        
        real_estate_series = {
            'CSUSHPISA': 'CSUSHPISA',
            'HOUST': 'HOUST',
            'MORTGAGE30US': 'MORTGAGE30US',
            'MSPNHSUS': 'MSPNHSUS'
        }
        
        with st.spinner('Cargando datos del sector inmobiliario...'):
            re_data = fetch_multiple_series(real_estate_series, start_date, end_date)
        
        if re_data is not None and not re_data.empty:
            col1, col2, col3 = st.columns(3)
            
            latest = re_data.iloc[-1]
            prev_year = re_data.iloc[-12] if len(re_data) > 12 else re_data.iloc[0]
            
            with col1:
                val = latest.get('CSUSHPISA', None)
                prev_val = prev_year.get('CSUSHPISA', None)
                if val and prev_val and not pd.isna(val) and not pd.isna(prev_val):
                    yoy_change = ((val / prev_val) - 1) * 100
                    st.metric("Índice Case-Shiller", 
                             safe_metric_value(val),
                             f"{yoy_change:+.2f}% Interanual")
                else:
                    st.metric("Índice Case-Shiller", "N/A")
            
            with col2:
                val = latest.get('MORTGAGE30US', None)
                prev_val = prev_year.get('MORTGAGE30US', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Hipoteca 30 Años", 
                         safe_metric_value(val, "{:.2f}%"),
                         f"{change:+.2f}%" if change != 0 else "")
            
            with col3:
                val = latest.get('HOUST', None)
                prev_val = prev_year.get('HOUST', None)
                change = val - prev_val if (val and prev_val and not pd.isna(val) and not pd.isna(prev_val)) else 0
                st.metric("Inicios de Construcción", 
                         safe_metric_value(val, "{:.0f}K"),
                         f"{change:+.0f}K" if change != 0 else "")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'CSUSHPISA' in re_data.columns:
                    fig1 = create_line_chart(
                        re_data[['CSUSHPISA']].dropna(),
                        "🏘️ Índice de Precios Case-Shiller",
                        "Índice (Ene 2000 = 100)",
                        colors=['#10b981']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if 'HOUST' in re_data.columns and 'MORTGAGE30US' in re_data.columns:
                    fig2 = create_dual_axis_chart(
                        re_data,
                        'HOUST',
                        'MORTGAGE30US',
                        "🏗️ Construcción vs Tasas Hipotecarias",
                        "Inicios de Construcción (K)",
                        "Tasa Hipotecaria (%)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos del sector inmobiliario")
    
    with tab5:
        st.markdown("## 📊 Curvas de Rendimiento del Tesoro")
        
        yield_series = {
            'DGS1MO': 'DGS1MO',
            'DGS3MO': 'DGS3MO',
            'DGS6MO': 'DGS6MO',
            'DGS1': 'DGS1',
            'DGS2': 'DGS2',
            'DGS3': 'DGS3',
            'DGS5': 'DGS5',
            'DGS7': 'DGS7',
            'DGS10': 'DGS10',
            'DGS20': 'DGS20',
            'DGS30': 'DGS30'
        }
        
        maturity_labels = {
            'DGS1MO': '1M', 'DGS3MO': '3M', 'DGS6MO': '6M',
            'DGS1': '1A', 'DGS2': '2A', 'DGS3': '3A',
            'DGS5': '5A', 'DGS7': '7A', 'DGS10': '10A',
            'DGS20': '20A', 'DGS30': '30A'
        }
        
        with st.spinner('Cargando datos de curva de rendimiento...'):
            yield_data = fetch_multiple_series(yield_series, start_date, end_date)
        
        if yield_data is not None and not yield_data.empty:
            st.markdown("### Curva de Rendimiento Actual")
            
            latest_yields = yield_data.iloc[-1].dropna()
            
            if not latest_yields.empty:
                maturities = [maturity_labels.get(col, col) for col in latest_yields.index]
                yields = list(latest_yields.values)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=maturities,
                    y=yields,
                    mode='lines+markers',
                    line=dict(color='#06b6d4', width=4),
                    marker=dict(size=10, color='#06b6d4'),
                    hovertemplate='<b>%{x}</b><br>Rendimiento: %{y:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text="📈 Curva de Rendimiento del Tesoro (Actual)", 
                              font=dict(size=24, color='#06b6d4', family='Arial Black')),
                    xaxis_title="Vencimiento",
                    yaxis_title="Rendimiento (%)",
                    plot_bgcolor='#0f172a',
                    paper_bgcolor='#1e293b',
                    font=dict(color='#e2e8f0', size=14),
                    xaxis=dict(showgrid=True, gridcolor='#334155'),
                    yaxis=dict(showgrid=True, gridcolor='#334155'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Spreads de Rendimiento Históricos")
            
            spreads_data = {}
            if 'DGS10' in yield_data.columns and 'DGS2' in yield_data.columns:
                spreads_data['10A-2A'] = yield_data['DGS10'] - yield_data['DGS2']
            if 'DGS10' in yield_data.columns and 'DGS3MO' in yield_data.columns:
                spreads_data['10A-3M'] = yield_data['DGS10'] - yield_data['DGS3MO']
            if 'DGS30' in yield_data.columns and 'DGS5' in yield_data.columns:
                spreads_data['30A-5A'] = yield_data['DGS30'] - yield_data['DGS5']
            
            if spreads_data:
                spreads_df = pd.DataFrame(spreads_data).dropna(how='all')
                fig2 = create_line_chart(
                    spreads_df,
                    "📊 Spreads de Rendimiento del Tesoro",
                    "Spread (%)",
                    colors=['#06b6d4', '#8b5cf6', '#ec4899']
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos de curva de rendimiento")
    
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

if __name__ == "__main__":
    main()
