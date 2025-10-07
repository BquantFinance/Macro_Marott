import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pandas_datareader import data as web
from datetime import datetime, timedelta
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="FRED Análisis Económico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para modo oscuro estético
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
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    h1 {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #06b6d4;
        font-weight: 700;
        border-bottom: 2px solid #06b6d4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
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
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    }
    
    .stAlert {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #06b6d4, transparent);
        margin: 2rem 0;
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

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date, end_date):
    """Obtener datos de FRED usando pandas_datareader"""
    try:
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        return df
    except Exception as e:
        st.warning(f"No se pudo obtener {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_multiple_series(series_dict, start_date, end_date):
    """Obtener múltiples series de FRED y combinarlas"""
    dfs = {}
    for name, series_id in series_dict.items():
        df = fetch_fred_data(series_id, start_date, end_date)
        if df is not None:
            dfs[name] = df
    
    if dfs:
        combined = pd.concat(dfs, axis=1)
        combined.columns = list(dfs.keys())
        return combined
    return None

def create_line_chart(df, title, y_title, colors=None):
    """Crear un gráfico de líneas hermoso con Plotly"""
    fig = go.Figure()
    
    if colors is None:
        colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    for i, col in enumerate(df.columns):
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
        xaxis=dict(
            showgrid=True,
            gridcolor='#334155',
            linecolor='#475569'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#334155',
            linecolor='#475569'
        ),
        legend=dict(
            bgcolor='#1e293b',
            bordercolor='#334155',
            borderwidth=1
        ),
        height=500
    )
    
    return fig

def create_area_chart(df, title, y_title, color='#06b6d4'):
    """Crear un gráfico de área hermoso"""
    fig = go.Figure()
    
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode='lines',
            fill='tozeroy',
            line=dict(width=2, color=color),
            fillcolor=f'rgba(6, 182, 212, 0.3)',
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
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col1], name=col1,
                  line=dict(color='#06b6d4', width=3),
                  hovertemplate=f'<b>{col1}</b><br>%{{y:.2f}}<extra></extra>'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col2], name=col2,
                  line=dict(color='#ec4899', width=3),
                  hovertemplate=f'<b>{col2}</b><br>%{{y:.2f}}<extra></extra>'),
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
    
    fig.update_yaxes(title_text=y1_title, secondary_y=False, 
                     showgrid=True, gridcolor='#334155')
    fig.update_yaxes(title_text=y2_title, secondary_y=True,
                     showgrid=False)
    
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
            'Tasa Fed Funds': 'FEDFUNDS',
            'Tesoro 2A': 'DGS2',
            'Tesoro 10A': 'DGS10',
            'Tesoro 30A': 'DGS30'
        }
        
        with st.spinner('Cargando datos de política monetaria...'):
            monetary_data = fetch_multiple_series(monetary_series, start_date, end_date)
        
        if monetary_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            latest = monetary_data.iloc[-1]
            prev = monetary_data.iloc[-30] if len(monetary_data) > 30 else monetary_data.iloc[0]
            
            with col1:
                change = latest['Tasa Fed Funds'] - prev['Tasa Fed Funds']
                st.metric("Tasa Fed Funds", 
                         f"{latest['Tasa Fed Funds']:.2f}%",
                         f"{change:+.2f}%")
            
            with col2:
                change = latest['Tesoro 2A'] - prev['Tesoro 2A']
                st.metric("Tesoro 2 Años", 
                         f"{latest['Tesoro 2A']:.2f}%",
                         f"{change:+.2f}%")
            
            with col3:
                change = latest['Tesoro 10A'] - prev['Tesoro 10A']
                st.metric("Tesoro 10 Años", 
                         f"{latest['Tesoro 10A']:.2f}%",
                         f"{change:+.2f}%")
            
            with col4:
                spread = latest['Tesoro 10A'] - latest['Tesoro 2A']
                st.metric("Spread 10A-2A", 
                         f"{spread:.2f}%",
                         "Invertida" if spread < 0 else "Normal",
                         delta_color="inverse")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    monetary_data[['Tasa Fed Funds', 'Tesoro 2A', 'Tesoro 10A']],
                    "📊 Tasas de Interés a lo Largo del Tiempo",
                    "Tasa (%)"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                spread_df = pd.DataFrame({
                    'Spread 10A-2A': monetary_data['Tesoro 10A'] - monetary_data['Tesoro 2A']
                })
                fig2 = create_area_chart(
                    spread_df,
                    "📉 Spread de Curva de Rendimiento (10A-2A)",
                    "Spread (%)",
                    color='#06b6d4'
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("## 📈 Panel de Métricas de Inflación")
        
        inflation_series = {
            'IPC (Todos)': 'CPIAUCSL',
            'IPC Subyacente': 'CPILFESL',
            'PCE': 'PCEPI',
            'PCE Subyacente': 'PCEPILFE'
        }
        
        with st.spinner('Cargando datos de inflación...'):
            inflation_data = fetch_multiple_series(inflation_series, start_date, end_date)
        
        if inflation_data is not None:
            inflation_yoy = inflation_data.pct_change(12) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            latest_yoy = inflation_yoy.iloc[-1]
            
            with col1:
                st.metric("IPC Interanual", 
                         f"{latest_yoy['IPC (Todos)']:.2f}%",
                         f"{latest_yoy['IPC (Todos)'] - inflation_yoy.iloc[-13]['IPC (Todos)']:+.2f}%")
            
            with col2:
                st.metric("IPC Subyacente Interanual", 
                         f"{latest_yoy['IPC Subyacente']:.2f}%",
                         f"{latest_yoy['IPC Subyacente'] - inflation_yoy.iloc[-13]['IPC Subyacente']:+.2f}%")
            
            with col3:
                st.metric("PCE Interanual", 
                         f"{latest_yoy['PCE']:.2f}%",
                         f"{latest_yoy['PCE'] - inflation_yoy.iloc[-13]['PCE']:+.2f}%")
            
            with col4:
                st.metric("PCE Subyacente Interanual", 
                         f"{latest_yoy['PCE Subyacente']:.2f}%",
                         f"{latest_yoy['PCE Subyacente'] - inflation_yoy.iloc[-13]['PCE Subyacente']:+.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    inflation_data,
                    "📊 Índices de Inflación (Nivel)",
                    "Nivel del Índice"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_line_chart(
                    inflation_yoy,
                    "📈 Tasa de Inflación (% Cambio Interanual)",
                    "Cambio Interanual (%)",
                    colors=['#f59e0b', '#ef4444', '#ec4899', '#8b5cf6']
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("## 👥 Panel del Mercado Laboral")
        
        labor_series = {
            'Tasa de Desempleo': 'UNRATE',
            'Nóminas No Agrícolas': 'PAYEMS',
            'Participación Laboral': 'CIVPART',
            'Solicitudes Iniciales': 'ICSA'
        }
        
        with st.spinner('Cargando datos del mercado laboral...'):
            labor_data = fetch_multiple_series(labor_series, start_date, end_date)
        
        if labor_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            latest = labor_data.iloc[-1]
            prev = labor_data.iloc[-2]
            
            with col1:
                change = latest['Tasa de Desempleo'] - prev['Tasa de Desempleo']
                st.metric("Tasa de Desempleo", 
                         f"{latest['Tasa de Desempleo']:.1f}%",
                         f"{change:+.1f}%",
                         delta_color="inverse")
            
            with col2:
                change = latest['Nóminas No Agrícolas'] - prev['Nóminas No Agrícolas']
                st.metric("Nóminas No Agrícolas", 
                         f"{latest['Nóminas No Agrícolas']:.0f}K",
                         f"{change:+.0f}K")
            
            with col3:
                st.metric("Participación Laboral", 
                         f"{latest['Participación Laboral']:.1f}%",
                         f"{latest['Participación Laboral'] - prev['Participación Laboral']:+.1f}%")
            
            with col4:
                st.metric("Solicitudes Iniciales", 
                         f"{latest['Solicitudes Iniciales']:.0f}K",
                         f"{latest['Solicitudes Iniciales'] - prev['Solicitudes Iniciales']:+.0f}K",
                         delta_color="inverse")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_area_chart(
                    labor_data[['Tasa de Desempleo']],
                    "📉 Tasa de Desempleo",
                    "Tasa (%)",
                    color='#8b5cf6'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_line_chart(
                    labor_data[['Nóminas No Agrícolas']],
                    "📊 Nóminas No Agrícolas",
                    "Miles",
                    colors=['#10b981']
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.markdown("## 🏠 Panel del Sector Inmobiliario")
        
        real_estate_series = {
            'Índice Case-Shiller': 'CSUSHPISA',
            'Inicios de Construcción': 'HOUST',
            'Hipoteca 30A': 'MORTGAGE30US',
            'Precio Medio Vivienda': 'MSPNHSUS',
            'Tasa Propiedad': 'RHORUSQ156N'
        }
        
        with st.spinner('Cargando datos del sector inmobiliario...'):
            re_data = fetch_multiple_series(real_estate_series, start_date, end_date)
        
        if re_data is not None:
            col1, col2, col3 = st.columns(3)
            
            latest = re_data.iloc[-1]
            prev_year = re_data.iloc[-12] if len(re_data) > 12 else re_data.iloc[0]
            
            with col1:
                yoy_change = ((latest['Índice Case-Shiller'] / prev_year['Índice Case-Shiller']) - 1) * 100
                st.metric("Índice Case-Shiller", 
                         f"{latest['Índice Case-Shiller']:.2f}",
                         f"{yoy_change:+.2f}% Interanual")
            
            with col2:
                st.metric("Hipoteca 30 Años", 
                         f"{latest['Hipoteca 30A']:.2f}%",
                         f"{latest['Hipoteca 30A'] - prev_year['Hipoteca 30A']:+.2f}%")
            
            with col3:
                st.metric("Inicios de Construcción", 
                         f"{latest['Inicios de Construcción']:.0f}K",
                         f"{latest['Inicios de Construcción'] - prev_year['Inicios de Construcción']:+.0f}K")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    re_data[['Índice Case-Shiller']],
                    "🏘️ Índice de Precios Case-Shiller",
                    "Índice (Ene 2000 = 100)",
                    colors=['#10b981']
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_dual_axis_chart(
                    re_data,
                    'Inicios de Construcción',
                    'Hipoteca 30A',
                    "🏗️ Construcción vs Tasas Hipotecarias",
                    "Inicios de Construcción (K)",
                    "Tasa Hipotecaria (%)"
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab5:
        st.markdown("## 📊 Curvas de Rendimiento del Tesoro")
        
        yield_series = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1A': 'DGS1',
            '2A': 'DGS2',
            '3A': 'DGS3',
            '5A': 'DGS5',
            '7A': 'DGS7',
            '10A': 'DGS10',
            '20A': 'DGS20',
            '30A': 'DGS30'
        }
        
        with st.spinner('Cargando datos de curva de rendimiento...'):
            yield_data = fetch_multiple_series(yield_series, start_date, end_date)
        
        if yield_data is not None:
            st.markdown("### Curva de Rendimiento Actual")
            
            latest_yields = yield_data.iloc[-1].dropna()
            maturities = list(latest_yields.index)
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
            
            spreads_df = pd.DataFrame({
                '10A-2A': yield_data['10A'] - yield_data['2A'],
                '10A-3M': yield_data['10A'] - yield_data['3M'],
                '30A-5A': yield_data['30A'] - yield_data['5A']
            })
            
            fig2 = create_line_chart(
                spreads_df,
                "📊 Spreads de Rendimiento del Tesoro",
                "Spread (%)",
                colors=['#06b6d4', '#8b5cf6', '#ec4899']
            )
            st.plotly_chart(fig2, use_container_width=True)
    
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
