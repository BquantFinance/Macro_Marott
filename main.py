import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data as web
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

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
    .stExpander {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_series(series_dict, start_date, end_date):
    """Obtener múltiples series de FRED"""
    dfs = {}
    for name, series_id in series_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            if df is not None and not df.empty:
                dfs[name] = df.iloc[:, 0]
        except:
            pass
    
    if dfs:
        combined = pd.DataFrame(dfs)
        combined = combined.dropna(how='all')
        return combined
    return None

def safe_value(value, format_str="{:.2f}"):
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

def get_last_valid(df, column):
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if len(series) == 0:
        return None
    return series.iloc[-1]

def create_scatter_with_regression(df, x_col, y_col, title, x_title, y_title, color='#f59e0b'):
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return create_empty_chart(title)
    
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) < 2:
        return create_empty_chart(title)
    
    x = valid_data[x_col].values
    y = valid_data[y_col].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Datos',
        marker=dict(size=8, color=color, opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode='lines',
        name=f'R²={r_value**2:.4f}',
        line=dict(color='#ef4444', width=2)
    ))
    
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>R² = {r_value**2:.4f}</sub>", 
                  font=dict(size=16, color='#06b6d4')),
        xaxis_title=x_title, yaxis_title=y_title,
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        legend=dict(bgcolor='#1e293b'),
        height=400
    )
    return fig

def create_bar_chart(df, title, y_title, color='#f59e0b'):
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    for col in df.columns:
        if df[col].notna().any():
            fig.add_trace(go.Bar(x=df.index, y=df[col], name=col, marker_color=color))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        xaxis_title="Fecha", yaxis_title=y_title,
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        height=400
    )
    return fig

def create_line_chart(df, title, y_title, colors=None):
    if df is None or df.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    if colors is None:
        colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    for i, col in enumerate(df.columns):
        if df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col, mode='lines',
                line=dict(width=2, color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        xaxis_title="Fecha", yaxis_title=y_title,
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        height=400
    )
    return fig

def create_dual_axis_chart(df, col1, col2, title, y1_title, y2_title):
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
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        height=400
    )
    
    fig.update_yaxes(title_text=y1_title, secondary_y=False, showgrid=True, gridcolor='#334155')
    fig.update_yaxes(title_text=y2_title, secondary_y=True)
    
    return fig

def create_empty_chart(title):
    fig = go.Figure()
    fig.add_annotation(
        text="No hay datos disponibles", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#94a3b8')
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        height=400, xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig

def info_box(title, content):
    """Crear caja de información"""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                padding: 15px; border-radius: 10px; border-left: 4px solid #06b6d4; margin: 10px 0;'>
        <h4 style='color: #06b6d4; margin: 0 0 10px 0;'>💡 {title}</h4>
        <p style='color: #e2e8f0; margin: 0; font-size: 0.95rem;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("• S&P Dow Jones Indices")
        
        st.markdown("---")
        st.markdown("### 🔄 Última Actualización")
        st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        try:
            st.image("marot.avif", use_container_width=True)
        except:
            pass
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Política Monetaria",
        "📈 Inflación & Laboral", 
        "🏠 Sector Inmobiliario",
        "📊 Curvas & Correlaciones",
        "📚 Guía de Interpretación"
    ])
    
    # TAB 1: POLÍTICA MONETARIA
    with tab1:
        st.markdown("## 💰 Panel de Política Monetaria")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Política Monetaria", 
                    "Análisis de las tasas de interés controladas por la Reserva Federal y los rendimientos de bonos del Tesoro. "
                    "Estos indicadores son fundamentales para entender el costo del dinero en la economía y las expectativas de inflación.")
        
        monetary_series = {
            'FEDFUNDS': 'FEDFUNDS', 'DGS2': 'DGS2', 'DGS10': 'DGS10',
            'DGS30': 'DGS30', 'T10Y2Y': 'T10Y2Y', 'T5YIE': 'T5YIE'
        }
        
        with st.spinner('Cargando datos...'):
            data = fetch_multiple_series(monetary_series, start_date, end_date)
        
        if data is not None and not data.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fed Funds", safe_value(get_last_valid(data, 'FEDFUNDS'), "{:.2f}%"))
            with col2:
                st.metric("Tesoro 10A", safe_value(get_last_valid(data, 'DGS10'), "{:.2f}%"))
            with col3:
                val = get_last_valid(data, 'T10Y2Y')
                st.metric("Spread 10A-2A", safe_value(val, "{:.2f}%"), 
                         "Invertida" if (val and val < 0) else "Normal")
            with col4:
                st.metric("Inflación Esperada 5A", safe_value(get_last_valid(data, 'T5YIE'), "{:.2f}%"))
            
            st.markdown("---")
            st.markdown("### 📈 Evolución de Tasas de Interés")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['FEDFUNDS', 'DGS2', 'DGS10', 'DGS30']].dropna(how='all'),
                    "Tasas de Interés Principales", "Tasa (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['T10Y2Y']].dropna(),
                    "Spread 10Y-2Y (Indicador de Recesión)", "Spread (%)",
                    colors=['#8b5cf6']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: INFLACIÓN & LABORAL
    with tab2:
        st.markdown("## 📈 Inflación y Mercado Laboral")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Inflación", 
                    "Mide el aumento de precios en la economía. El IPC (Índice de Precios al Consumidor) y el PCE "
                    "(Gasto de Consumo Personal) son las principales métricas que la Fed monitorea.")
            info_box("Mercado Laboral", 
                    "Indicadores clave: tasa de desempleo, nóminas no agrícolas (NFP), salarios y vacantes (JOLTS). "
                    "Un mercado laboral fuerte puede presionar la inflación al alza.")
        
        combined_series = {
            'CPIAUCSL': 'CPIAUCSL', 'CPILFESL': 'CPILFESL',
            'UNRATE': 'UNRATE', 'PAYEMS': 'PAYEMS',
            'CES0500000003': 'CES0500000003', 'JTSJOL': 'JTSJOL'
        }
        
        with st.spinner('Cargando datos...'):
            data = fetch_multiple_series(combined_series, start_date, end_date)
        
        if data is not None and not data.empty:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            yoy = data[['CPIAUCSL', 'CPILFESL']].pct_change(12) * 100
            
            with col1:
                st.metric("IPC YoY", safe_value(get_last_valid(yoy, 'CPIAUCSL'), "{:.2f}%"))
            with col2:
                st.metric("IPC Subyacente YoY", safe_value(get_last_valid(yoy, 'CPILFESL'), "{:.2f}%"))
            with col3:
                st.metric("Desempleo", safe_value(get_last_valid(data, 'UNRATE'), "{:.1f}%"))
            with col4:
                st.metric("Nóminas", safe_value(get_last_valid(data, 'PAYEMS'), "{:.0f}K"))
            
            st.markdown("---")
            st.markdown("### 💹 Inflación - Cambios Interanuales")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    yoy.dropna(),
                    "IPC - Evolución YoY", "% YoY",
                    colors=['#f59e0b', '#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cambios mensuales
                mom = data['CPIAUCSL'].pct_change(1) * 100
                fig = create_bar_chart(
                    pd.DataFrame({'IPC MoM': mom}).dropna().tail(60),
                    "IPC - Cambios Mensuales", "% MoM"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 👥 Mercado Laboral")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['UNRATE', 'JTSJOL']].dropna(how='all'),
                    "Desempleo y Vacantes JOLTS", "Miles / %",
                    colors=['#8b5cf6', '#06b6d4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                nfp_chg = data['PAYEMS'].diff()
                fig = create_bar_chart(
                    pd.DataFrame({'NFP Cambio': nfp_chg}).dropna().tail(60),
                    "Cambios Mensuales en Nóminas", "Miles",
                    color='#10b981'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔗 Correlaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                scatter_data = pd.DataFrame({
                    'UNRATE': data['UNRATE'],
                    'CPI_YoY': yoy['CPIAUCSL']
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'UNRATE', 'CPI_YoY',
                    "Desempleo vs Inflación", "Desempleo (%)", "IPC YoY (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'CES0500000003' in data.columns:
                    scatter_data = pd.DataFrame({
                        'Wage': data['CES0500000003'].pct_change(12) * 100,
                        'CPI_YoY': yoy['CPIAUCSL']
                    }).dropna()
                    
                    fig = create_scatter_with_regression(
                        scatter_data, 'Wage', 'CPI_YoY',
                        "Salarios vs Inflación", "Salarios YoY (%)", "IPC YoY (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: SECTOR INMOBILIARIO
    with tab3:
        st.markdown("## 🏠 Sector Inmobiliario")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Precios de Vivienda", 
                    "Case-Shiller es el índice más importante de precios de vivienda en EE.UU. "
                    "Mide la evolución de los precios en las principales áreas metropolitanas.")
            info_box("Actividad de Construcción", 
                    "Housing Starts (inicios de construcción) y Building Permits (permisos) son indicadores "
                    "adelantados de la actividad económica y demanda de vivienda.")
            info_box("Financiamiento", 
                    "Las tasas hipotecarias determinan la asequibilidad de las viviendas. Tasas más altas "
                    "reducen la demanda y pueden enfriar el mercado inmobiliario.")
        
        re_series = {
            'CSUSHPISA': 'CSUSHPISA', 'HOUST': 'HOUST',
            'MORTGAGE30US': 'MORTGAGE30US', 'HSN1FNSA': 'HSN1FNSA',
            'PERMIT': 'PERMIT', 'MSPNHSUS': 'MSPNHSUS',
            'EXHOSLUSM495S': 'EXHOSLUSM495S', 'DGS10': 'DGS10'
        }
        
        with st.spinner('Cargando datos inmobiliarios...'):
            data = fetch_multiple_series(re_series, start_date, end_date)
        
        if data is not None and not data.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Case-Shiller", safe_value(get_last_valid(data, 'CSUSHPISA')))
            with col2:
                st.metric("Hipoteca 30A", safe_value(get_last_valid(data, 'MORTGAGE30US'), "{:.2f}%"))
            with col3:
                st.metric("Housing Starts", safe_value(get_last_valid(data, 'HOUST'), "{:.0f}K"))
            with col4:
                st.metric("Ventas Nuevas", safe_value(get_last_valid(data, 'HSN1FNSA'), "{:.0f}K"))
            
            st.markdown("---")
            st.markdown("### 📊 Precios de Vivienda")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['CSUSHPISA']].dropna(),
                    "Índice Case-Shiller", "Índice (2000=100)",
                    colors=['#10b981']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cs_yoy = data['CSUSHPISA'].pct_change(12) * 100
                fig = create_line_chart(
                    pd.DataFrame({'CS YoY': cs_yoy}).dropna(),
                    "Case-Shiller - Cambio Interanual", "% YoY",
                    colors=['#06b6d4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🏗️ Actividad de Construcción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['HOUST', 'PERMIT']].dropna(how='all'),
                    "Inicios y Permisos de Construcción", "Miles (Anualizado)",
                    colors=['#06b6d4', '#8b5cf6']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['HSN1FNSA', 'EXHOSLUSM495S']].dropna(how='all'),
                    "Ventas de Viviendas", "Miles",
                    colors=['#10b981', '#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📉 Impacto de Tasas de Interés")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'HOUST' in data.columns and 'MORTGAGE30US' in data.columns:
                    fig = create_dual_axis_chart(
                        data, 'HOUST', 'MORTGAGE30US',
                        "Housing Starts vs Tasa Hipotecaria",
                        "Starts (K)", "Tasa (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                scatter_data = pd.DataFrame({
                    'Mortgage': data['MORTGAGE30US'],
                    'Sales': data['HSN1FNSA']
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'Mortgage', 'Sales',
                    "Tasas Hipotecarias vs Ventas", "Tasa 30Y (%)", "Ventas (K)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💵 Asequibilidad")
            
            if 'MSPNHSUS' in data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_line_chart(
                        data[['MSPNHSUS']].dropna(),
                        "Precio Medio de Vivienda Nueva", "USD",
                        colors=['#ec4899']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Spread hipoteca vs 10Y
                    if 'DGS10' in data.columns:
                        spread = data['MORTGAGE30US'] - data['DGS10']
                        fig = create_line_chart(
                            pd.DataFrame({'Spread': spread}).dropna(),
                            "Spread: Hipoteca 30Y - Tesoro 10Y", "% Puntos",
                            colors=['#f59e0b']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: CURVAS & CORRELACIONES
    with tab4:
        st.markdown("## 📊 Análisis Avanzado")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Curva de Rendimientos", 
                    "La forma de la curva de rendimientos refleja las expectativas del mercado sobre crecimiento "
                    "e inflación futura. Una curva invertida (10Y < 2Y) históricamente ha precedido recesiones.")
            info_box("Análisis de Correlaciones", 
                    "Los scatter plots muestran relaciones históricas entre variables económicas, útiles para "
                    "entender cómo se mueven juntos diferentes indicadores.")
        
        advanced_series = {
            'DGS2': 'DGS2', 'DGS10': 'DGS10', 'DGS30': 'DGS30',
            'FEDFUNDS': 'FEDFUNDS', 'CPIAUCSL': 'CPIAUCSL',
            'UNRATE': 'UNRATE', 'CSUSHPISA': 'CSUSHPISA'
        }
        
        with st.spinner('Cargando datos...'):
            data = fetch_multiple_series(advanced_series, start_date, end_date)
        
        if data is not None and not data.empty:
            st.markdown("### 📈 Curva de Rendimientos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['DGS2', 'DGS10', 'DGS30']].dropna(how='all'),
                    "Rendimientos del Tesoro", "Rendimiento (%)",
                    colors=['#06b6d4', '#8b5cf6', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                spread = data['DGS10'] - data['DGS2']
                fig = create_line_chart(
                    pd.DataFrame({'10Y-2Y': spread}).dropna(),
                    "Spread 10Y-2Y", "Spread (%)",
                    colors=['#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔗 Matriz de Correlaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpi_yoy = data['CPIAUCSL'].pct_change(12) * 100
                scatter_data = pd.DataFrame({
                    'Fed_Funds': data['FEDFUNDS'],
                    'CPI_YoY': cpi_yoy
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'Fed_Funds', 'CPI_YoY',
                    "Fed Funds vs Inflación", "Fed Funds (%)", "IPC YoY (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                scatter_data = pd.DataFrame({
                    'DGS10': data['DGS10'],
                    'UNRATE': data['UNRATE']
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'UNRATE', 'DGS10',
                    "Desempleo vs Tesoro 10Y", "Desempleo (%)", "Rendimiento 10Y (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cs_yoy = data['CSUSHPISA'].pct_change(12) * 100
                scatter_data = pd.DataFrame({
                    'CPI_YoY': cpi_yoy,
                    'CS_YoY': cs_yoy
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'CPI_YoY', 'CS_YoY',
                    "Inflación vs Precios Vivienda", "IPC YoY (%)", "Case-Shiller YoY (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                scatter_data = pd.DataFrame({
                    'DGS10': data['DGS10'],
                    'CS': data['CSUSHPISA']
                }).dropna()
                
                fig = create_scatter_with_regression(
                    scatter_data, 'DGS10', 'CS',
                    "Tesoro 10Y vs Case-Shiller", "Rendimiento 10Y (%)", "Case-Shiller"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: GUÍA
    with tab5:
        st.markdown("## 📚 Guía de Interpretación")
        
        st.markdown("### 🎯 ¿Cómo usar este dashboard?")
        
        with st.expander("💰 Política Monetaria - Guía"):
            st.markdown("""
            **Fed Funds Rate (Tasa de Fondos Federales)**
            - Tasa de interés de referencia controlada por la Reserva Federal
            - Influye en todas las demás tasas de la economía
            - Subidas = política restrictiva (frenar inflación)
            - Bajadas = política expansiva (estimular economía)
            
            **Rendimientos del Tesoro**
            - DGS2: Bonos a 2 años → expectativas corto plazo
            - DGS10: Bonos a 10 años → expectativas largo plazo
            - DGS30: Bonos a 30 años → expectativas muy largo plazo
            
            **Spread 10Y-2Y**
            - Positivo: Curva normal (economía saludable)
            - Negativo: Curva invertida (⚠️ señal de recesión)
            - Históricamente ha precedido todas las recesiones desde 1970
            """)
        
        with st.expander("📈 Inflación - Guía"):
            st.markdown("""
            **Índice de Precios al Consumidor (IPC)**
            - Mide el cambio en precios de una canasta de bienes y servicios
            - IPC Headline: incluye alimentos y energía (más volátil)
            - IPC Core (Subyacente): excluye alimentos y energía (más estable)
            - Meta de la Fed: ~2% anual
            
            **Interpretación**
            - < 2%: Inflación baja, posible estímulo monetario
            - 2-3%: Rango objetivo, economía saludable
            - > 3%: Inflación elevada, probable endurecimiento monetario
            - > 5%: Inflación alta, política monetaria muy restrictiva
            
            **Cambios Mensuales (MoM)**
            - Permiten ver tendencias más recientes
            - Más volátiles pero más actuales que YoY
            """)
        
        with st.expander("👥 Mercado Laboral - Guía"):
            st.markdown("""
            **Tasa de Desempleo**
            - % de la fuerza laboral que está desempleada
            - < 4%: Mercado laboral muy ajustado
            - 4-5%: Rango normal/saludable
            - > 6%: Mercado laboral débil
            
            **Nóminas No Agrícolas (NFP)**
            - Cambio mensual en empleos
            - > 200K: Creación fuerte de empleo
            - 100-200K: Crecimiento moderado
            - < 100K: Crecimiento débil
            - Negativo: Pérdida de empleos
            
            **JOLTS (Job Openings)**
            - Vacantes de empleo disponibles
            - Alto = empresas buscan contratar (economía fuerte)
            - Bajo = pocas oportunidades (economía débil)
            
            **Salarios**
            - Crecimiento acelerado puede presionar inflación
            - La Fed monitorea closely esta relación
            """)
        
        with st.expander("🏠 Sector Inmobiliario - Guía"):
            st.markdown("""
            **Case-Shiller Index**
            - Índice más importante de precios de vivienda
            - Cubre principales áreas metropolitanas de EE.UU.
            - Valor base = 100 en Enero 2000
            
            **Housing Starts**
            - Número de nuevas construcciones iniciadas (anualizado)
            - Indicador adelantado de actividad económica
            - Sensible a tasas de interés
            
            **Tasas Hipotecarias**
            - Determinan asequibilidad de viviendas
            - Generalmente siguen al Tesoro 10Y + spread
            - Subidas reducen demanda y pueden enfriar precios
            
            **Building Permits**
            - Permisos de construcción otorgados
            - Precede a Housing Starts (indicador adelantado)
            
            **Ventas de Viviendas**
            - New Home Sales: viviendas nuevas
            - Existing Home Sales: viviendas usadas
            - Miden demanda actual del mercado
            """)
        
        with st.expander("📊 Correlaciones - Guía"):
            st.markdown("""
            **R² (Coeficiente de Determinación)**
            - Mide qué tan bien la regresión explica la variación
            - Rango: 0 a 1
            - R² = 0.8: 80% de la variación es explicada
            - R² > 0.7: Correlación fuerte
            - R² 0.3-0.7: Correlación moderada
            - R² < 0.3: Correlación débil
            
            **Interpretación de Pendiente**
            - Positiva: Variables se mueven en la misma dirección
            - Negativa: Variables se mueven en direcciones opuestas
            - Empinada: Relación fuerte
            - Plana: Relación débil
            
            **Importante**
            - Correlación ≠ Causalidad
            - Relaciones históricas pueden cambiar
            - Usar múltiples indicadores para decisiones
            """)
        
        st.markdown("---")
        st.markdown("### 🎓 Conceptos Clave")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **YoY (Year-over-Year)**
            - Comparación con mismo período año anterior
            - Elimina estacionalidad
            - Más estable que MoM
            
            **MoM (Month-over-Month)**
            - Comparación con mes anterior
            - Más volátil pero más actual
            - Útil para tendencias recientes
            """)
        
        with col2:
            st.markdown("""
            **Política Restrictiva**
            - Fed sube tasas
            - Objetivo: frenar inflación
            - Efecto: enfría economía
            
            **Política Expansiva**
            - Fed baja tasas
            - Objetivo: estimular economía
            - Efecto: puede aumentar inflación
            """)
    
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

if __name__ == "__main__":
    main()
