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
    errors = []
    
    for name, series_id in series_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            if df is not None and not df.empty:
                dfs[name] = df.iloc[:, 0]
            else:
                errors.append(name)
        except Exception as e:
            errors.append(name)
    
    if len(errors) > 0 and len(errors) < len(series_dict):
        # Algunas series no disponibles pero otras sí
        pass
    elif len(errors) == len(series_dict):
        # Ninguna serie disponible
        st.warning(f"⚠️ No se pudieron obtener datos para el período seleccionado. Intente un rango de fechas diferente.")
        return None
    
    if dfs:
        combined = pd.DataFrame(dfs)
        combined = combined.dropna(how='all')
        if combined.empty:
            return None
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

def calculate_curve_metrics(df, front_col, back_col, lookback=10, threshold=1.0):
    """Calcular métricas de curva de rendimiento"""
    # Calcular spread
    df['curve'] = (df[back_col] - df[front_col]) * 100
    df['curve_smooth'] = df['curve'].rolling(window=2).mean()
    
    # Valores lookback
    df['curve_lookback'] = df['curve'].shift(lookback)
    df['front_lookback'] = df[front_col].shift(lookback)
    df['back_lookback'] = df[back_col].shift(lookback)
    
    # Clasificar movimientos
    df['bullsteepener'] = np.where(
        (df['curve'] > df['curve_lookback'] + threshold) &
        (df[front_col] < df['front_lookback']) &
        (df[back_col] < df['back_lookback']),
        df['curve'], 0
    )
    
    df['bearsteepener'] = np.where(
        (df['curve'] > df['curve_lookback'] + threshold) &
        (df[front_col] > df['front_lookback']) &
        (df[back_col] > df['back_lookback']),
        df['curve'], 0
    )
    
    df['bullflattener'] = np.where(
        (df['curve'] < df['curve_lookback'] - threshold) &
        (df[front_col] < df['front_lookback']) &
        (df[back_col] < df['back_lookback']),
        df['curve'], 0
    )
    
    df['bearflattener'] = np.where(
        (df['curve'] < df['curve_lookback'] - threshold) &
        (df[front_col] > df['front_lookback']) &
        (df[back_col] > df['back_lookback']),
        df['curve'], 0
    )
    
    df['steepenertwist'] = np.where(
        (df['curve'] > df['curve_lookback'] + threshold) &
        ((df[front_col] > df['front_lookback']) != (df[back_col] > df['back_lookback'])),
        df['curve'], 0
    )
    
    df['flattenertwist'] = np.where(
        (df['curve'] < df['curve_lookback'] - threshold) &
        ((df[front_col] > df['front_lookback']) != (df[back_col] > df['back_lookback'])),
        df['curve'], 0
    )
    
    return df

def create_curve_behavior_chart(df, title, front_leg, back_leg):
    """Crear gráfico de comportamiento de curva estilo Bloomberg"""
    fig = go.Figure()
    
    # Colores estilo Bloomberg
    colors = {
        'bullsteepener': '#00D000',
        'bearsteepener': '#D00000',
        'steepenertwist': '#FF00FF',
        'bullflattener': '#00D0D0',
        'bearflattener': '#D0D000',
        'flattenertwist': '#8B008B'
    }
    
    labels = {
        'bullsteepener': 'Bull Steepener',
        'bearsteepener': 'Bear Steepener',
        'steepenertwist': 'Steepener Twist',
        'bullflattener': 'Bull Flattener',
        'bearflattener': 'Bear Flattener',
        'flattenertwist': 'Flattener Twist'
    }
    
    # Agregar barras
    for indicator, color in colors.items():
        if indicator in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[indicator],
                name=labels[indicator],
                marker_color=color,
                opacity=0.95
            ))
    
    # Agregar línea de curva
    if 'curve_smooth' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['curve_smooth'],
            name='Curve',
            line=dict(color='white', width=2),
            mode='lines'
        ))
    
    front_display = '3M' if front_leg == 'DGS3MO' else front_leg.replace('DGS', '') + 'Y'
    back_display = back_leg.replace('DGS', '') + 'Y'
    
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{front_display}-{back_display} Spread</sub>", 
                  font=dict(size=18, color='#06b6d4')),
        xaxis_title="Fecha",
        yaxis_title="Basis Points",
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=True, gridcolor='#1a1a1a'),
        yaxis=dict(showgrid=True, gridcolor='#1a1a1a', range=[-160, 400]),
        barmode='stack',
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
        height=500
    )
    
    return fig

def create_bar_chart(df, title, y_title, color='#f59e0b'):
    if df is None or df.empty:
        return create_empty_chart(title)
    
    # Filtrar solo datos válidos
    df_clean = df.dropna(how='all')
    if df_clean.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    for col in df_clean.columns:
        if df_clean[col].notna().any():
            fig.add_trace(go.Bar(x=df_clean.index, y=df_clean[col], name=col, marker_color=color))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        xaxis_title="Fecha", yaxis_title=y_title,
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#334155', zeroline=True, zerolinecolor='#475569'),
        height=400
    )
    return fig

def create_line_chart(df, title, y_title, colors=None):
    if df is None or df.empty:
        return create_empty_chart(title)
    
    # Filtrar datos válidos
    df_clean = df.dropna(how='all')
    if df_clean.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    if colors is None:
        colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    has_data = False
    for i, col in enumerate(df_clean.columns):
        if df_clean[col].notna().any() and len(df_clean[col].dropna()) > 0:
            fig.add_trace(go.Scatter(
                x=df_clean.index, y=df_clean[col], name=col, mode='lines',
                line=dict(width=2, color=colors[i % len(colors)])
            ))
            has_data = True
    
    if not has_data:
        return create_empty_chart(title)
    
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

def create_area_chart(df, title, y_title, color='#06b6d4'):
    if df is None or df.empty:
        return create_empty_chart(title)
    
    # Filtrar datos válidos
    df_clean = df.dropna(how='all')
    if df_clean.empty:
        return create_empty_chart(title)
    
    fig = go.Figure()
    has_data = False
    for col in df_clean.columns:
        if df_clean[col].notna().any() and len(df_clean[col].dropna()) > 0:
            fig.add_trace(go.Scatter(
                x=df_clean.index, y=df_clean[col], name=col, mode='lines',
                fill='tozeroy', line=dict(width=2, color=color),
                fillcolor='rgba(6, 182, 212, 0.3)'
            ))
            has_data = True
    
    if not has_data:
        return create_empty_chart(title)
    
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

def create_combo_chart(df, bar_cols, line_cols, title, y1_title, y2_title=None):
    """Crear gráfico combinado barras + líneas"""
    if df is None or df.empty:
        return create_empty_chart(title)
    
    if y2_title:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    
    colors_bar = ['#f59e0b', '#ef4444']
    colors_line = ['#06b6d4', '#8b5cf6', '#10b981']
    
    # Barras
    for i, col in enumerate(bar_cols):
        if col in df.columns and df[col].notna().any():
            if y2_title:
                fig.add_trace(
                    go.Bar(x=df.index, y=df[col], name=col, 
                          marker_color=colors_bar[i % len(colors_bar)]),
                    secondary_y=False
                )
            else:
                fig.add_trace(go.Bar(x=df.index, y=df[col], name=col,
                                    marker_color=colors_bar[i % len(colors_bar)]))
    
    # Líneas
    for i, col in enumerate(line_cols):
        if col in df.columns and df[col].notna().any():
            if y2_title:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=col, mode='lines',
                              line=dict(width=3, color=colors_line[i % len(colors_line)])),
                    secondary_y=True
                )
            else:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines',
                                        line=dict(width=3, color=colors_line[i % len(colors_line)])))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=11),
        xaxis=dict(showgrid=False, gridcolor='#334155'),
        height=450
    )
    
    if y2_title:
        fig.update_yaxes(title_text=y1_title, secondary_y=False, showgrid=True, gridcolor='#334155')
        fig.update_yaxes(title_text=y2_title, secondary_y=True, showgrid=False)
    else:
        fig.update_yaxes(title_text=y1_title, showgrid=True, gridcolor='#334155')
    
    return fig

def create_empty_chart(title):
    fig = go.Figure()
    fig.add_annotation(
        text="📊 No hay datos disponibles<br><sub>Intente seleccionar un rango de fechas diferente</sub>",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, 
        font=dict(size=16, color='#94a3b8'),
        align="center"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#06b6d4')),
        plot_bgcolor='#0f172a', paper_bgcolor='#1e293b',
        height=400, 
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_curve_explanation_chart():
    """Crear gráfico explicativo de movimientos de curva"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Bull Flattening', 'Bear Flattening', 
                       'Bull Steepening', 'Bear Steepening'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Datos de ejemplo para las curvas
    x = [0, 1, 2]  # Short, Medium, Long
    
    # Bull Flattening: Front sube menos, back baja más
    y_initial_bf = [1, 2, 3]
    y_final_bf = [1.2, 2.1, 2.5]
    
    fig.add_trace(go.Scatter(x=x, y=y_initial_bf, mode='lines', 
                            line=dict(color='#94a3b8', width=2, dash='solid'),
                            showlegend=False, name='Inicial'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_final_bf, mode='lines',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            showlegend=False, name='Final'),
                 row=1, col=1)
    # Flechas
    fig.add_annotation(x=0, y=1, ax=0, ay=1.2, xref='x1', yref='y1', axref='x1', ayref='y1',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#10b981',
                      row=1, col=1)
    fig.add_annotation(x=2, y=3, ax=2, ay=2.5, xref='x1', yref='y1', axref='x1', ayref='y1',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#10b981',
                      row=1, col=1)
    
    # Bear Flattening: Front sube más, back sube menos
    y_initial_bearf = [1, 2, 3]
    y_final_bearf = [2, 2.8, 3.2]
    
    fig.add_trace(go.Scatter(x=x, y=y_initial_bearf, mode='lines',
                            line=dict(color='#94a3b8', width=2, dash='solid'),
                            showlegend=False),
                 row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y_final_bearf, mode='lines',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            showlegend=False),
                 row=1, col=2)
    fig.add_annotation(x=0, y=1, ax=0, ay=2, xref='x2', yref='y2', axref='x2', ayref='y2',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#ef4444',
                      row=1, col=2)
    fig.add_annotation(x=2, y=3, ax=2, ay=3.2, xref='x2', yref='y2', axref='x2', ayref='y2',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#ef4444',
                      row=1, col=2)
    
    # Bull Steepening: Front baja más, back baja menos
    y_initial_bs = [2, 2.5, 3]
    y_final_bs = [1, 1.8, 2.7]
    
    fig.add_trace(go.Scatter(x=x, y=y_initial_bs, mode='lines',
                            line=dict(color='#94a3b8', width=2, dash='solid'),
                            showlegend=False),
                 row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_final_bs, mode='lines',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            showlegend=False),
                 row=2, col=1)
    fig.add_annotation(x=0, y=2, ax=0, ay=1, xref='x3', yref='y3', axref='x3', ayref='y3',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#10b981',
                      row=2, col=1)
    fig.add_annotation(x=2, y=3, ax=2, ay=2.7, xref='x3', yref='y3', axref='x3', ayref='y3',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#10b981',
                      row=2, col=1)
    
    # Bear Steepening: Front sube menos, back sube más
    y_initial_bears = [1, 2, 3]
    y_final_bears = [1.3, 2.5, 3.8]
    
    fig.add_trace(go.Scatter(x=x, y=y_initial_bears, mode='lines',
                            line=dict(color='#94a3b8', width=2, dash='solid'),
                            showlegend=False),
                 row=2, col=2)
    fig.add_trace(go.Scatter(x=x, y=y_final_bears, mode='lines',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            showlegend=False),
                 row=2, col=2)
    fig.add_annotation(x=0, y=1, ax=0, ay=1.3, xref='x4', yref='y4', axref='x4', ayref='y4',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#ef4444',
                      row=2, col=2)
    fig.add_annotation(x=2, y=3, ax=2, ay=3.8, xref='x4', yref='y4', axref='x4', ayref='y4',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#ef4444',
                      row=2, col=2)
    
    # Layout
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    
    fig.update_layout(
        height=600,
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=14),
        showlegend=False,
        title=dict(
            text="Guía Visual: Movimientos de la Curva de Rendimiento",
            font=dict(size=20, color='#06b6d4', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[
            dict(text="🟢 = Tasas bajan | 🔴 = Tasas suben", 
                 xref="paper", yref="paper",
                 x=0.5, y=-0.05, showarrow=False,
                 font=dict(size=12, color='#94a3b8'))
        ]
    )
    
    return fig
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
        "📊 Análisis de Curvas",
        "📚 Guía"
    ])
    
    # TAB 1: POLÍTICA MONETARIA
    with tab1:
        st.markdown("## 💰 Panel de Política Monetaria")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Política Monetaria", 
                    "Análisis de las tasas de interés controladas por la Reserva Federal y los rendimientos de bonos del Tesoro.")
        
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
                    "Spread 10Y-2Y", "Spread (%)",
                    colors=['#8b5cf6']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No se pudieron cargar los datos de política monetaria. Por favor, intente con un rango de fechas diferente o verifique su conexión.")
    
    # TAB 2: INFLACIÓN & LABORAL
    with tab2:
        st.markdown("## 📈 Inflación y Mercado Laboral")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Inflación & Empleo", 
                    "Indicadores clave de precios y mercado laboral que la Fed monitorea para sus decisiones de política monetaria.")
        
        combined_series = {
            'CPIAUCSL': 'CPIAUCSL', 'CPILFESL': 'CPILFESL', 'PCEPI': 'PCEPI',
            'UNRATE': 'UNRATE', 'PAYEMS': 'PAYEMS',
            'CES0500000003': 'CES0500000003', 'JTSJOL': 'JTSJOL',
            'ICSA': 'ICSA', 'JTSQUL': 'JTSQUL', 'JTSLDL': 'JTSLDL'
        }
        
        with st.spinner('Cargando datos...'):
            data = fetch_multiple_series(combined_series, start_date, end_date)
        
        if data is not None and not data.empty:
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            yoy = data[['CPIAUCSL', 'CPILFESL', 'PCEPI']].pct_change(12) * 100
            
            with col1:
                st.metric("IPC YoY", safe_value(get_last_valid(yoy, 'CPIAUCSL'), "{:.2f}%"))
            with col2:
                st.metric("IPC Core YoY", safe_value(get_last_valid(yoy, 'CPILFESL'), "{:.2f}%"))
            with col3:
                st.metric("Desempleo", safe_value(get_last_valid(data, 'UNRATE'), "{:.1f}%"))
            with col4:
                st.metric("Nóminas", safe_value(get_last_valid(data, 'PAYEMS'), "{:.0f}K"))
            
            st.markdown("---")
            st.markdown("### 💹 Inflación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    yoy.dropna(),
                    "Inflación YoY", "% YoY",
                    colors=['#f59e0b', '#ef4444', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                mom = data[['CPIAUCSL', 'CPILFESL']].pct_change(1) * 100
                mom_clean = mom[['CPIAUCSL']].dropna()
                
                if not mom_clean.empty and len(mom_clean) > 0:
                    fig = create_bar_chart(
                        mom_clean.tail(60),
                        "IPC - Cambios Mensuales", "% MoM"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Datos de cambios mensuales no disponibles")
            
            st.markdown("### 👥 Mercado Laboral")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_area_chart(
                    data[['UNRATE']].dropna(),
                    "Tasa de Desempleo", "%",
                    color='#8b5cf6'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_area_chart(
                    data[['JTSJOL']].dropna(),
                    "Vacantes JOLTS", "Miles",
                    color='#06b6d4'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                nfp_chg = data['PAYEMS'].diff().dropna()
                
                if not nfp_chg.empty and len(nfp_chg) > 0:
                    fig = create_bar_chart(
                        pd.DataFrame({'NFP': nfp_chg}).tail(60),
                        "Cambios Mensuales NFP", "Miles",
                        color='#10b981'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Datos de cambios NFP no disponibles")
            
            with col2:
                fig = create_area_chart(
                    data[['ICSA']].dropna(),
                    "Solicitudes Iniciales Desempleo", "Miles",
                    color='#ef4444'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 JOLTS - Dinámica Laboral")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['JTSQUL']].dropna(),
                    "Renuncias JOLTS", "Miles",
                    colors=['#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_line_chart(
                    data[['JTSLDL']].dropna(),
                    "Despidos JOLTS", "Miles",
                    colors=['#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No se pudieron cargar los datos de inflación y mercado laboral. Por favor, intente con un rango de fechas diferente.")
    
    # TAB 3: SECTOR INMOBILIARIO
    with tab3:
        st.markdown("## 🏠 Sector Inmobiliario")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Sector Inmobiliario", 
                    "Precios de vivienda, actividad de construcción y financiamiento hipotecario.")
        
        re_series = {
            'CSUSHPISA': 'CSUSHPISA', 'HOUST': 'HOUST',
            'MORTGAGE30US': 'MORTGAGE30US', 'HSN1FNSA': 'HSN1FNSA',
            'PERMIT': 'PERMIT', 'MSPNHSUS': 'MSPNHSUS',
            'EXHOSLUSM495S': 'EXHOSLUSM495S', 'DGS10': 'DGS10',
            'USSTHPI': 'USSTHPI', 'PRRESCON': 'PRRESCON'
        }
        
        with st.spinner('Cargando datos...'):
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
                    data[['CSUSHPISA', 'USSTHPI']].dropna(how='all'),
                    "Índices de Precios", "Índice",
                    colors=['#10b981', '#06b6d4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'CSUSHPISA' in data.columns:
                    cs_yoy = data['CSUSHPISA'].pct_change(12) * 100
                    cs_yoy_clean = cs_yoy.dropna()
                    
                    if not cs_yoy_clean.empty and len(cs_yoy_clean) > 0:
                        fig = create_line_chart(
                            pd.DataFrame({'CS YoY': cs_yoy_clean}),
                            "Case-Shiller YoY", "% YoY",
                            colors=['#06b6d4']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Datos de Case-Shiller YoY no disponibles")
                else:
                    st.info("Datos de Case-Shiller no disponibles")
            
            st.markdown("### 🏗️ Actividad de Construcción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    data[['HOUST', 'PERMIT']].dropna(how='all'),
                    "Inicios y Permisos", "Miles",
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
            
            st.markdown("### 💰 Tasas y Asequibilidad")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'HOUST' in data.columns and 'MORTGAGE30US' in data.columns:
                    fig = create_dual_axis_chart(
                        data, 'HOUST', 'MORTGAGE30US',
                        "Construcción vs Hipotecas",
                        "Starts (K)", "Tasa (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'MSPNHSUS' in data.columns:
                    fig = create_line_chart(
                        data[['MSPNHSUS']].dropna(),
                        "Precio Medio Vivienda Nueva", "USD",
                        colors=['#ec4899']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Gasto en construcción
            if 'PRRESCON' in data.columns:
                st.markdown("### 🔨 Gasto en Construcción")
                fig = create_area_chart(
                    data[['PRRESCON']].dropna(),
                    "Gasto en Construcción Residencial", "Millones USD",
                    color='#3b82f6'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No se pudieron cargar los datos del sector inmobiliario. Por favor, intente con un rango de fechas diferente.")
    
    # TAB 4: ANÁLISIS DE CURVAS
    with tab4:
        st.markdown("## 📊 Análisis de Curvas de Rendimiento")
        
        with st.expander("ℹ️ ¿Qué muestra esta sección?"):
            info_box("Comportamiento de Curva", 
                    "Análisis avanzado de los movimientos de la curva de rendimientos. "
                    "Bull/Bear Steepener/Flattener indican diferentes escenarios económicos y expectativas del mercado.")
        
        # Agregar guía visual
        st.markdown("### 📚 Guía Visual de Movimientos")
        
        with st.expander("👁️ Ver Guía Visual (Recomendado para nuevos usuarios)", expanded=False):
            st.markdown("""
            **Leyenda:**
            - **Línea sólida gris:** Curva inicial
            - **Línea punteada naranja:** Curva final
            - **Flechas verdes (🟢):** Tasas bajando
            - **Flechas rojas (🔴):** Tasas subiendo
            
            **Interpretación:**
            - **Flattening:** El spread se reduce (curva se aplana)
            - **Steepening:** El spread aumenta (curva se empina)
            - **Bull:** Movimiento con tasas bajando (bueno para bonos)
            - **Bear:** Movimiento con tasas subiendo (malo para bonos)
            """)
            
            fig_guide = create_curve_explanation_chart()
            st.plotly_chart(fig_guide, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configurar Análisis de Curva")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            front_leg = st.selectbox(
                "Tenor Corto",
                options=['DGS3MO', 'DGS2', 'DGS5'],
                index=0,
                format_func=lambda x: {'DGS3MO': '3 Meses', 'DGS2': '2 Años', 'DGS5': '5 Años'}[x]
            )
        
        with col2:
            back_leg = st.selectbox(
                "Tenor Largo",
                options=['DGS10', 'DGS30'],
                index=1,
                format_func=lambda x: {'DGS10': '10 Años', 'DGS30': '30 Años'}[x]
            )
        
        with col3:
            lookback = st.slider("Lookback (días)", 5, 30, 10)
        
        # Obtener datos de curva
        curve_series = {
            front_leg: front_leg,
            back_leg: back_leg
        }
        
        with st.spinner('Analizando curva de rendimiento...'):
            curve_data = fetch_multiple_series(curve_series, start_date, end_date)
        
        if curve_data is not None and not curve_data.empty:
            # Calcular métricas
            curve_data = calculate_curve_metrics(curve_data, front_leg, back_leg, lookback)
            
            # Gráfico principal de comportamiento
            fig = create_curve_behavior_chart(curve_data, "Comportamiento de la Curva", front_leg, back_leg)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📈 Análisis Complementario")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    curve_data[[front_leg, back_leg]].dropna(how='all'),
                    "Rendimientos por Tenor", "Rendimiento (%)",
                    colors=['#06b6d4', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'curve' in curve_data.columns:
                    fig = create_line_chart(
                        pd.DataFrame({'Spread': curve_data['curve']/100}).dropna(),
                        "Evolución del Spread", "Spread (%)",
                        colors=['#8b5cf6']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Curvas Históricas Adicionales")
        
        # Otros spreads importantes
        other_series = {
            'DGS2': 'DGS2',
            'DGS10': 'DGS10',
            'DGS30': 'DGS30',
            'FEDFUNDS': 'FEDFUNDS'
        }
        
        with st.spinner('Cargando spreads adicionales...'):
            other_data = fetch_multiple_series(other_series, start_date, end_date)
        
        if other_data is not None and not other_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_line_chart(
                    other_data[['DGS2', 'DGS10', 'DGS30']].dropna(how='all'),
                    "Principales Rendimientos del Tesoro", "Rendimiento (%)",
                    colors=['#06b6d4', '#8b5cf6', '#ec4899']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                spreads = pd.DataFrame({
                    '10Y-2Y': other_data['DGS10'] - other_data['DGS2'],
                    '30Y-10Y': other_data['DGS30'] - other_data['DGS10']
                }).dropna()
                
                fig = create_line_chart(
                    spreads,
                    "Spreads Clave", "Spread (%)",
                    colors=['#f59e0b', '#10b981']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No hay datos adicionales de spreads disponibles para este rango de fechas.")
    
    # TAB 5: GUÍA
    with tab5:
        st.markdown("## 📚 Guía de Interpretación")
        
        with st.expander("💰 Política Monetaria"):
            st.markdown("""
            **Fed Funds Rate:** Tasa de interés de referencia. Subidas = política restrictiva, Bajadas = política expansiva.
            
            **Rendimientos del Tesoro:** DGS2 (corto plazo), DGS10 (largo plazo), DGS30 (muy largo plazo).
            
            **Spread 10Y-2Y:** Positivo = economía saludable, Negativo = señal de recesión.
            """)
        
        with st.expander("📈 Inflación"):
            st.markdown("""
            **IPC:** Cambio en precios de bienes y servicios. Meta de la Fed: ~2% anual.
            
            **Interpretación:** < 2% = baja, 2-3% = objetivo, > 3% = elevada, > 5% = alta.
            
            **MoM vs YoY:** MoM más volátil pero actual, YoY más estable.
            """)
        
        with st.expander("👥 Mercado Laboral"):
            st.markdown("""
            **Desempleo:** < 4% = ajustado, 4-5% = normal, > 6% = débil.
            
            **NFP:** > 200K = fuerte, 100-200K = moderado, < 100K = débil.
            
            **JOLTS:** Vacantes disponibles. Alto = economía fuerte.
            """)
        
        with st.expander("🏠 Sector Inmobiliario"):
            st.markdown("""
            **Case-Shiller:** Índice principal de precios (base 100 en 2000).
            
            **Housing Starts:** Indicador adelantado de actividad económica.
            
            **Tasas Hipotecarias:** Siguen al Tesoro 10Y + spread. Afectan asequibilidad.
            """)
        
        with st.expander("📊 Curvas de Rendimiento - Guía Detallada"):
            st.markdown("""
            ### Movimientos Principales de la Curva
            
            #### 🟢 Bull Flattening (Aplanamiento Alcista)
            - **Qué pasa:** El spread entre tasas largas y cortas se reduce
            - **Movimiento:** Tasas cortas SUBEN menos, tasas largas BAJAN más
            - **Significado:** Expectativas de desaceleración económica
            - **Escenario:** La Fed puede estar terminando un ciclo de subidas
            
            #### 🔴 Bear Flattening (Aplanamiento Bajista)
            - **Qué pasa:** El spread se reduce con tasas subiendo
            - **Movimiento:** Tasas cortas SUBEN más, tasas largas SUBEN menos
            - **Significado:** La Fed está endureciendo agresivamente
            - **Escenario:** Política monetaria restrictiva (combatir inflación)
            
            #### 🟢 Bull Steepening (Empinamiento Alcista)
            - **Qué pasa:** El spread aumenta con tasas bajando
            - **Movimiento:** Tasas cortas BAJAN más, tasas largas BAJAN menos
            - **Significado:** Expectativas de estímulo monetario agresivo
            - **Escenario:** La Fed está recortando tasas (recesión o crisis)
            
            #### 🔴 Bear Steepening (Empinamiento Bajista)
            - **Qué pasa:** El spread aumenta con tasas subiendo
            - **Movimiento:** Tasas cortas SUBEN menos, tasas largas SUBEN más
            - **Significado:** Expectativas de inflación a largo plazo
            - **Escenario:** Preocupaciones sobre déficit o inflación futura
            
            #### 🔄 Twists (Torsiones)
            - **Qué pasa:** Movimientos mixtos no clasificables
            - **Significado:** Incertidumbre o transición entre escenarios
            
            ### ¿Cómo Interpretar?
            
            **"Bull" vs "Bear":**
            - Bull (🟢) = Tasas bajando = Bueno para bonos = Preocupación económica
            - Bear (🔴) = Tasas subiendo = Malo para bonos = Fortaleza económica o inflación
            
            **"Steepening" vs "Flattening":**
            - Steepening = Spread aumenta = Mayor diferencia entre corto y largo plazo
            - Flattening = Spread disminuye = Menor diferencia entre corto y largo plazo
            
            **Curva Invertida (Caso Especial):**
            - Cuando tasas cortas > tasas largas (spread negativo)
            - Históricamente precede recesiones
            - Señal de alerta importante
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
