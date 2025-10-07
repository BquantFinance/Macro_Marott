import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pandas_datareader import data as web
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="FRED Economic Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode aesthetic
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #06b6d4;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Headers */
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
    
    /* Tabs */
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
    
    /* Info boxes */
    .stAlert {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #06b6d4, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date, end_date):
    """Fetch data from FRED using pandas_datareader"""
    try:
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        return df
    except Exception as e:
        st.warning(f"Could not fetch {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_multiple_series(series_dict, start_date, end_date):
    """Fetch multiple FRED series and combine them"""
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
    """Create a beautiful line chart with Plotly"""
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
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Date",
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
    """Create a beautiful area chart"""
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
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#06b6d4', family='Arial Black')),
        xaxis_title="Date",
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
    """Create a dual-axis chart"""
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
    
    fig.update_yaxis(title_text=y1_title, secondary_y=False, 
                     showgrid=True, gridcolor='#334155')
    fig.update_yaxis(title_text=y2_title, secondary_y=True,
                     showgrid=False)
    
    return fig

# Main App
def main():
    # Header
    st.markdown("<h1>📊 FRED Economic Analytics Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Federal Reserve Economic Data Visualization Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Date range selector
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years
        
        date_range = st.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        st.markdown("---")
        st.markdown("### 📌 Data Sources")
        st.markdown("• Federal Reserve Economic Data (FRED)")
        st.markdown("• U.S. Treasury")
        st.markdown("• Bureau of Labor Statistics")
        st.markdown("• S&P Dow Jones Indices")
        
        st.markdown("---")
        st.markdown("### 🔄 Last Updated")
        st.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Monetary Policy",
        "📈 Inflation Metrics", 
        "👥 Labor Market",
        "🏠 Real Estate",
        "📊 Yield Curves"
    ])
    
    # TAB 1: MONETARY POLICY
    with tab1:
        st.markdown("## 💰 Monetary Policy Dashboard")
        
        # Fetch data
        monetary_series = {
            'Fed Funds Rate': 'FEDFUNDS',
            '2Y Treasury': 'DGS2',
            '10Y Treasury': 'DGS10',
            '30Y Treasury': 'DGS30'
        }
        
        with st.spinner('Loading monetary policy data...'):
            monetary_data = fetch_multiple_series(monetary_series, start_date, end_date)
        
        if monetary_data is not None:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            latest = monetary_data.iloc[-1]
            prev = monetary_data.iloc[-30] if len(monetary_data) > 30 else monetary_data.iloc[0]
            
            with col1:
                change = latest['Fed Funds Rate'] - prev['Fed Funds Rate']
                st.metric("Fed Funds Rate", 
                         f"{latest['Fed Funds Rate']:.2f}%",
                         f"{change:+.2f}%")
            
            with col2:
                change = latest['2Y Treasury'] - prev['2Y Treasury']
                st.metric("2-Year Treasury", 
                         f"{latest['2Y Treasury']:.2f}%",
                         f"{change:+.2f}%")
            
            with col3:
                change = latest['10Y Treasury'] - prev['10Y Treasury']
                st.metric("10-Year Treasury", 
                         f"{latest['10Y Treasury']:.2f}%",
                         f"{change:+.2f}%")
            
            with col4:
                spread = latest['10Y Treasury'] - latest['2Y Treasury']
                st.metric("10Y-2Y Spread", 
                         f"{spread:.2f}%",
                         "Inverted" if spread < 0 else "Normal",
                         delta_color="inverse")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    monetary_data[['Fed Funds Rate', '2Y Treasury', '10Y Treasury']],
                    "📊 Interest Rates Over Time",
                    "Rate (%)"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Calculate spread
                spread_df = pd.DataFrame({
                    '10Y-2Y Spread': monetary_data['10Y Treasury'] - monetary_data['2Y Treasury']
                })
                fig2 = create_area_chart(
                    spread_df,
                    "📉 Yield Curve Spread (10Y-2Y)",
                    "Spread (%)",
                    color='#06b6d4'
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 2: INFLATION
    with tab2:
        st.markdown("## 📈 Inflation Metrics Dashboard")
        
        inflation_series = {
            'CPI (All Items)': 'CPIAUCSL',
            'Core CPI': 'CPILFESL',
            'PCE': 'PCEPI',
            'Core PCE': 'PCEPILFE'
        }
        
        with st.spinner('Loading inflation data...'):
            inflation_data = fetch_multiple_series(inflation_series, start_date, end_date)
        
        if inflation_data is not None:
            # Calculate YoY changes
            inflation_yoy = inflation_data.pct_change(12) * 100
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest_yoy = inflation_yoy.iloc[-1]
            
            with col1:
                st.metric("CPI YoY", 
                         f"{latest_yoy['CPI (All Items)']:.2f}%",
                         f"{latest_yoy['CPI (All Items)'] - inflation_yoy.iloc[-13]['CPI (All Items)']:+.2f}%")
            
            with col2:
                st.metric("Core CPI YoY", 
                         f"{latest_yoy['Core CPI']:.2f}%",
                         f"{latest_yoy['Core CPI'] - inflation_yoy.iloc[-13]['Core CPI']:+.2f}%")
            
            with col3:
                st.metric("PCE YoY", 
                         f"{latest_yoy['PCE']:.2f}%",
                         f"{latest_yoy['PCE'] - inflation_yoy.iloc[-13]['PCE']:+.2f}%")
            
            with col4:
                st.metric("Core PCE YoY", 
                         f"{latest_yoy['Core PCE']:.2f}%",
                         f"{latest_yoy['Core PCE'] - inflation_yoy.iloc[-13]['Core PCE']:+.2f}%")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    inflation_data,
                    "📊 Inflation Indices (Level)",
                    "Index Level"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_line_chart(
                    inflation_yoy,
                    "📈 Inflation Rate (YoY % Change)",
                    "YoY Change (%)",
                    colors=['#f59e0b', '#ef4444', '#ec4899', '#8b5cf6']
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 3: LABOR MARKET
    with tab3:
        st.markdown("## 👥 Labor Market Dashboard")
        
        labor_series = {
            'Unemployment Rate': 'UNRATE',
            'Nonfarm Payrolls': 'PAYEMS',
            'Labor Force Participation': 'CIVPART',
            'Initial Claims': 'ICSA'
        }
        
        with st.spinner('Loading labor market data...'):
            labor_data = fetch_multiple_series(labor_series, start_date, end_date)
        
        if labor_data is not None:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest = labor_data.iloc[-1]
            prev = labor_data.iloc[-2]
            
            with col1:
                change = latest['Unemployment Rate'] - prev['Unemployment Rate']
                st.metric("Unemployment Rate", 
                         f"{latest['Unemployment Rate']:.1f}%",
                         f"{change:+.1f}%",
                         delta_color="inverse")
            
            with col2:
                change = latest['Nonfarm Payrolls'] - prev['Nonfarm Payrolls']
                st.metric("Nonfarm Payrolls", 
                         f"{latest['Nonfarm Payrolls']:.0f}K",
                         f"{change:+.0f}K")
            
            with col3:
                st.metric("Labor Force Participation", 
                         f"{latest['Labor Force Participation']:.1f}%",
                         f"{latest['Labor Force Participation'] - prev['Labor Force Participation']:+.1f}%")
            
            with col4:
                st.metric("Initial Claims", 
                         f"{latest['Initial Claims']:.0f}K",
                         f"{latest['Initial Claims'] - prev['Initial Claims']:+.0f}K",
                         delta_color="inverse")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_area_chart(
                    labor_data[['Unemployment Rate']],
                    "📉 Unemployment Rate",
                    "Rate (%)",
                    color='#8b5cf6'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_line_chart(
                    labor_data[['Nonfarm Payrolls']],
                    "📊 Nonfarm Payrolls",
                    "Thousands",
                    colors=['#10b981']
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 4: REAL ESTATE
    with tab4:
        st.markdown("## 🏠 Real Estate Dashboard")
        
        real_estate_series = {
            'Case-Shiller Index': 'CSUSHPISA',
            'Housing Starts': 'HOUST',
            '30Y Mortgage Rate': 'MORTGAGE30US',
            'Median Home Price': 'MSPNHSUS',
            'Home Ownership Rate': 'RHORUSQ156N'
        }
        
        with st.spinner('Loading real estate data...'):
            re_data = fetch_multiple_series(real_estate_series, start_date, end_date)
        
        if re_data is not None:
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            latest = re_data.iloc[-1]
            prev_year = re_data.iloc[-12] if len(re_data) > 12 else re_data.iloc[0]
            
            with col1:
                yoy_change = ((latest['Case-Shiller Index'] / prev_year['Case-Shiller Index']) - 1) * 100
                st.metric("Case-Shiller Index", 
                         f"{latest['Case-Shiller Index']:.2f}",
                         f"{yoy_change:+.2f}% YoY")
            
            with col2:
                st.metric("30Y Mortgage Rate", 
                         f"{latest['30Y Mortgage Rate']:.2f}%",
                         f"{latest['30Y Mortgage Rate'] - prev_year['30Y Mortgage Rate']:+.2f}%")
            
            with col3:
                st.metric("Housing Starts", 
                         f"{latest['Housing Starts']:.0f}K",
                         f"{latest['Housing Starts'] - prev_year['Housing Starts']:+.0f}K")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_line_chart(
                    re_data[['Case-Shiller Index']],
                    "🏘️ Case-Shiller Home Price Index",
                    "Index (Jan 2000 = 100)",
                    colors=['#10b981']
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_dual_axis_chart(
                    re_data,
                    'Housing Starts',
                    '30Y Mortgage Rate',
                    "🏗️ Housing Starts vs Mortgage Rates",
                    "Housing Starts (K)",
                    "Mortgage Rate (%)"
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 5: YIELD CURVES
    with tab5:
        st.markdown("## 📊 Treasury Yield Curves")
        
        yield_series = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        with st.spinner('Loading yield curve data...'):
            yield_data = fetch_multiple_series(yield_series, start_date, end_date)
        
        if yield_data is not None:
            # Current yield curve
            st.markdown("### Current Yield Curve")
            
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
                hovertemplate='<b>%{x}</b><br>Yield: %{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text="📈 Treasury Yield Curve (Latest)", 
                          font=dict(size=24, color='#06b6d4', family='Arial Black')),
                xaxis_title="Maturity",
                yaxis_title="Yield (%)",
                plot_bgcolor='#0f172a',
                paper_bgcolor='#1e293b',
                font=dict(color='#e2e8f0', size=14),
                xaxis=dict(showgrid=True, gridcolor='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#334155'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical spreads
            st.markdown("### Historical Yield Spreads")
            
            spreads_df = pd.DataFrame({
                '10Y-2Y': yield_data['10Y'] - yield_data['2Y'],
                '10Y-3M': yield_data['10Y'] - yield_data['3M'],
                '30Y-5Y': yield_data['30Y'] - yield_data['5Y']
            })
            
            fig2 = create_line_chart(
                spreads_df,
                "📊 Treasury Yield Spreads",
                "Spread (%)",
                colors=['#06b6d4', '#8b5cf6', '#ec4899']
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
