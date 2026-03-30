import os
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- DIRECTORY ANCHOR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# --- CONSTANTS & INSTITUTIONAL PALETTE ---
NBER_RECESSION_PERIODS = [
    (pd.to_datetime('1990-07-01'), pd.to_datetime('1991-03-01')),
    (pd.to_datetime('2001-03-01'), pd.to_datetime('2001-11-01')),
    (pd.to_datetime('2007-12-01'), pd.to_datetime('2009-06-01')),
    (pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-01'))
]

# Terminal Colors
BG_COLOR = "#0B0F19" # Deep Char-Blue Institutional Background
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
TEXT_COLOR = "#A0AEC0"
COLOR_PRIMARY = "#00D2FF"   # Neon Cyan
COLOR_WARN = "#FF3366"      # Crimson Red
COLOR_YIELD = "#FFD700"     # Gold
COLOR_LIQ = "#00E676"       # Emerald Green
COLOR_CREDIT = "#B388FF"    # Soft Purple

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AQI Genesis Terminal", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to force the Terminal Aesthetic
st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG_COLOR}; }}
    div[data-testid="stMetricValue"] {{ color: #FFFFFF; font-weight: 700; font-size: 28px; }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_COLOR}; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ color: {TEXT_COLOR}; font-weight: 600; padding: 10px 20px; }}
    .stTabs [aria-selected="true"] {{ color: {COLOR_PRIMARY} !important; border-bottom-color: {COLOR_PRIMARY} !important; }}
    </style>
""", unsafe_allow_html=True)

# --- 1.5 SECURITY GATEWAY (THE PAYWALL) ---
TERMINAL_KEY = "GENESIS-APR-928" # Change this on the 1st of every month

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # The Login Screen
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background-color: #121826; padding: 40px; border-radius: 8px; border-top: 6px solid {COLOR_PRIMARY}; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.5);">
            <h2 style="color: white; margin-bottom: 5px;">AQI GENESIS</h2>
            <p style="color: {TEXT_COLOR}; margin-bottom: 25px;">Institutional Terminal Access</p>
        </div>
        """, unsafe_allow_html=True)
        
        entered_key = st.text_input("Enter Terminal Access Key", type="password", placeholder="Access Key")
        
        if st.button("Authenticate", use_container_width=True):
            if entered_key == TERMINAL_KEY:
                st.session_state["authenticated"] = True
                st.rerun() # Refreshes the page to show the dashboard
            else:
                st.error("❌ Invalid Access Key. Check the bottom of this week's Substack briefing.")
                
    st.stop() # This entirely stops the rest of the script from loading if they aren't logged in!

# --- 2. DATA INGESTION ---
@st.cache_data(ttl=300) 
def load_data():
    try:
        # Anchored Database Path
        db_path = os.path.join(BASE_DIR, "aqi_saas_backend.db")
        conn = sqlite3.connect(db_path)
        
        df = pd.read_sql("SELECT * FROM macro_state", conn, index_col='index')
        df.index = pd.to_datetime(df.index)
        
        forecast_df = pd.read_sql("SELECT * FROM macro_forecast", conn, index_col='index')
        forecast_df.index = pd.to_datetime(forecast_df.index)
        
        ai_state = pd.read_sql("SELECT * FROM ai_system_state", conn).iloc[0]
        conn.close()
        return df, forecast_df, ai_state
    except Exception as e:
        st.error(f"Failed to connect to SaaS Database at {db_path}. Error: {e}")
        return None, None, None

df, forecast_df, ai_state = load_data()

# --- HELPER: SHARED AESTHETICS ---
def apply_shared_aesthetics(fig, df_plot, last_hist_date, end_date, pulse_intensity, num_rows):
    """Applies recession bands, today line, minimal grids, and regime shading to all subplots."""
    fig.update_xaxes(type="date", showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)
    
    # 1. Subtle Recession Bars
    for start, end in NBER_RECESSION_PERIODS:
        if start < last_hist_date and end > df_plot.index.min():
            v_start = max(start, df_plot.index.min()).strftime('%Y-%m-%d')
            v_end = min(end, last_hist_date).strftime('%Y-%m-%d')
            for r in range(1, num_rows + 1):
                fig.add_vrect(x0=v_start, x1=v_end, fillcolor="#2D3748", opacity=0.3, layer="below", line_width=0, row=r, col=1)
                
    # 2. Glowing Today Line
    today_str = last_hist_date.strftime('%Y-%m-%d')
    for r in range(1, num_rows + 1):
        fig.add_vline(x=today_str, line_width=1.5, line_dash="dot", line_color="rgba(255,255,255,0.4)", row=r, col=1)

    # 3. Dynamic Regime Color Coding (Refined Opacity)
    if pulse_intensity > 75:
        regime_bg_color = "rgba(255, 51, 102, 0.1)"  # Crimson
    elif pulse_intensity > 50:
        regime_bg_color = "rgba(255, 215, 0, 0.05)"  # Gold
    else:
        regime_bg_color = "rgba(0, 230, 118, 0.05)"  # Emerald
        
    regime_start = last_hist_date - pd.DateOffset(months=6)
    regime_start_str = regime_start.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    for r in range(1, num_rows + 1): 
        fig.add_vrect(x0=regime_start_str, x1=end_date_str, fillcolor=regime_bg_color, opacity=1, layer="below", line_width=0, row=r, col=1)
        
    fig.add_annotation(x=regime_start_str, y=1.05, yref="paper", text="<b>FORWARD REGIME</b>", 
                       showarrow=False, font=dict(size=10, color=TEXT_COLOR), xanchor="left", row=1, col=1)
    
    # Universal Layout Updates
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, Roboto, Arial", color=TEXT_COLOR),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    return fig


if df is not None:
    df_plot = df[df.index > '2000-01-01']
    last_hist_date = df_plot.index[-1]
    start_date = df_plot.index[0]
    unemp_fc = forecast_df['unrate_forecast'] if not forecast_df.empty else None
    end_date = unemp_fc.index[-1] if unemp_fc is not None else last_hist_date

    # --- 3. TOP NAVIGATION & HUD ---
    st.markdown(f"""
        <div style="background-color: #121826; padding: 20px; border-radius: 8px; border-left: 6px solid {COLOR_PRIMARY}; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h2 style="margin:0; color: #FFFFFF; font-weight: 800; letter-spacing: 1px;">AQI GENESIS: INSTITUTIONAL TERMINAL</h2>
            <p style="margin:5px 0 0 0; color: {COLOR_PRIMARY}; font-size: 14px; font-weight: 500;">Live Interactive State Space & Macro Routing Model</p>
        </div>
    """, unsafe_allow_html=True)

    lookback = -4 if len(df) >= 4 else 0
    delta_pulse = df['PULSE_INTENSITY'].iloc[-1] - df['PULSE_INTENSITY'].iloc[lookback]
    delta_prob = df['recession_prob'].iloc[-1] - df['recession_prob'].iloc[lookback]
    delta_liq = (df['LIQUIDITY_ROC_3M'].iloc[-1] - df['LIQUIDITY_ROC_3M'].iloc[lookback]) * 100
    delta_spread = df['CREDIT_SPREAD'].iloc[-1] - df['CREDIT_SPREAD'].iloc[lookback]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cole Pulse Intensity", f"{ai_state['pulse_intensity']:.1f} / 100", f"{delta_pulse:+.1f} (3M)", delta_color="inverse")
    col2.metric("Recession Probability", f"{ai_state['recession_prob']:.1f}%", f"{delta_prob:+.1f}% (3M)", delta_color="inverse")
    col3.metric("Net Liquidity (3M ROC)", f"{ai_state['liquidity_roc']*100:.2f}%", f"{delta_liq:+.2f}% (3M)", delta_color="normal")
    col4.metric("High Yield Spread", f"{ai_state['credit_spread']:.2f} bps", f"{delta_spread:+.2f} bps (3M)", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # --- 4. INTERACTIVE TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 MACRO & LIQUIDITY", "🌊 THE COLE PULSE", "🧊 3D KINEMATICS"])

    # ==========================================
    # TAB 1: MACRO & LIQUIDITY
    # ==========================================
    with tab1:
        fig_macro = make_subplots(
            rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
            subplot_titles=(
                "<b>LABOR</b> | <span style='color:#E2E8F0'>Unemployment</span> & <span style='color:#ff4444'>Forecast</span>", 
                "<b>POLICY</b> | <span style='color:#00D2FF'>Fed Funds</span> vs <span style='color:#FFD700'>CPI YoY</span>", 
                "<b>LIQUIDITY</b> | <span style='color:#00E676'>Net Liquidity</span> vs <span style='color:#B388FF'>HY Spread</span>", 
                "<b>RISK</b> | <span style='color:#FF9800'>VIX</span> vs <span style='color:#00D2FF'>AUD/USD</span>", 
                "<b>FX</b> | <span style='color:#A0AEC0'>Dollar (DXY)</span> vs <span style='color:#B388FF'>USD/JPY</span>", 
                "<b>DEMOGRAPHICS</b> | <span style='color:#00D2FF'>US Emp Ratio</span> vs <span style='color:#FFD700'>AU Emp Ratio</span>",
                "<b>GROWTH</b> | <span style='color:#00D2FF'>US GDP (YoY)</span> vs <span style='color:#FFD700'>AU GDP (YoY)</span>" 
            ),
            specs=[[{"secondary_y": True}]] * 7
        )
        
        # 1. LABOR
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['UNRATE'], name="Unemployment", line=dict(color='#E2E8F0', width=2)), row=1, col=1)
        if unemp_fc is not None:
            fig_macro.add_trace(go.Scatter(x=unemp_fc.index, y=unemp_fc, name="AQI Forecast", mode='lines+markers', line=dict(color=COLOR_WARN, dash='dot', width=2.5), marker=dict(symbol='circle', size=6, color=BG_COLOR, line=dict(color=COLOR_WARN, width=2))), row=1, col=1)
        
        # 2. POLICY
        funds, cpi = df_plot['FED_FUNDS'], df_plot['CPI_YOY']
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=cpi, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=np.where(funds >= cpi, funds, cpi), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Restrictive"), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=funds, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=np.where(cpi > funds, cpi, funds), fill='tonexty', fillcolor='rgba(255, 51, 102, 0.1)', line=dict(width=0), name="Accommodative"), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=funds, name="Fed Funds", line=dict(color=COLOR_PRIMARY, width=2)), row=2, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=cpi, name="CPI YoY", line=dict(color=COLOR_YIELD, dash='dot', width=2)), row=2, col=1)
        
        # 3. LIQUIDITY
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['NET_LIQUIDITY'], name="Net Liquidity", line=dict(color=COLOR_LIQ, width=2)), row=3, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['CREDIT_SPREAD'], name="HY Spread", line=dict(color=COLOR_CREDIT, width=2)), row=3, col=1, secondary_y=True)
        
        # 4. RISK
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VIX'], name="VIX", line=dict(color='#FF9800', width=2)), row=4, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AUD_USD'], name="AUD/USD", line=dict(color=COLOR_PRIMARY, width=2)), row=4, col=1, secondary_y=True)
        
        # 5. FX
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DXY'], name="DXY", line=dict(color='#A0AEC0', width=2)), row=5, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['USD_JPY'], name="USD/JPY", line=dict(color=COLOR_CREDIT, width=2)), row=5, col=1, secondary_y=True)

        # 6. DEMOGRAPHICS
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['US_EMP_WORKING_AGE_RATIO'], name="US Emp Ratio", line=dict(color=COLOR_PRIMARY, width=2)), row=6, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AU_EMP_WORKING_AGE_RATIO'], name="AU Emp Ratio", line=dict(color=COLOR_YIELD, dash='dot', width=2)), row=6, col=1)

        # 7. GDP
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['US_GDP_YOY'], name="US GDP (YoY %)", line=dict(color=COLOR_PRIMARY, width=2)), row=7, col=1)
        fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AU_GDP_YOY'], name="AU GDP (YoY %)", line=dict(color=COLOR_YIELD, width=2)), row=7, col=1)
        fig_macro.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=7, col=1)

        fig_macro = apply_shared_aesthetics(fig_macro, df_plot, last_hist_date, end_date, ai_state['pulse_intensity'], 7)
        fig_macro.update_layout(height=1800, margin=dict(l=40, r=40, t=60, b=40), showlegend=False)
        # Style subplot titles
        for annotation in fig_macro['layout']['annotations']: annotation['font'] = dict(size=12, color=TEXT_COLOR)
        st.plotly_chart(fig_macro, use_container_width=True)

    # ==========================================
    # TAB 2: THE PHYSICS ENGINE
    # ==========================================
    with tab2:
        fig_physics = make_subplots(
            rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
            subplot_titles=(
                "<b>PULSE INTENSITY SCORE</b> | <span style='color:#FFD700'>Systemic Fragility</span>", 
                "<b>KINEMATIC POSITION</b> | <span style='color:#00D2FF'>Base Labor</span> vs <span style='color:#FF3366'>Sahm Rule</span>", 
                "<b>PULSE MOMENTUM</b> | 3-Month Trend", 
                "<b>PROBABILITY SIGNAL</b> | <span style='color:#B388FF'>Recession Risk %</span>", 
                "<b>KINETIC ENERGY</b> | <span style='color:#FF9800'>System Mass x Velocity²</span>", 
                "<b>RISK VELOCITY</b> | <span style='color:#00D2FF'>Labor Deterioration</span>", 
                "<b>SHOCK MAGNITUDE</b> | <span style='color:#FF3366'>Jerk Z-Score</span>",
                "<b>INTENSITY CONTEXT</b> | <span style='color:#FF3366'>Current Velocity</span> vs Benchmarks"
            )
        )
        
        # 1. Intensity Score
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['PULSE_INTENSITY'], name="Intensity", line=dict(color=COLOR_YIELD, width=2), fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)'), row=1, col=1)
        fig_physics.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="WARNING (55)", row=1, col=1)
        fig_physics.add_hline(y=75, line_dash="dash", line_color=COLOR_WARN, annotation_text="CONTRACTION (75)", row=1, col=1)

        # 2. Kinematic Position
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['POSITION_INDEX'], name="Position Gap", line=dict(color=COLOR_PRIMARY, width=2)), row=2, col=1)
        fig_physics.add_hline(y=0.18, line_dash="dot", line_color=TEXT_COLOR, annotation_text="Base (0.18)", row=2, col=1)
        fig_physics.add_hline(y=0.5, line_dash="dash", line_color=COLOR_WARN, annotation_text="Sahm Rule (0.50)", row=2, col=1)

        # 3. Pulse Momentum
        mom_colors = [COLOR_WARN if val >= 0.10 else COLOR_YIELD if val > 0 else COLOR_LIQ for val in df_plot['POSITION_INDEX_TREND']]
        fig_physics.add_trace(go.Bar(x=df_plot.index, y=df_plot['POSITION_INDEX_TREND'], name="Momentum", marker_color=mom_colors, marker_line_width=0), row=3, col=1)
        fig_physics.add_hline(y=0.10, line_dash="dash", line_color=COLOR_WARN, annotation_text="TRIGGER (+0.10)", row=3, col=1)

        # 4-8. Derivatives
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['recession_prob'], name="Prob %", fill='tozeroy', line=dict(color=COLOR_CREDIT, width=2), fillcolor='rgba(179, 136, 255, 0.15)'), row=4, col=1)
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['kinetic_energy'], name="Energy", fill='tozeroy', line=dict(color='#FF9800', width=2), fillcolor='rgba(255, 152, 0, 0.15)'), row=5, col=1)
        fig_physics.add_bar(x=df_plot.index, y=df_plot['velocity'], name="Velocity", marker_color=COLOR_PRIMARY, marker_line_width=0, row=6, col=1)
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['jerk_z'], name="Jerk Z", line=dict(color=COLOR_WARN, width=1.5)), row=7, col=1)
        fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['velocity'], name="Current Velocity", line=dict(color=COLOR_WARN, width=2)), row=8, col=1)
        fig_physics.add_hline(y=15.0, line_dash="dash", line_color="orange", annotation_text="2008 GFC", row=8, col=1)
        fig_physics.add_hline(y=25.0, line_dash="dash", line_color=COLOR_WARN, annotation_text="1929 Level", row=8, col=1)

        fig_physics = apply_shared_aesthetics(fig_physics, df_plot, last_hist_date, end_date, ai_state['pulse_intensity'], 8)
        fig_physics.update_layout(height=2000, margin=dict(l=40, r=40, t=60, b=40), showlegend=False)
        for annotation in fig_physics['layout']['annotations']: annotation['font'] = dict(size=12, color=TEXT_COLOR)
        st.plotly_chart(fig_physics, use_container_width=True)

    # ==========================================
    # TAB 3: 3D KINEMATICS (HOLOGRAPHIC RENDER)
    # ==========================================
    with tab3:
        st.markdown(f"<p style='color: {TEXT_COLOR}; font-size: 14px;'>Drag to rotate the model. Scroll to zoom in on specific historical trajectories.</p>", unsafe_allow_html=True)
        
        df_3d = df[df.index > '2005-01-01'].copy()
        
        # Projections
        latest = df_3d.iloc[-1]
        v_smooth = df_3d['sahm_gap_smooth'].diff(1).iloc[-1]
        a_smooth = df_3d['sahm_gap_smooth'].diff(1).diff(1).iloc[-1]
        future_steps = [1, 2, 3]
        proj_gaps = [max(0, latest['sahm_gap_smooth'] + (v_smooth * t) + (0.5 * a_smooth * (t**2))) for t in future_steps]
        proj_v = latest['velocity'] + (latest['acceleration'] * 3)
        proj_a = latest['acceleration']
        
        state_color = COLOR_WARN if ai_state['pulse_intensity'] > 55 else COLOR_YIELD if ai_state['pulse_intensity'] > 40 else COLOR_PRIMARY
            
        recession_mask = df_3d.index.map(lambda x: any(start <= x <= end for start, end in NBER_RECESSION_PERIODS))
        df_recessions = df_3d[recession_mask]
        
        fig_3d = go.Figure()
        
        # 1. Ghosted Historical Path (Thinner, highly transparent to remove the 'spaghetti' effect)
        fig_3d.add_trace(go.Scatter3d(
            x=df_3d['sahm_gap_smooth'], y=df_3d['velocity'], z=df_3d['acceleration'], 
            mode='lines', name='Historical Path', 
            line=dict(color='rgba(160, 174, 192, 0.15)', width=2)
        ))
        
        # 2. Soft Glowing Spheres for History (Using a cooler, tech-focused colorscale)
        energy_sizes = np.interp(df_3d['kinetic_energy'], [df_3d['kinetic_energy'].min(), df_3d['kinetic_energy'].max()], [3, 12])
        fig_3d.add_trace(go.Scatter3d(
            x=df_3d['sahm_gap_smooth'], y=df_3d['velocity'], z=df_3d['acceleration'], 
            mode='markers', name='History', 
            marker=dict(
                size=energy_sizes, color=df_3d['jerk_z'], colorscale='IceFire', opacity=0.5,
                line=dict(width=0) # Removed hard borders for a softer look
            )
        ))
        
        # 3. Subtle Recession Zones (Deep crimson, slight glow)
        fig_3d.add_trace(go.Scatter3d(
            x=df_recessions['sahm_gap_smooth'], y=df_recessions['velocity'], z=df_recessions['acceleration'], 
            mode='markers', name='Recession Zones', 
            marker=dict(size=4, color=COLOR_WARN, opacity=0.6)
        ))
        
        # 4. Neon Forecast Line (Dotted, cleaner trajectory)
        fig_3d.add_trace(go.Scatter3d(
            x=[latest['sahm_gap_smooth']] + proj_gaps, 
            y=[latest['velocity'], latest['velocity'], latest['velocity'], proj_v], 
            z=[latest['acceleration'], latest['acceleration'], latest['acceleration'], proj_a], 
            mode='lines', name='3M Forecast', 
            line=dict(color=state_color, width=5, dash='dot')
        ))
        
        # 5. Metallic Rendered Momentum Cone (Sleeker proportions, light reflection physics)
        scale = 0.4 
        fig_3d.add_trace(go.Cone(
            x=[latest['sahm_gap_smooth']], y=[latest['velocity']], z=[latest['acceleration']], 
            u=[latest['momentum'] * scale], v=[latest['velocity'] * scale], w=[latest['acceleration'] * scale], 
            colorscale=[[0, state_color], [1, state_color]], showscale=False, name='Momentum Vector',
            sizemode="absolute", sizeref=3, # Makes the cone thinner and sharper
            lighting=dict(ambient=0.3, diffuse=0.8, specular=0.8, roughness=0.2, fresnel=0.2) # Brushed metal effect
        ))
        
        # 6. Sleek Current State Indicator (Sphere with a crisp halo)
        fig_3d.add_trace(go.Scatter3d(
            x=[latest['sahm_gap_smooth']], y=[latest['velocity']], z=[latest['acceleration']], 
            mode='markers', name='CURRENT', 
            marker=dict(size=14, color=state_color, symbol='circle', line=dict(color='#FFFFFF', width=2)), 
            text=[f"Status: {state_color.upper()}<br>Energy: {latest['kinetic_energy']:.2f}"]
        ))
        
        # 7. Holographic Environment (No walls, ghosted grid)
        axis_format = dict(
            showbackground=False, 
            gridcolor="rgba(255, 255, 255, 0.03)", 
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="rgba(255, 255, 255, 0.3)"),
            title_font=dict(color="rgba(255, 255, 255, 0.6)")
        )
        
        # Dynamic color-coded title acting as the space-saving legend
        title_html = f"<b>3D KINEMATICS</b> | <span style='color:#A0AEC0'>Historical Path</span> | <span style='color:{COLOR_WARN}'>Recessions</span> | <span style='color:{state_color}'>Current & Forecast</span>"
        
        fig_3d.update_layout(
            title=dict(text=title_html, font=dict(size=14, color=TEXT_COLOR), x=0.0, y=0.98),
            showlegend=False, # <--- This kills the bulky side legend
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            scene=dict(
                xaxis_title='Cole Pulse (Position)', yaxis_title='Risk Velocity', zaxis_title='Acceleration',
                xaxis=axis_format, yaxis=axis_format, zaxis=axis_format,
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)) 
            ),
            height=850, margin=dict(l=0, r=0, t=40, b=0) # Increased top margin (t=40) to fit the new title
        )
        st.plotly_chart(fig_3d, use_container_width=True)