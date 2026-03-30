import os
# --- SILENCE TENSORFLOW WARNINGS  ---
# 3 = Filter INFO, WARNING, and ERROR messages (keeps FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Disable OneDNN optimizations which often trigger the specific "Executor start aborting" logs on CPUs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import pandas as pd
import numpy as np
import time
import smtplib
import os
import json
import argparse
import sys
import functools
import requests
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from typing import List, Tuple, Dict, Any, Optional

# --- AI & Data Imports ---
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from fredapi import Fred
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf # Import TF only after setting env vars
import warnings

# --- Setup ---
warnings.simplefilter('ignore')
# Explicitly silence Python-level TF logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Silence Kaleido/YFinance noise
logging.getLogger('kaleido').setLevel(logging.WARNING)
logging.getLogger('yfinance').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')



tf.get_logger().setLevel('ERROR')

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

LOGO_PATH = os.path.join(BASE_DIR, "AQI_Logo.png")

NBER_RECESSION_PERIODS = [
    (pd.to_datetime('1990-07-01'), pd.to_datetime('1991-03-01')),
    (pd.to_datetime('2001-03-01'), pd.to_datetime('2001-11-01')),
    (pd.to_datetime('2007-12-01'), pd.to_datetime('2009-06-01')),
    (pd.to_datetime('2020-02-01'), pd.to_datetime('2020-04-01'))
]

# --- Helper Decorators ---
def retry(times=3, delay=5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == times - 1: raise
                    time.sleep(delay)
        return wrapper
    return decorator

# --- Data Engine ---

def load_config(config_path: str) -> Dict[str, Any]:
    path = os.path.expanduser(config_path)
    if not os.path.exists(path):
        logging.critical(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f: config = json.load(f)
    return config

@retry(times=3, delay=10) # Increased delay to 10 seconds to outlast quick server blips
def fetch_grand_unified_data(api_key: str, start_date: str = '1980-01-01') -> pd.DataFrame:
    """
    Fetches both standard Macro data and specific metrics required for the Cole Pulse Physics Engine.
    Includes an Institutional Fallback Cache in case the FRED API goes down (HTTP 500).
    """
    logging.info("--- Starting Grand Unified Data Ingestion ---")
    fred = Fred(api_key=api_key)
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    cache_path = os.path.join(BASE_DIR, "aqi_macro_cache.pkl")
    
    try:
        # 1. PHYSICS ENGINE INPUTS (Specific Series)
        logging.info("Fetching Physics Engine Inputs (Job Losers, Income, Hours)...")
        job_losers = fred.get_series('LNS13023653', observation_start=start_date).asfreq('MS').ffill()
        income = fred.get_series('W875RX1', observation_start=start_date).asfreq('MS').ffill()
        hours = fred.get_series('AWHAETP', observation_start=start_date).asfreq('MS').ffill()

        # 2. CORE MACRO (USA)
        logging.info("Fetching US Core Macro...")
        philly = fred.get_series('USPHCI', observation_start=start_date).asfreq('MS').ffill()
        cfnai = fred.get_series('CFNAI', observation_start=start_date).asfreq('MS').ffill()
        ind_prod = fred.get_series('INDPRO', observation_start=start_date).asfreq('MS').ffill()
        cpi = fred.get_series('CPIAUCSL', observation_start=start_date).pct_change(12).asfreq('MS').ffill() * 100
        breakeven_5y = fred.get_series('T5YIE', observation_start=start_date).resample('MS').mean().ffill()
        unrate = fred.get_series('UNRATE', observation_start=start_date).asfreq('MS').ffill()
        claims = fred.get_series('ICSA', observation_start=start_date).resample('MS').mean().ffill()
        sahm_official = fred.get_series('SAHMREALTIME', observation_start=start_date).asfreq('MS').ffill()
        sentiment = fred.get_series('UMCSENT', observation_start=start_date).asfreq('MS').ffill()
        us_gdp_yoy = fred.get_series('A191RO1Q156NBEA', observation_start=start_date).resample('MS').ffill()

        # 3. MONETARY & LIQUIDITY
        logging.info("Fetching Global Policy Inputs...")
        fed_funds = fred.get_series('FEDFUNDS', observation_start=start_date).asfreq('MS').ffill()
        au_unrate = fred.get_series('LRHUTTTTAUM156S', observation_start=start_date).asfreq('MS').ffill()
        au_cash_rate = fred.get_series('IRSTCI01AUM156N', observation_start=start_date).asfreq('MS').ffill()
        au_cpi_yoy = fred.get_series('CPALTT01AUQ659N', observation_start=start_date).resample('MS').ffill()
        
        us2y = fred.get_series('DGS2', observation_start=start_date).resample('MS').mean().ffill()
        walcl = fred.get_series('WALCL', observation_start=start_date).resample('MS').mean().ffill() 
        tga = fred.get_series('WTREGEN', observation_start=start_date).resample('MS').mean().ffill() 
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date).resample('MS').mean().ffill() 

        # 4. MARKET RISK
        curve_10y2y = fred.get_series('T10Y2Y', observation_start=start_date).resample('MS').mean().ffill()
        real_yield_10y = fred.get_series('DFII10', observation_start=start_date).resample('MS').mean().ffill()
        
        # 5. CREDIT
        cp_90d = fred.get_series('RIFSPPFAAD90NB', observation_start=start_date).resample('MS').mean().ffill()
        us3m = fred.get_series('DGS3MO', observation_start=start_date).resample('MS').mean().ffill()
        hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date).resample('MS').mean().ffill()

        # 6. DEMOGRAPHICS
        us_emp_raw = fred.get_series('CE16OV', observation_start=start_date).asfreq('MS').ffill() * 1000
        us_pop_working = fred.get_series('LFWA64TTUSM647S', observation_start=start_date).asfreq('MS').ffill()
        au_emp_raw = fred.get_series('LFEMTTTTAUM647S', observation_start=start_date).asfreq('MS').ffill()
        au_pop_working = fred.get_series('LFWA64TTAUM647S', observation_start=start_date).asfreq('MS').ffill()
        au_gdp_raw = fred.get_series('NGDPRSAXDCAUQ', observation_start=start_date)
        au_gdp_yoy = (au_gdp_raw.pct_change(4) * 100).resample('MS').ffill()

    except Exception as e:
        logging.error(f"❌ FRED API Failed ({e}). Attempting to load from local cache...")
        if os.path.exists(cache_path):
            logging.info("♻️ Loading data from local cache.")
            return pd.read_pickle(cache_path)
        else:
            raise ValueError("FRED Fetch Failed and no local cache exists.")

    # 7. MARKET INTERNALS (YFinance)
    logging.info("Fetching Market Internals...")
    tickers = {
        'CPER': 'COPPER', 'GC=F': 'GOLD', 'CL=F': 'OIL_PRICE', 
        'DX-Y.NYB': 'DXY', 'JPY=X': 'USD_JPY', 'AUDUSD=X': 'AUD_USD', 
        'BTC-USD': 'BITCOIN', 'SMH': 'SEMI_EQ', 
        '^VIX': 'VIX', '^VIX3M': 'VIX3M',
        'TLT': 'TLT_PRICE', 'SPY': 'SPY', 'RSP': 'RSP', 
        'KBE': 'BANK_US', 'EUFN': 'BANK_EU', 'IXG': 'GLOBAL_FIN',
        'EWA': 'AUS_EQ', 'EWY': 'KOREA_EQ', 
        'HYG': 'HY_BOND', 'LQD': 'IG_BOND', 'EMB': 'EM_BOND', 'BNDX': 'INTL_BOND'
    }
    
    market_data = {}
    master_index = unrate.index 

    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start_date, progress=False)
            if isinstance(raw.columns, pd.MultiIndex): series = raw['Close'].squeeze()
            else: series = raw['Close']
            if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
            
            if name == 'TLT_PRICE': 
                daily_ret = series.pct_change()
                vol_series = daily_ret.rolling(window=21).std() * np.sqrt(252) * 100
                vol_monthly = vol_series.resample('MS').mean()
                vol_monthly.index = vol_monthly.index.tz_localize(None)
                market_data['RATE_VOL'] = vol_monthly
            
            series_monthly = series.resample('MS').mean()
            series_monthly.index = series_monthly.index.tz_localize(None)
            market_data[name] = series_monthly
        except Exception:
            market_data[name] = pd.Series(np.nan, index=master_index)

    # 8. ASSEMBLY
    data_dict = {
        "JOB_LOSERS": job_losers, "REAL_INCOME": income, "WEEKLY_HOURS": hours,
        "UNRATE": unrate, "SAHM_OFFICIAL": sahm_official, "JOBLESS_CLAIMS": claims, "SENTIMENT": sentiment,
        "PHILLY_FED": philly, "CHICAGO_FED_ACT": cfnai, "IND_PROD": ind_prod,
        "CPI_YOY": cpi, "INFLATION_BREAKEVEN": breakeven_5y,
        "FED_FUNDS": fed_funds, "US_2Y": us2y, "CURVE_10Y2Y": curve_10y2y,
        "REAL_YIELD_10Y": real_yield_10y, 
        "CREDIT_SPREAD": hy_spread, "CP_RATE": cp_90d, "US_3M": us3m, 
        "FED_ASSETS": walcl, "TGA": tga, "RRP": rrp,
        "AU_UNRATE": au_unrate, "AU_CASH_RATE": au_cash_rate, "AU_CPI_YOY": au_cpi_yoy,
        "US_EMP_TOT": us_emp_raw, "US_POP_WORKING_AGE": us_pop_working,
        "AU_EMP_TOT": au_emp_raw, "AU_POP_WORKING_AGE": au_pop_working, 
        "US_GDP_YOY": us_gdp_yoy, "AU_GDP_YOY": au_gdp_yoy,
        **market_data
    }
    
    df = pd.DataFrame(data_dict)

    # 9. CLEANING & DERIVED METRICS
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.ffill().bfill() 

    df['CP_SPREAD'] = df['CP_RATE'] - df['US_3M']
    df['NET_LIQUIDITY'] = df['FED_ASSETS'] - df['TGA'] - df['RRP']
    df['LIQUIDITY_ROC_3M'] = df['NET_LIQUIDITY'].pct_change(3)
    df['BREADTH_RATIO'] = df['RSP'] / df['SPY'] 
    df['VIX_TERM'] = df['VIX'] / df['VIX3M'] 
    df['CARRY_STRESS'] = df['USD_JPY'].pct_change(3) 
    df['POLICY_ERROR_SPREAD'] = df['US_2Y'] - df['FED_FUNDS']
    df['CLAIMS_ROC_3M'] = df['JOBLESS_CLAIMS'].pct_change(3)
    df['AU_LABOR_CRACK'] = df['AU_UNRATE'] - df['AU_UNRATE'].rolling(12).min()
    df['US_EMP_WORKING_AGE_RATIO'] = df['US_EMP_TOT'] / df['US_POP_WORKING_AGE']
    df['AU_EMP_WORKING_AGE_RATIO'] = df['AU_EMP_TOT'] / df['AU_POP_WORKING_AGE']
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_TREND'] = df[col].diff(periods=3)

    # --- SAVE TO CACHE FOR FUTURE FAILURES ---
    try:
        df.to_pickle(cache_path)
        logging.info("💾 Saved fresh data to local cache.")
    except Exception as e:
        logging.warning(f"⚠️ Could not save local cache: {e}")

    return df

# --- THE PHYSICS ENGINE (Logic Core) ---

def apply_cole_pulse_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upgraded Cole Pulse Physics Engine (Continuous ML Edition).
    Incorporates Log-Normalized Mass, EWMA Smoothing, Regime-Adaptive Z-Scoring, 
    and Continuous Sigmoid Scoring for differentiable feature generation.
    """
    logging.info("Initializing Continuous Cole Pulse Physics Engine...")
    
    # --- Helper: Sigmoid Scoring Function ---
    def sigmoid_score(series, L, k, x0):
        """
        L = Maximum points allocated
        k = Steepness of the curve
        x0 = Midpoint (Z-score threshold where risk accelerates)
        """
        return L / (1 + np.exp(-k * (series - x0)))

    # --- A. KINEMATIC POSITION (The Level) ---
    df['unrate_ma3'] = df['UNRATE'].rolling(window=3).mean()
    df['unrate_12m_low'] = df['unrate_ma3'].rolling(window=12, min_periods=1).min()
    df['POSITION_INDEX'] = df['unrate_ma3'] - df['unrate_12m_low']
    df['POSITION_INDEX_TREND'] = df['POSITION_INDEX'].diff(3).fillna(0)
    df['sahm_gap_smooth'] = df['POSITION_INDEX']

    # --- B. DYNAMICS & LOG-NORMALIZED MASS ---
    df['income_yoy'] = df['REAL_INCOME'].ffill().pct_change(12)
    df['hours_yoy'] = df['WEEKLY_HOURS'].ffill().pct_change(12)
    
    df['loser_share'] = df['JOB_LOSERS'] / (df['UNRATE'] * 100)
    
    # OPTIMIZATION: Log-Normalized Mass using np.log1p for stability
    job_losers_ma60 = df['JOB_LOSERS'].rolling(window=60, min_periods=1).mean()
    raw_mass = df['JOB_LOSERS'] / job_losers_ma60
    
    # RETAIL OPTIMIZATION 5: Clip "Black Swan" outliers (e.g., April 2020) to preserve signal integrity
    clip_limit = raw_mass.quantile(0.98)
    df['mass'] = np.log1p(np.clip(raw_mass, a_min=None, a_max=clip_limit))
        
    # --- C. PROBABILITY & KINEMATICS (EWMA Smoothing) ---
    df['recession_prob_raw'] = np.interp(df['POSITION_INDEX'], [0, 0.5], [0, 100])
    
    # OPTIMIZATION: Exponentially Weighted Moving Average
    df['recession_prob'] = df['recession_prob_raw'].ewm(span=3, adjust=False).mean()
    
    # Derivatives
    df['velocity'] = df['recession_prob'].diff(1).fillna(0)
    df['acceleration'] = df['velocity'].diff(1).fillna(0)
    df['kinetic_energy'] = 0.5 * df['mass'] * (df['velocity']**2)
    df['momentum'] = df['mass'] * df['velocity']

    # Jerk (Shock Detection)
    df['jerk'] = df['acceleration'].diff(1).fillna(0)
    jerk_std = df['jerk'].rolling(window=60, min_periods=1).std().replace(0, np.nan)
    df['jerk_z'] = (df['jerk'] / jerk_std).fillna(0)

    # --- D. REGIME-ADAPTIVE Z-SCORING ---
    # OPTIMIZATION: 5-Year rolling baselines to normalize macro regimes
    pos_roll_mean = df['POSITION_INDEX'].rolling(window=60, min_periods=1).mean()
    pos_roll_std = df['POSITION_INDEX'].rolling(window=60, min_periods=1).std().replace(0, np.nan)
    df['POSITION_Z'] = ((df['POSITION_INDEX'] - pos_roll_mean) / pos_roll_std).fillna(0)

    mom_roll_mean = df['POSITION_INDEX_TREND'].rolling(window=60, min_periods=1).mean()
    mom_roll_std = df['POSITION_INDEX_TREND'].rolling(window=60, min_periods=1).std().replace(0, np.nan)
    df['MOMENTUM_Z'] = ((df['POSITION_INDEX_TREND'] - mom_roll_mean) / mom_roll_std).fillna(0)

    nrg_roll_mean = df['kinetic_energy'].rolling(window=60, min_periods=1).mean()
    nrg_roll_std = df['kinetic_energy'].rolling(window=60, min_periods=1).std().replace(0, np.nan)
    df['ENERGY_Z'] = ((df['kinetic_energy'] - nrg_roll_mean) / nrg_roll_std).fillna(0)

    # --- E. CONTINUOUS SIGMOID SCORING MATRIX ---
    # OPTIMIZATION: Differentiable S-Curves mapping standard deviations to points
    # Midpoints (x0) are set to +1.5 or +2.0 standard deviations above the regime mean
    df['score_pos'] = sigmoid_score(df['POSITION_Z'], L=35, k=2.0, x0=1.5)
    df['score_mom'] = sigmoid_score(df['MOMENTUM_Z'], L=35, k=2.0, x0=1.5)
    df['score_nrg'] = sigmoid_score(df['ENERGY_Z'], L=30, k=1.5, x0=2.0)
    
    # Total Intensity Score
    df['PULSE_INTENSITY'] = df['score_pos'] + df['score_mom'] + df['score_nrg']
    
    # --- F. MULTI-STAGE TRIGGER ---
    df['pulse_raw'] = (df['PULSE_INTENSITY'] > 55).astype(int)
    df['COLE_PULSE_ACTIVE'] = df['pulse_raw'].rolling(window=6, min_periods=1).max()
    
    # RETAIL OPTIMIZATION 5: Standardize to measured, non-alarmist terminology
    df['shock_intensity'] = np.where(df['PULSE_INTENSITY'] > 75, "SYSTEM CONTRACTION", 
                            np.where(df['PULSE_INTENSITY'] > 50, "LATE-CYCLE WARNING", "STABLE"))

    return df

# --- AI & Analysis ---

def generate_freemium_analysis(api_key: str, context: Dict) -> Tuple[str, str]:
    logging.info("Generating Two-Tiered Substack Briefing (Free vs Paid)...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        # --- THE FREE TEASER PROMPT (Macro Context Only) ---
        free_prompt = f"""
        ACT AS: Lead Systematic Macro Allocator at Angel Quantum Intelligence.
        AUDIENCE: Public newsletter subscribers (Retail & Institutional).
        TASK: Write a 3-paragraph macro teaser. 
        TONE: Data-driven, ominous but disciplined, highly professional.
        
        DATA:
        - Cole Pulse Intensity: {context['pulse_intensity']:.1f}/100 
        - Recession Probability: {context['recession_prob']:.1f}%
        - Momentum Trend: {context['sahm_trend']:.3f}
        
        INSTRUCTIONS:
        1. State the current systemic regime based on the Pulse Intensity. 
        2. Explain the interaction between liquidity and labor momentum right now.
        3. End with a cliffhanger about building fragility or opportunity, explicitly stating: "Below the paywall, we break down the exact Z-score triggers, asset allocations, and directional bets our STGNN is positioning for this week."
        DO NOT MENTION SPECIFIC TICKERS OR TRADES. Format in clean Markdown.
        """
        
        # --- THE PAID ALPHA PROMPT (Execution & Sizing) ---
        paid_prompt = f"""
        ACT AS: Lead Systematic Macro Allocator at Angel Quantum Intelligence.
        AUDIENCE: Paying Hedge Fund / Premium Subscribers.
        TASK: Write the execution and deployment brief.
        TONE: Ruthless, quantitative, actionable.
        
        DATA:
        - Cole Pulse Intensity: {context['pulse_intensity']:.1f}/100 
        - High Yield Spread: {context['credit_spread']:.2f}%
        - Net Liquidity ROC: {context['liquidity_roc']:.2%}
        
        INSTRUCTIONS:
        1. Write a section called "SYSTEMATIC TRADE HIERARCHY".
        2. Based on the credit spread and liquidity data, explicitly define the Primary, Secondary, and Tertiary directional asset biases (e.g., Short QQQ, Long XLF, Short USD/JPY).
        3. Write a section called "CAPITAL ALLOCATION". Provide a hypothetical portfolio weighting percentage for these assets based on current volatility.
        4. Write a section called "INVALIDATION LEVEL". What breaks this thesis?
        Format in clean, aggressive Markdown with bullet points.
        """
        
        free_content = model.generate_content(free_prompt).text
        paid_content = model.generate_content(paid_prompt).text
        
        return free_content, paid_content
    except Exception as e:
        return f"AI Error (Free): {e}", f"AI Error (Paid): {e}"

# --- Forecasting ---

def run_forecast(df: pd.DataFrame) -> pd.Series:
    logging.info("Running Multivariate LSTM Forecast (CPU Execution)...")
    
    # Force TensorFlow to run strictly on CPU
    tf.config.set_visible_devices([], 'GPU')
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    tf.random.set_seed(42)
    
    features = ['UNRATE', 'NET_LIQUIDITY', 'CREDIT_SPREAD', 'CP_SPREAD', 'VIX']
    valid_feats = [f for f in features if f in df.columns]
    
    model_df = df[valid_feats].dropna()
    data = model_df.values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    look_back = 12
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i+look_back])
        y.append(scaled[i+look_back])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([LSTM(50, input_shape=(look_back, len(valid_feats))), Dense(len(valid_feats))])
    model.compile(optimizer='adam', loss='mse')
    
    # RETAIL OPTIMIZATION 4: Cache LSTM Weights to save compute and stabilize forecast
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    weights_path = os.path.join(BASE_DIR, "aqi_lstm_weights.h5")

    retrain = True
    if os.path.exists(weights_path):
        file_age_days = (time.time() - os.path.getmtime(weights_path)) / (60 * 60 * 24)
        if file_age_days < 7:  # Retrain once a week
            try:
                model.load_weights(weights_path)
                retrain = False
                logging.info("Loaded cached LSTM weights.")
            except Exception as e:
                logging.warning(f"Failed to load weights: {e}. Retraining...")
                
    if retrain:
        logging.info("Training LSTM from scratch (This may take a moment)...")
        model.fit(X, y, epochs=30, batch_size=16, verbose=0)
        try:
            model.save_weights(weights_path)
            logging.info("Saved new LSTM weights to cache.")
        except Exception as e:
            logging.warning(f"Could not save weights: {e}")
    
    curr = scaled[-look_back:].reshape(1, look_back, len(valid_feats))

    fc_scaled = []
    for _ in range(12):
        pred = model.predict(curr, verbose=0)
        fc_scaled.append(pred[0])
        curr = np.append(curr[:, 1:, :], pred.reshape(1, 1, len(valid_feats)), axis=1)
        
    fc_vals = scaler.inverse_transform(np.array(fc_scaled))
    
    # Safety Lock Logic
    idx = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    raw_forecast = pd.Series(fc_vals[:, 0], index=idx)
    
    # RETAIL OPTIMIZATION 3: Smooth the raw forecast to eliminate visual LSTM "jitter"
    lstm_forecast = raw_forecast.rolling(window=3, min_periods=1).mean()
    
    # Apply Slope Drift if Position Index is elevating
    slope = np.polyfit(np.arange(6), df['UNRATE'].iloc[-6:], 1)[0]
    start_val = df['UNRATE'].iloc[-1]
    if df['POSITION_INDEX'].iloc[-1] > 0.18:
        drift = [start_val + (max(slope, 0.02) * (i+1)) for i in range(12)]
        final_vals = np.maximum(lstm_forecast, drift)
        final_series = pd.Series(final_vals, index=idx)
    else:
        final_series = lstm_forecast
        
    # RETAIL OPTIMIZATION 3 (Cont.): Cap maximum month-over-month velocity at 0.5%
    capped_vals = [final_series.iloc[0]]
    for i in range(1, len(final_series)):
        diff = final_series.iloc[i] - capped_vals[-1]
        if diff > 0.5:
            capped_vals.append(capped_vals[-1] + 0.5)
        elif diff < -0.5:
            capped_vals.append(capped_vals[-1] - 0.5)
        else:
            capped_vals.append(final_series.iloc[i])
            
    return pd.Series(capped_vals, index=idx)

def generate_cycle_visual(current_regime_status, pulse_intensity):
    """Generates a stylized Business Cycle plot with uncertainty bands and forward scenarios."""
    import numpy as np
    import plotly.graph_objects as go
    
    # Create a stylized cycle using a sine wave
    x = np.linspace(0, 10, 200)
    y = np.sin(x * (np.pi / 5)) # Single wave from 0 to 10
    
    fig_cycle = go.Figure()
    
    # 1. Add Uncertainty Band (Shaded Area)
    y_upper = y + 0.15
    y_lower = y - 0.15
    fig_cycle.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(y_upper) + list(y_lower)[::-1],
        fill='toself',
        fillcolor='rgba(200, 200, 200, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Margin of Variance"
    ))
    
    # The stylized cycle line
    fig_cycle.add_trace(go.Scatter(
        x=x, y=y, mode='lines', line=dict(color='black', width=3), showlegend=False
    ))
    
    # Shade the "Late Cycle" region (Plateau and initial descent)
    fig_cycle.add_vrect(x0=1.5, x1=4.5, fillcolor="orange", opacity=0.15, layer="below", line_width=0, annotation_text="LATE CYCLE PLATEAU", annotation_position="top left")
    
    # Historical Markers (Neutralized Labels + Soft Landing Precedent)
    hist_markers_x = [1.5, 2.5, 3.2, 4.2]
    hist_markers_y = [np.sin(mx * (np.pi / 5)) for mx in hist_markers_x]
    labels = ["1995 (Soft Landing)", "2018-2019 (Growth Slowdown)", "1999-2000 (Late-Cycle Extension)", "2006-2007 (Peak Tightening)"]
    
    fig_cycle.add_trace(go.Scatter(
        x=hist_markers_x, y=hist_markers_y, mode='markers+text',
        marker=dict(size=10, color='gray', symbol='circle'),
        text=labels, textposition="top center", name="Historical Precedents"
    ))
    
    # 2. "TODAY" Marker - Dynamic Positioning
    # If intensity < 55, position on the plateau. If > 55, drop to the downward slope.
    if pulse_intensity >= 55:
        today_x = 3.8
    else:
        today_x = 2.2 # Positioned safely on the expansionary plateau
        
    today_y = np.sin(today_x * (np.pi / 5))
    
    fig_cycle.add_trace(go.Scatter(
        x=[today_x], y=[today_y], mode='markers+text',
        marker=dict(size=16, color='red', symbol='diamond'),
        text=["TODAY"], textposition="bottom center", textfont=dict(color="red", size=14, weight="bold"),
        name="Current Position"
    ))

    # 3. Add "Possible Paths" Forward
    # Path A: Base Case (60%) - Stagnation/Plateau
    fig_cycle.add_annotation(
        x=today_x + 1.4, y=today_y + 0.1, 
        ax=40, ay=-40,
        xref="x", yref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="gray",
        text="<b>Base Case (60%)</b><br>Stagnation / Grind", font=dict(color="gray", size=11),
        xanchor="left"
    )
    
    # Path B: Downside Risk (40%) - Cyclical Downturn
    fig_cycle.add_annotation(
        x=today_x + 1.2, y=today_y - 0.7, 
        ax=40, ay=50,
        xref="x", yref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="red",
        text="<b>Downside Risk (40%)</b><br>Cyclical Downturn", font=dict(color="red", size=11),
        xanchor="left"
    )

    # 4. Add Disclaimer to Title
    fig_cycle.update_layout(
        title="<b>MACRO REGIME CONTEXT</b> | Stylized Systematic Cycle Map<br><sup style='color:gray;'>*Conceptual positioning — not to scale. Paths reflect probability-weighted Cole Pulse scenarios.</sup>",
        height=450, template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.8]),
        margin=dict(l=40, r=40, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_cycle

# --- Grand Unified Visualization (Grid Locked) ---

def generate_unified_html(df, unemp_fc, html_path):
    logging.info("Generating Grand Unified Dashboard (Macro + Physics + 3D)...")
    
    # --- INSTITUTIONAL DARK MODE PALETTE ---
    BG_COLOR = "#0B0F19"
    TEXT_COLOR = "#A0AEC0"
    GRID_COLOR = "rgba(255, 255, 255, 0.05)"
    
    # 1. DEFINE GLOBAL DATE RANGE & FIX NANOSECOND BUG
    df_plot = df[df.index > '2000-01-01'].copy()
    last_hist_date = df_plot.index[-1]
    start_date = df_plot.index[0]
    
    if unemp_fc is not None and len(unemp_fc) > 0:
        end_date = unemp_fc.index[-1]
        # THE FIX: Force the forecast index to strings
        unemp_fc.index = unemp_fc.index.strftime('%Y-%m-%d')
    else:
        end_date = last_hist_date
        
    global_x_range = [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    
    # THE FIX: Force the main dataframe index to strings so Plotly draws the lines
    df_plot.index = df_plot.index.strftime('%Y-%m-%d')

    # --- 2. HUD & REGIME CALCULATION ---
    last = df.iloc[-1]
    
    regime_status = "LATE CYCLE / STRESS" if last['recession_prob'] > 40 else "EXPANSION"
    regime_color = "#ff4444" if last['recession_prob'] > 40 else "#00C851"
    
    liq_status = "CONTRACTING" if last['LIQUIDITY_ROC_3M'] < 0 else "EXPANDING"
    liq_arrow = "↓" if last['LIQUIDITY_ROC_3M'] < 0 else "↑"
    
    mom_status = "ACCELERATING (RISK)" if last['POSITION_INDEX_TREND'] > 0.10 else "STABLE"
    mom_color = "#ffbb33" if last['POSITION_INDEX_TREND'] > 0.10 else "#33b5e5"

    header_html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; color: white; padding: 20px; border-bottom: 4px solid #007bff; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-weight: 700; font-size: 24px;">AQI MACRO MONITOR</h2>
                <p style="margin: 5px 0 0 0; color: #888; font-size: 14px;">Institutional Strategy Dashboard | {last_hist_date.strftime('%Y-%m-%d')}</p>
            </div>
            <div style="display: flex; gap: 20px;">
                <div style="background: #2b2b2b; padding: 10px 20px; border-radius: 4px; border-left: 4px solid {regime_color};">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Current Regime</div>
                    <div style="font-weight: bold; font-size: 16px;">{regime_status}</div>
                </div>
                <div style="background: #2b2b2b; padding: 10px 20px; border-radius: 4px; border-left: 4px solid #9933cc;">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Global Liquidity</div>
                    <div style="font-weight: bold; font-size: 16px;">{liq_status} {liq_arrow}</div>
                </div>
                <div style="background: #2b2b2b; padding: 10px 20px; border-radius: 4px; border-left: 4px solid {mom_color};">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Pulse Momentum</div>
                    <div style="font-weight: bold; font-size: 16px;">{mom_status}</div>
                </div>
            </div>
        </div>
    </div>
    """

    # --- 3. MACRO MONITOR PLOTS ---
    fig_macro = make_subplots(
        rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
        subplot_titles=(
            "<b>CORE SIGNAL: LABOR</b> | Unemployment & Forecast", 
            "<b>CORE SIGNAL: POLICY</b> | Real Rates Regime (Fed Funds vs CPI)", 
            "<b>LIQUIDITY</b> | Net Liquidity vs Credit Spreads", 
            "<b>RISK SENTIMENT</b> | VIX vs AUD/USD", 
            "<b>FX CONTEXT</b> | Dollar (DXY) vs JPY", 
            "<b>STRUCTURAL CONTEXT</b> | Demographics",
            "<b>ECONOMIC GROWTH</b> | US vs Australia Real GDP (YoY %)" 
        ),
        specs=[[{"secondary_y": True}]] * 7
    )
    
    # ROW 1: LABOR (Color changed to light silver for dark mode)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['UNRATE'], name="Unemployment (Hist)", line=dict(color='#E2E8F0', width=2)), row=1, col=1)
    if unemp_fc is not None:
        fig_macro.add_trace(go.Scatter(
            x=unemp_fc.index, y=unemp_fc, name="AQI Forecast", mode='lines+markers',
            line=dict(color='#ff4444', dash='dot', width=2), marker=dict(symbol='circle-open', size=6, color='#ff4444')
        ), row=1, col=1)
    
    # ROW 2: POLICY
    funds = df_plot['FED_FUNDS']
    cpi = df_plot['CPI_YOY']
    y_restrictive = np.where(funds >= cpi, funds, cpi)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=cpi, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=y_restrictive, fill='tonexty', fillcolor='rgba(0, 200, 81, 0.15)', line=dict(width=0), name="Restrictive", showlegend=True), row=2, col=1)
    y_accommodative = np.where(cpi > funds, cpi, funds)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=funds, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=y_accommodative, fill='tonexty', fillcolor='rgba(255, 68, 68, 0.15)', line=dict(width=0), name="Accommodative", showlegend=True), row=2, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=funds, name="Fed Funds", line=dict(color='#007bff', width=2)), row=2, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=cpi, name="CPI YoY", line=dict(color='#ffbb33', dash='dot', width=2)), row=2, col=1)
    
    # ROW 3: LIQUIDITY
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['NET_LIQUIDITY'], name="Net Liquidity", line=dict(color='#00C851')), row=3, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['CREDIT_SPREAD'], name="HY Spread", line=dict(color='#9933cc')), row=3, col=1, secondary_y=True)
    
    # ROW 4: RISK
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VIX'], name="VIX", line=dict(color='#ffbb33')), row=4, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AUD_USD'], name="AUD/USD", line=dict(color='#33b5e5')), row=4, col=1, secondary_y=True)
    
    # ROW 5: FX
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DXY'], name="DXY", line=dict(color='#2BBBAD')), row=5, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['USD_JPY'], name="USD/JPY", line=dict(color='#aa66cc')), row=5, col=1, secondary_y=True)

    # ROW 6: DEMOGRAPHICS
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['US_EMP_WORKING_AGE_RATIO'], name="US Emp Ratio", line=dict(color='#0d47a1')), row=6, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AU_EMP_WORKING_AGE_RATIO'], name="AU Emp Ratio", line=dict(color='#ffbb33', dash='dot')), row=6, col=1)

    # ROW 7: GDP
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['US_GDP_YOY'], name="US Real GDP (YoY %)", line=dict(color='#007bff', width=2)), row=7, col=1)
    fig_macro.add_trace(go.Scatter(x=df_plot.index, y=df_plot['AU_GDP_YOY'], name="AU Real GDP (YoY %)", line=dict(color='#ffbb33', width=2)), row=7, col=1)
    fig_macro.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=7, col=1)

    # LAYOUT UPDATE: Dark Mode Aesthetics
    fig_macro.update_layout(
        height=1850, 
        template="plotly_dark", 
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Arial", color=TEXT_COLOR),
        xaxis_range=global_x_range,  
        margin=dict(l=60, r=80, t=60, b=40), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)") 
    )

    # --- 4. PHYSICS ENGINE PLOTS ---
    fig_physics = make_subplots(
        rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=(
            "<b>1. PULSE INTENSITY SCORE</b> | The New Cole Pulse (Target < 50)", 
            "<b>2. KINEMATIC POSITION</b> | Base Labor Level vs Sahm Rule", 
            "<b>3. PULSE MOMENTUM</b> | 3-Month Trend (Warning Trigger > +0.10)",
            "<b>4. PROBABILITY SIGNAL</b> | Recession Risk", 
            "<b>5. KINETIC ENERGY</b> | System Mass x Velocity²", 
            "<b>6. RISK VELOCITY</b> | Rate of Labor Deterioration", 
            "<b>7. SHOCK MAGNITUDE</b> | Jerk Z-Score (Non-Linearity)",
            "<b>8. INTENSITY CONTEXT</b> | Historical Velocity Benchmarks"
        )
    )
    
    # ROW 1: The Upgraded Cole Pulse (Color changed to light silver for dark mode)
    fig_physics.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['PULSE_INTENSITY'], 
        name="Intensity Score", line=dict(color='#dcdcdc', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)'
    ), row=1, col=1)
    
    fig_physics.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="LATE-CYCLE WARNING (55)", annotation_position="top right", row=1, col=1)
    fig_physics.add_hline(y=75, line_dash="longdash", line_width=2, line_color="red", annotation_text="SYSTEM CONTRACTION (75)", annotation_position="top right", row=1, col=1)

    # ROW 2: Kinematic Position
    fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['POSITION_INDEX'], name="Position Gap", line=dict(color='blue')), row=2, col=1)
    fig_physics.add_hline(y=0.18, line_dash="dash", line_color="orange", annotation_text="Position Base (0.18)", annotation_position="bottom right", row=2, col=1)
    fig_physics.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Sahm Rule (0.50)", annotation_position="top right", row=2, col=1)

    # ROW 3: Pulse Momentum
    mom_colors = ['#ff4444' if val >= 0.10 else '#ffbb33' if val > 0 else '#00C851' for val in df_plot['POSITION_INDEX_TREND']]
    fig_physics.add_trace(go.Bar(x=df_plot.index, y=df_plot['POSITION_INDEX_TREND'], name="Momentum", marker_color=mom_colors, marker_line_width=0), row=3, col=1)
    fig_physics.add_hline(y=0.10, line_dash="dash", line_color="red", annotation_text="TRIGGER LIMIT (+0.10)", annotation_position="top right", row=3, col=1)
    fig_physics.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)", row=3, col=1)

    # SHIFTED ROWS (4 to 8)
    fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['recession_prob'], name="Prob %", fill='tozeroy'), row=4, col=1)
    fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['kinetic_energy'], name="Energy", fill='tozeroy'), row=5, col=1)
    fig_physics.add_bar(x=df_plot.index, y=df_plot['velocity'], name="Velocity", row=6, col=1)
    fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['jerk_z'], name="Jerk Z"), row=7, col=1)
    fig_physics.add_trace(go.Scatter(x=df_plot.index, y=df_plot['velocity'], name="Current Velocity", line=dict(color='red')), row=8, col=1)
    
    fig_physics.add_hline(y=15.0, line_dash="dash", line_color="orange", annotation_text="2008 GFC Level", annotation_position="top right", row=8, col=1)
    fig_physics.add_hline(y=25.0, line_dash="dash", line_color="darkred", annotation_text="1929 Depression Level", annotation_position="top right", row=8, col=1)

    # LAYOUT UPDATE: Dark Mode Aesthetics for Physics
    fig_physics.update_layout(
        height=2050, 
        template="plotly_dark", 
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Arial", color=TEXT_COLOR),
        xaxis_range=global_x_range,
        margin=dict(l=60, r=80, t=100, b=40), 
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)") 
    )

    # SHARED AESTHETICS LOOP (Recession Bars & Today Line)
    figures = [fig_macro, fig_physics]
    for fig in figures:
        fig.update_xaxes(type="date", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
        
        for start, end in NBER_RECESSION_PERIODS:
            if start < last_hist_date and end > start_date:
                rows = 7 if fig == fig_macro else 8 
                
                v_start = max(start, start_date).strftime('%Y-%m-%d')
                v_end = min(end, last_hist_date).strftime('%Y-%m-%d')
                
                for r in range(1, rows + 1):
                    fig.add_vrect(x0=v_start, x1=v_end, fillcolor="gray", opacity=0.15, layer="below", line_width=0, row=r, col=1)
        
        # Today Line (Updated for dark mode visibility)
        rows = 7 if fig == fig_macro else 8
        today_str = last_hist_date.strftime('%Y-%m-%d')
        for r in range(1, rows + 1):
            fig.add_vline(x=today_str, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=r, col=1)
            if r == 1:
                fig.add_annotation(x=today_str, y=1, yref="paper", text="TODAY", showarrow=False, 
                                     font=dict(size=10, color=TEXT_COLOR), xanchor="right", xshift=-5, row=r, col=1)

    # RETAIL OPTIMIZATION 4: Dynamic Regime Color Coding
    if last['PULSE_INTENSITY'] > 75:
        regime_bg_color = "rgba(255, 68, 68, 0.08)"  # Red
    elif last['PULSE_INTENSITY'] > 50:
        regime_bg_color = "rgba(255, 187, 51, 0.08)" # Orange
    else:
        regime_bg_color = "rgba(0, 200, 81, 0.08)"   # Green
        
    regime_start = last_hist_date - pd.DateOffset(months=6) 
    regime_start_str = regime_start.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    for r in range(1, 8): 
        fig_macro.add_vrect(x0=regime_start_str, x1=end_date_str, fillcolor=regime_bg_color, opacity=1, layer="below", line_width=0, row=r, col=1)
    
    fig_macro.add_annotation(x=regime_start_str, y=1.05, yref="paper", text="<b>CURRENT SYSTEM REGIME</b>", showarrow=False, font=dict(size=10, color=TEXT_COLOR), xanchor="left", row=1, col=1)

    # --- 5. 3D KINEMATICS ---
    latest = df.iloc[-1]
    v_smooth = df['sahm_gap_smooth'].diff(1).iloc[-1]
    a_smooth = df['sahm_gap_smooth'].diff(1).diff(1).iloc[-1]
    future_steps = [1, 2, 3]
    proj_gaps = [max(0, latest['sahm_gap_smooth'] + (v_smooth * t) + (0.5 * a_smooth * (t**2))) for t in future_steps]
    proj_v = latest['velocity'] + (latest['acceleration'] * 3)
    proj_a = latest['acceleration']
    
    if latest['PULSE_INTENSITY'] > 55:
        state_color = 'red'
    elif latest['PULSE_INTENSITY'] > 40:
        state_color = 'orange'
    else:
        state_color = 'cyan'
        
    recession_mask = df_plot.index.map(lambda x: any(start <= pd.to_datetime(x) <= end for start, end in NBER_RECESSION_PERIODS))
    df_recessions = df_plot[recession_mask]
    
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(x=df_plot['sahm_gap_smooth'], y=df_plot['velocity'], z=df_plot['acceleration'], mode='lines', name='Historical Path', line=dict(color='rgba(255,255,255,0.15)', width=2)))
    energy_sizes = np.interp(df_plot['kinetic_energy'], [df_plot['kinetic_energy'].min(), df_plot['kinetic_energy'].max()], [4, 15])
    fig_3d.add_trace(go.Scatter3d(x=df_plot['sahm_gap_smooth'], y=df_plot['velocity'], z=df_plot['acceleration'], mode='markers', name='History', marker=dict(size=energy_sizes, color=df_plot['jerk_z'], colorscale='RdBu_r', opacity=0.6)))
    fig_3d.add_trace(go.Scatter3d(x=df_recessions['sahm_gap_smooth'], y=df_recessions['velocity'], z=df_recessions['acceleration'], mode='markers', name='Recession Zones', marker=dict(size=6, color='red', symbol='cross', opacity=0.8)))
    fig_3d.add_trace(go.Scatter3d(x=[latest['sahm_gap_smooth']] + proj_gaps, y=[latest['velocity'], latest['velocity'], latest['velocity'], proj_v], z=[latest['acceleration'], latest['acceleration'], latest['acceleration'], proj_a], mode='lines', name='3M Forecast', line=dict(color=state_color, width=8, dash='dash')))
    scale = 0.5 
    fig_3d.add_trace(go.Cone(x=[latest['sahm_gap_smooth']], y=[latest['velocity']], z=[latest['acceleration']], u=[latest['momentum'] * scale], v=[latest['velocity'] * scale], w=[latest['acceleration'] * scale], colorscale=[[0, state_color], [1, state_color]], showscale=False, name='Momentum Vector'))
    fig_3d.add_trace(go.Scatter3d(x=[latest['sahm_gap_smooth']], y=[latest['velocity']], z=[latest['acceleration']], mode='markers', name='CURRENT', marker=dict(size=25, color=state_color, symbol='diamond'), text=[f"Status: {state_color.upper()}<br>Energy: {latest['kinetic_energy']:.2f}"]))
    fig_3d.update_layout(title="<b>PART 3: AQI KINEMATIC STATE SPACE</b>", scene=dict(xaxis_title='Cole Pulse', yaxis_title='Risk Velocity', zaxis_title='Acceleration'), template="plotly_dark", height=800)
    fig_cycle = generate_cycle_visual(regime_status, last['PULSE_INTENSITY'])
    
    # --- SUBSTACK EXPORT: Static High-Res Images ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        teaser_path = os.path.join(BASE_DIR, "AQI_Teaser.png")
        fig_cycle.write_image(teaser_path, width=800, height=450, scale=2)
        macro_img_path = os.path.join(BASE_DIR, "AQI_Macro_Dashboard.png")
        physics_img_path = os.path.join(BASE_DIR, "AQI_Physics_Dashboard.png")       
        fig_macro.write_image(macro_img_path, width=1080, height=1600, scale=2)
        fig_physics.write_image(physics_img_path, width=1080, height=1800, scale=2)
        logging.info("✅ Saved high-res static PNGs for Substack export.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save static dashboards: {e}")

    # --- WRITE FILES ---
    appendix_html = """
    <div style="background-color: #2b2b2b; color: #dcdcdc; padding: 30px; margin: 20px; border-radius: 4px; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <h2 style="color: #ffc107; border-bottom: 1px solid #555; padding-bottom: 10px;">APPENDIX: What Does “Late in the Cycle” Mean?</h2>
        <p>The economy moves in cycles — <i>expansion → peak → slowdown → recession → recovery → expansion again</i>.</p>
        <p>When we say <b>“late in the cycle,”</b> we mean:</p>
        <ul>
            <li>The economy has been growing for a long time.</li>
            <li>Unemployment is very low.</li>
            <li>Corporate profits are high, and interest rates are usually elevated.</li>
            <li>Financial markets are priced optimistically.</li>
        </ul>
        <p>It’s the stage where growth is still happening — but risks are building underneath. Think of it like the final miles of a long road trip. The car is still moving forward, but fuel is lower, parts are hotter, and small issues can cause bigger problems than earlier in the journey.</p>
        
        <h3 style="color: #a0cfa0; margin-top: 25px;">What Has “Late Cycle” Looked Like Historically?</h3>
        
        <p><b>1️⃣ 2006–2007 (Before the Global Financial Crisis)</b><br>
        Unemployment was low, housing prices were high, and markets were strong. Everything looked fine on the surface. But underneath, credit risks were building and leverage was extreme. Late cycle didn’t mean recession immediately — it meant the system was fragile and vulnerable to a shock.</p>
        
        <p><b>2️⃣ 1999–2000 (Dot-Com Peak)</b><br>
        Economic growth was strong and unemployment was very low, but stock valuations were extremely high. This was classic late-cycle behavior: strong economy + stretched asset prices. The slowdown that followed was milder economically, but markets fell sharply.</p>
        
        <p><b>3️⃣ 2018–2019 (Pre-COVID Slowdown Phase)</b><br>
        Unemployment was at historic lows, the Fed had raised rates, and corporate debt was high. The economy was not in recession, but momentum was fading. COVID became the external shock that tipped it over.</p>

        <h3 style="color: #a0cfa0; margin-top: 25px;">Why Investors Care</h3>
        <p>Early-cycle investing is about riding acceleration. <b>Late-cycle investing is about managing risk.</b></p>
        <ul>
            <li>Defensive sectors often outperform.</li>
            <li>Cash and quality balance sheets matter more.</li>
            <li>High-risk assets become more vulnerable, and market timing becomes harder.</li>
        </ul>
        <p><b>Simple Summary:</b> The economy is still expanding, but the buffer is thin. Growth continues — but the system has less room for error.</p>
    </div>
    """

    unified_html_path = os.path.join(BASE_DIR, html_path)
    model_3d_path = os.path.join(BASE_DIR, "AQI_3d_physics_model.html")

    with open(html_path, 'w') as f:
        f.write("<html><head><title>AQI Institutional Monitor</title></head><body style='margin:0; background:#0B0F19;'>")
        f.write(header_html)
        
        f.write("<div style='background:#0B0F19; margin: 0 20px 20px 20px; border-radius: 4px;'>")
        f.write(pio.to_html(fig_macro, full_html=False, include_plotlyjs='cdn'))
        f.write("</div>")
        
        f.write("<div style='background:#0B0F19; margin: 20px; border-radius: 4px;'>")
        f.write(pio.to_html(fig_physics, full_html=False, include_plotlyjs=False))
        f.write("</div>")

        f.write("<div style='background:#0B0F19; margin: 20px; border-radius: 4px;'>")
        f.write(pio.to_html(fig_cycle, full_html=False, include_plotlyjs=False))
        f.write("</div>")

        f.write("<div style='background:#111; margin: 20px; border-radius: 4px;'>")
        f.write(pio.to_html(fig_3d, full_html=False, include_plotlyjs=False))
        f.write("</div>")
        
        f.write(appendix_html)
        f.write("</body></html>")
    
    pio.write_html(fig_3d, file=model_3d_path, auto_open=False)
    logging.info("Saved AQI_Unified.html and standalone AQI_3d_physics_model.html")

def send_email(config, subject, body, embedded_images):
    try:
        msg = MIMEMultipart('related')
        msg['From'] = config['email']
        msg['To'] = ", ".join(config['recipient_emails'])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Dynamically load and embed all images provided in the dictionary
        for cid, path in embedded_images.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', f'<{cid}>')
                    msg.attach(img)
            else:
                logging.warning(f"⚠️ Image not found for email embed: {path}")

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(config['email'], config['password'])
            server.send_message(msg)
        logging.info("✅ Briefing Sent Successfully.")
    except Exception as e:
        logging.error(f"❌ Email Failed: {e}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=os.path.join(BASE_DIR, 'config.json'))
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        
        # 1. Fetch
        df = fetch_grand_unified_data(config['fred_api_key'])
        
        # 2. Physics Engine
        df = apply_cole_pulse_physics(df)
        
        # 3. Forecast
        unemp_fc = run_forecast(df)
        
        # 4. Context for AI
        last = df.iloc[-1]
        
        context = {
            'cole_pulse_active': last['COLE_PULSE_ACTIVE'],
            'pulse_intensity': last['PULSE_INTENSITY'], # New Metric
            'position_index': last['POSITION_INDEX'],
            'kinetic_energy': last['kinetic_energy'],
            'recession_prob': last['recession_prob'],
            'sahm_trend': last['POSITION_INDEX_TREND'],
            'shock_status': last['shock_intensity'], # New Metric
            'liquidity_roc': last['LIQUIDITY_ROC_3M'],
            'credit_spread': last['CREDIT_SPREAD'],
            'vix': last['VIX'],
            'unrate': last['UNRATE'],
            'us_emp_ratio': last['US_EMP_WORKING_AGE_RATIO'] * 100,
            'au_emp_ratio': last['AU_EMP_WORKING_AGE_RATIO'] * 100
        }
        
        # 5. Dashboard Generation (Visuals Integration)
        html_path = config.get('plot_html_filename', 'AQI_Unified.html')
        # This now generates both AQI_Unified.html AND AQI_3d_physics_model.html
        generate_unified_html(df, unemp_fc, html_path)
        
        # 6. AI Analysis (Freemium Split)
        if config.get('GEMINI_API_KEY'):
            free_text, paid_text = generate_freemium_analysis(config['GEMINI_API_KEY'], context)
        else:
            free_text, paid_text = "AI Offline", "AI Offline"
            
        # 7. SUBSTACK PACKAGING
        substack_draft_path = os.path.join(BASE_DIR, f"Substack_Draft_{datetime.now().strftime('%Y%m%d')}.md")
        with open(substack_draft_path, "w") as f:
            # Free Section
            f.write("Welcome to the weekly AQI Genesis macro update.\n\n")
            f.write(free_text + "\n\n")
            f.write("*(⬇️ Drag and drop 'AQI_Macro_Dashboard.png' here ⬇️)*\n\n")
            
            # The Teaser / Upsell
            f.write("---\n\n")
            f.write("### 🔒 Institutional Execution & Asset Allocation\n")
            f.write("Below the paywall, we break down the exact Z-score triggers, asset allocations, and directional bets our STGNN is positioning for this week based on the Cole Pulse Physics Engine.\n\n")
            
            # The explicit instruction to use the Substack UI tool
            f.write("[ INSERT NATIVE SUBSTACK PAYWALL HERE ]\n\n")
            
            # Paid Section
            f.write(paid_text + "\n\n")
            f.write("*(⬇️ Drag and drop 'AQI_Physics_Dashboard.png' here ⬇️)*\n\n")
            
            f.write("---\n")
            f.write("*AQI Genesis is an institutional systematic macro engine. This briefing is for informational purposes and does not constitute financial advice.*\n")
            
        logging.info(f"✅ Substack Draft generated: {substack_draft_path}")

        # --- 8. THE SAAS DATABASE HOOK (STREAMLIT INTEGRATION) ---
        import sqlite3
        logging.info("💾 Pushing live data matrix to SaaS Database (Streamlit backend)...")
        try:
            db_path = os.path.join(BASE_DIR, "aqi_saas_backend.db")
            conn = sqlite3.connect(db_path)
            
            # Push the dataframes
            df.to_sql("macro_state", conn, if_exists="replace", index=True)
            if unemp_fc is not None:
                unemp_fc_df = unemp_fc.to_frame(name="unrate_forecast")
                unemp_fc_df.to_sql("macro_forecast", conn, if_exists="replace", index=True)
                
            context_df = pd.DataFrame([context])
            context_df.to_sql("ai_system_state", conn, if_exists="replace", index=False)
            
            conn.close()
            logging.info("✅ SaaS Database updated. The AQI Genesis Streamlit Terminal is now live.")
        except Exception as db_err:
            logging.error(f"❌ Database Push Failed: {db_err}")

        # --- 9. EMAIL DELIVERY ---
        if last['recession_prob'] > 50:
            status_headline = "Regime Shift, Downturn Elevated"
        elif last['PULSE_INTENSITY'] > 55:
            status_headline = "Fragility Building, Defensive Posture"
        else:
            status_headline = "Expansion Intact, Margin for Error Narrowing"

        subject = f"AQI System Update: {status_headline} | {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""<html><body style="background-color: #1e1e1e; color: #dcdcdc; margin: 0; padding: 0;">
        <div style='text-align: center; background-color: #111; padding: 20px;'><img src='cid:aqi_logo' width='150'></div>
        
        <div style="max-width: 1080px; margin: auto; padding: 30px;">
            <h2 style="color: #007bff; border-bottom: 1px solid #444; padding-bottom: 10px;">FREE PUBLIC TEASER</h2>
            <div style="white-space: pre-wrap; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6;">{free_text}</div>
            
            <div style='text-align: center; margin-top: 30px; margin-bottom: 20px;'>
                <img src='cid:aqi_macro' style='max-width: 100%; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);'>
            </div>
            
            <h2 style="color: #ffc107; border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 40px;">PAID PREMIUM ANALYSIS</h2>
            <div style="white-space: pre-wrap; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6;">{paid_text}</div>
            
            <div style='text-align: center; margin-top: 30px; margin-bottom: 20px;'>
                <img src='cid:aqi_physics' style='max-width: 100%; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);'>
            </div>
        </div>
        
        <hr style="border-color: #333; max-width: 1080px;">
        <p style='color: #888; text-align: center; font-family: sans-serif; font-size: 14px; padding-bottom: 30px;'>
            <i>To view the live, interactive 3D Kinematic State Space model and wargame specific policy shocks, log in to the <b>AQI Genesis Terminal</b>.</i>
        </p>
        </body></html>"""
        
        if 'recipient_emails' in config:
            embedded_images = {
                'aqi_logo': LOGO_PATH,
                'aqi_macro': os.path.join(BASE_DIR, "AQI_Macro_Dashboard.png"),
                'aqi_physics': os.path.join(BASE_DIR, "AQI_Physics_Dashboard.png")
            }
            send_email(config, subject, body, embedded_images)

    except Exception as e:
        logging.critical(f"System Failure: {e}", exc_info=True)

if __name__ == "__main__":
    main()