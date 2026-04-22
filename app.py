"""
DataLens — Smart Data Visualizer  v4.1  (Single File)
======================================================
Fixes in v4.1
-------------
  • ollama_chat: stream=True + split timeout=(10,300) — no more 120 s wall-clock freeze
  • Partial responses are returned on timeout instead of being lost
  • Spinner replaced with st.empty() placeholder so the page stays responsive
  • NLQ /ask route typo fixed: x_cs → x_c
  • ollama_available() and list_ollama_models() use connect-only timeout (2 s)

Install
-------
    pip install streamlit pandas numpy plotly openpyxl scikit-learn requests

Run
---
    streamlit run app.py
"""

import io
import re
import json
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests as _req
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except ImportError:
    SKLEARN = False


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  +  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataLens",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; letter-spacing: 0.01em; }
.stApp { background: #080c17; color: #c8d0e0; }

[data-testid="stSidebar"] { background: #0c1120 !important; border-right: 1px solid #182035; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio span {
    color: #8899bb !important; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.08em;
}

[data-testid="metric-container"] {
    background: #0f1524 !important; border: 1px solid #1b2844 !important;
    border-radius: 8px !important; padding: 14px 16px !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 10px !important; text-transform: uppercase; letter-spacing: 0.1em;
    color: #384a68 !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 700 !important;
    color: #f0b429 !important; font-family: 'JetBrains Mono', monospace !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0c1120; border-radius: 8px; border: 1px solid #182035;
    padding: 4px; gap: 2px; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    color: #3d506e; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    border-radius: 5px; padding: 7px 14px;
}
.stTabs [aria-selected="true"] {
    background: #f0b429 !important; color: #080c17 !important; border-radius: 5px;
}

.stButton > button {
    background: transparent; border: 1px solid #243352; color: #c8d0e0;
    border-radius: 6px; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 7px 18px; transition: all 0.15s;
}
.stButton > button:hover { border-color: #f0b429; color: #f0b429; background: rgba(240,180,41,0.06); }
.stButton > button[kind="primary"] { background: #f0b429; border-color: #f0b429; color: #080c17; font-weight: 800; }
.stButton > button[kind="primary"]:hover { background: #d9a025; border-color: #d9a025; }

.stTextInput input, .stTextArea textarea, [data-baseweb="select"] {
    background: #0f1524 !important; border: 1px solid #1b2844 !important;
    border-radius: 6px !important; color: #c8d0e0 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #f0b429 !important; border-color: #f0b429 !important; }

.stDataFrame { border: 1px solid #1b2844 !important; border-radius: 8px; }
.stDataFrame th { background: #0f1524 !important; color: #f0b429 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.06em; }
.stDataFrame td { font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; color: #8899bb !important; }

.page-title { font-family:'Syne',sans-serif; font-weight:800; font-size:30px;
    color:#e8edf5; letter-spacing:-0.03em; margin-bottom:0; }
.page-title span { color:#f0b429; }

.dash-card-title { font-family:'JetBrains Mono',monospace; font-size:11px;
    text-transform:uppercase; letter-spacing:0.1em; color:#f0b429; margin-bottom:8px; }

.ai-bubble { background:#0f1a30; border:1px solid #1b3355; border-left:3px solid #f0b429;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:12px 0;
    font-size:13px; color:#c8d8f0; line-height:1.7; }
.ai-label { font-family:'JetBrains Mono',monospace; font-size:10px; text-transform:uppercase;
    letter-spacing:0.1em; color:#f0b429; margin-bottom:6px; }

h3 { font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:12px !important; text-transform:uppercase !important;
    letter-spacing:0.1em !important; color:#384a68 !important;
    border-bottom:1px solid #182035 !important; padding-bottom:5px !important;
    margin-top:20px !important; }

.badge { display:inline-block; background:#182035; border:1px solid #243352;
    border-radius:20px; padding:3px 10px; font-family:'JetBrains Mono',monospace;
    font-size:10px; color:#8899bb; margin:2px; }
.badge-green { border-color:#22c55e; color:#22c55e; background:rgba(34,197,94,0.08); }
.badge-amber { border-color:#f0b429; color:#f0b429; background:rgba(240,180,41,0.08); }
.badge-red   { border-color:#ef4444; color:#ef4444; background:rgba(239,68,68,0.08); }

hr { border-color:#182035 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Safe":   px.colors.qualitative.Safe,
    "Dark24": px.colors.qualitative.Dark24,
}
SAMPLE_DATA = {
    "Sales": lambda: pd.DataFrame({
        "Month":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Sales":[12000,15000,13500,18000,21000,19500,22000,24000,20000,17000,25000,28000],
        "Profit":[2400,3200,2700,4000,5000,4500,5200,6100,4800,3900,6500,7800],
        "Customers":[120,145,138,167,189,175,195,212,181,160,230,260],
        "Region":["North","South"]*6,
        "Category":["A","B","A","C","B","A","C","B","A","C","B","A"],
    }),
    "Students": lambda: pd.DataFrame({
        "Student":[f"S{i:02d}" for i in range(1,31)],
        "Math":list(np.random.RandomState(42).randint(60,100,30)),
        "Science":list(np.random.RandomState(43).randint(55,100,30)),
        "English":list(np.random.RandomState(44).randint(65,100,30)),
        "Study_Hrs":list(np.random.RandomState(45).uniform(1,8,30).round(1)),
        "Grade":list(np.random.RandomState(46).choice(["A","B","C","D"],30,p=[.3,.4,.2,.1])),
        "School":list(np.random.RandomState(47).choice(["Public","Private"],30)),
    }),
    "E-Commerce": lambda: pd.DataFrame({
        "Date":[str(d.date()) for d in pd.date_range("2024-01-01",periods=90,freq="D")],
        "Revenue":list(np.cumsum(np.random.RandomState(42).normal(1000,200,90))+50000),
        "Orders":list(np.random.RandomState(43).randint(50,300,90)),
        "AOV":list(np.random.RandomState(44).uniform(30,120,90).round(2)),
        "Returns":list(np.random.RandomState(45).randint(0,30,90)),
        "Channel":list(np.random.RandomState(46).choice(["Organic","Paid","Email","Social"],90)),
    }),
    "World Cities": lambda: pd.DataFrame({
        "City":["New York","London","Tokyo","Paris","Sydney","Dubai","Singapore","Toronto","Berlin","Mumbai"],
        "Country":["USA","UK","Japan","France","Australia","UAE","Singapore","Canada","Germany","India"],
        "ISO":["USA","GBR","JPN","FRA","AUS","ARE","SGP","CAN","DEU","IND"],
        "Lat":[40.71,51.51,35.68,48.85,-33.87,25.20,1.35,43.65,52.52,19.08],
        "Lon":[-74.00,-0.13,139.69,2.35,151.21,55.27,103.82,-79.38,13.40,72.88],
        "Population":[8.3,9.0,13.9,2.1,5.3,3.3,5.9,2.9,3.7,20.7],
        "GDP_Index":[100,82,91,78,75,95,88,80,76,45],
        "Region":["Americas","Europe","Asia","Europe","Oceania","Middle East","Asia","Americas","Europe","Asia"],
    }),
}
UNIVARIATE_CHARTS = {
    "categorical": ["Bar Chart","Horizontal Bar","Pie Chart","Treemap"],
    "numerical":   ["Histogram","Box Plot","Violin Plot","Density Plot"],
    "datetime":    ["Line Chart","Area Chart","Bar Chart"],
    "boolean":     ["Bar Chart","Pie Chart"],
    "text":        ["Bar Chart (Top N)"],
}
BIVARIATE_CHARTS = {
    "categorical|categorical": ["Grouped Bar","Stacked Bar","Heatmap (Counts)"],
    "categorical|numerical":   ["Box Plot","Violin Plot","Strip Plot","Bar (Aggregated)"],
    "numerical|categorical":   ["Box Plot","Violin Plot","Strip Plot","Bar (Aggregated)"],
    "numerical|numerical":     ["Scatter Plot","Line Plot","Hexbin","2D Density"],
    "datetime|numerical":      ["Line Chart","Area Chart"],
    "numerical|datetime":      ["Line Chart","Area Chart"],
}
MULTIVARIATE_CHARTS = ["Correlation Heatmap","Pair Plot","Parallel Coordinates","3D Scatter"]
OLLAMA_URL = "http://localhost:11434"


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def col_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):            return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):  return "datetime"
    if pd.api.types.is_numeric_dtype(s):         return "numerical"
    if s.dtype == object:
        avg = s.dropna().astype(str).str.split().str.len().mean()
        return "text" if avg and avg > 4 else "categorical"
    return "unknown"

def num_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def cat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if col_kind(df[c]) in ("categorical","boolean")]

def quality_score(df: pd.DataFrame) -> float:
    s = 100 - (df.isna().mean().mean()*50) - (df.duplicated().mean()*30)
    return round(min(100.0, max(0.0, s)), 1)

def get_pal(name: str) -> list:
    return PALETTES.get(name, px.colors.qualitative.Plotly)

CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,23,0.7)",
    font=dict(family="JetBrains Mono, monospace", color="#8899bb"),
)

def apply_bg(fig, h: int = 460):
    fig.update_layout(height=h, **CHART_BG)
    return fig


# ══════════════════════════════════════════════════════════════════
# OLLAMA HELPERS  — FIX: streaming + split timeout
# ══════════════════════════════════════════════════════════════════
def ollama_available() -> bool:
    """Quick connectivity check — 2 s connect timeout only."""
    try:
        return _req.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except Exception:
        return False


def list_ollama_models() -> list:
    """Return list of locally pulled model names."""
    try:
        r = _req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


# ── FIX: was stream=False with timeout=120 → now stream=True with
#         split timeout=(connect=10 s, read=300 s).
#         Partial responses are preserved if a ReadTimeout still occurs.
def ollama_chat(model: str, system: str, user: str) -> str:
    """
    Stream a chat response from a local Ollama model.

    Split timeout:
        connect = 10 s  — fail fast if Ollama isn't reachable
        read    = 300 s — give the model up to 5 min to finish streaming
    """
    collected: list[str] = []
    try:
        with _req.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": True,                          # ← was False
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
            timeout=(10, 300),                           # ← was timeout=120
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        collected.append(delta)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        return "".join(collected).strip() or "⚠️ Empty response from model."

    except _req.exceptions.ConnectionError:
        return "❌ Cannot reach Ollama. Make sure `ollama serve` is running."

    except _req.exceptions.Timeout:
        # Return whatever arrived before the timeout instead of losing it
        partial = "".join(collected).strip()
        if partial:
            return partial + "\n\n*(response may be incomplete — read timeout)*"
        return (
            "⏱️ Timed out waiting for Ollama. "
            "Try a smaller/faster model such as `phi3` or `gemma2`."
        )

    except Exception as exc:
        return f"Ollama error: {exc}"


def extract_json(text: str) -> Optional[dict]:
    try:
        clean = re.sub(r"```(?:json)?|```", "", text).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# CHART RENDERER
# ══════════════════════════════════════════════════════════════════
def render_chart(df, chart_type, x_col, y_col=None, color_by=None,
                 agg="sum", palette_name="Plotly"):
    p   = get_pal(palette_name)
    t   = "plotly_dark"
    fig = None
    y   = y_col   if y_col   and y_col   in df.columns else None
    col = color_by if color_by and color_by in df.columns else None
    try:
        if chart_type == "Bar Chart":
            if y:
                grp=[x_col]+([col] if col else [])
                gdf=df.groupby(grp,observed=True)[y].agg(agg).reset_index()
                fig=px.bar(gdf,x=x_col,y=y,color=col,barmode="group",color_discrete_sequence=p,template=t)
            else:
                vc=df[x_col].value_counts().reset_index(); vc.columns=[x_col,"count"]
                fig=px.bar(vc,x=x_col,y="count",color_discrete_sequence=p,template=t)
        elif chart_type == "Horizontal Bar":
            if y:
                grp=[x_col]+([col] if col else [])
                gdf=df.groupby(grp,observed=True)[y].agg(agg).reset_index()
                fig=px.bar(gdf,x=y,y=x_col,color=col,orientation="h",color_discrete_sequence=p,template=t)
            else:
                vc=df[x_col].value_counts().reset_index(); vc.columns=[x_col,"count"]
                fig=px.bar(vc,x="count",y=x_col,orientation="h",color_discrete_sequence=p,template=t)
        elif chart_type == "Pie Chart":
            vc=df[x_col].value_counts().head(12)
            fig=px.pie(names=vc.index,values=vc.values,hole=0.4,color_discrete_sequence=p)
        elif chart_type == "Treemap":
            vc=df[x_col].value_counts().reset_index(); vc.columns=[x_col,"count"]
            fig=px.treemap(vc,path=[x_col],values="count",color_discrete_sequence=p)
        elif chart_type == "Histogram":
            fig=px.histogram(df,x=x_col,color=col,nbins=30,color_discrete_sequence=p,template=t)
        elif chart_type == "Box Plot":
            y_use=y if y else x_col
            fig=px.box(df,x=col or None,y=y_use,color=col or None,color_discrete_sequence=p,template=t)
        elif chart_type == "Violin Plot":
            y_use=y if y else x_col
            fig=px.violin(df,x=col or None,y=y_use,color=col or None,box=True,color_discrete_sequence=p,template=t)
        elif chart_type == "Density Plot":
            fig=px.histogram(df,x=x_col,color=col,marginal="rug",histnorm="density",nbins=40,color_discrete_sequence=p,template=t)
        elif chart_type in ("Line Chart","Line Plot"):
            if not y: st.warning("Y column required."); return None
            fig=px.line(df.sort_values(x_col),x=x_col,y=y,color=col,markers=True,color_discrete_sequence=p,template=t)
        elif chart_type == "Area Chart":
            if not y: st.warning("Y column required."); return None
            fig=px.area(df.sort_values(x_col),x=x_col,y=y,color=col,color_discrete_sequence=p,template=t)
        elif chart_type == "Bar Chart (Top N)":
            vc=df[x_col].value_counts().head(20).reset_index(); vc.columns=[x_col,"count"]
            fig=px.bar(vc,x=x_col,y="count",color_discrete_sequence=p,template=t)
        elif chart_type == "Scatter Plot":
            if not y: st.warning("Y column required."); return None
            fig=px.scatter(df,x=x_col,y=y,color=col,trendline="ols" if not col else None,color_discrete_sequence=p,template=t)
        elif chart_type == "Hexbin":
            if not y: st.warning("Y column required."); return None
            fig=px.density_heatmap(df,x=x_col,y=y,nbinsx=20,nbinsy=20,color_continuous_scale="Cividis",template=t)
        elif chart_type == "2D Density":
            if not y: st.warning("Y column required."); return None
            fig=px.density_contour(df,x=x_col,y=y,color=col,color_discrete_sequence=p,template=t)
            fig.update_traces(contours_coloring="fill",contours_showlabels=True)
        elif chart_type == "Strip Plot":
            y_use=y if y else x_col
            fig=px.strip(df,x=col or None,y=y_use,color=col or None,color_discrete_sequence=p,template=t)
        elif chart_type == "Grouped Bar":
            if not y or not col: st.warning("Y + Colour-by required."); return None
            gdf=df.groupby([x_col,col],observed=True)[y].agg(agg).reset_index()
            fig=px.bar(gdf,x=x_col,y=y,color=col,barmode="group",color_discrete_sequence=p,template=t)
        elif chart_type == "Stacked Bar":
            if not y or not col: st.warning("Y + Colour-by required."); return None
            gdf=df.groupby([x_col,col],observed=True)[y].agg(agg).reset_index()
            fig=px.bar(gdf,x=x_col,y=y,color=col,barmode="stack",color_discrete_sequence=p,template=t)
        elif chart_type == "Heatmap (Counts)":
            if not y: st.warning("Y column required."); return None
            ct=pd.crosstab(df[x_col],df[y])
            fig=px.imshow(ct,text_auto=True,color_continuous_scale="Blues",template=t)
        elif chart_type == "Bar (Aggregated)":
            if not y: st.warning("Y column required."); return None
            grp=[x_col]+([col] if col else [])
            gdf=df.groupby(grp,observed=True)[y].agg(agg).reset_index()
            fig=px.bar(gdf,x=x_col,y=y,color=col,color_discrete_sequence=p,template=t)
        elif chart_type == "Correlation Heatmap":
            nc=num_cols(df)
            if len(nc)<2: st.warning("Need ≥2 numeric columns."); return None
            corr=df[nc].corr()
            fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,template=t)
        elif chart_type == "Pair Plot":
            nc=num_cols(df)
            if len(nc)<2: st.warning("Need ≥2 numeric columns."); return None
            fig=px.scatter_matrix(df,dimensions=nc[:5],color=col,color_discrete_sequence=p,template=t)
            fig.update_traces(diagonal_visible=False,marker=dict(size=3,opacity=0.5))
        elif chart_type == "Parallel Coordinates":
            nc=num_cols(df)
            if not nc: st.warning("No numeric columns."); return None
            fig=px.parallel_coordinates(df,dimensions=nc[:8],color_continuous_scale=px.colors.sequential.Plasma,template=t)
        elif chart_type == "3D Scatter":
            nc=num_cols(df)
            if len(nc)<3: st.warning("Need ≥3 numeric columns."); return None
            fig=px.scatter_3d(df,x=nc[0],y=nc[1],z=nc[2],color=col,color_discrete_sequence=p,template=t)
        else:
            st.warning(f"Unknown chart type: {chart_type}"); return None
    except Exception as e:
        st.error(f"Chart error: {e}"); return None
    if fig:
        apply_bg(fig)
    return fig


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for _k, _v in {"df": None, "history": [], "dashboard": []}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def get_df(): return st.session_state.df

def save_snapshot(label: str):
    st.session_state.history = st.session_state.history[-9:]
    st.session_state.history.append((label, st.session_state.df.copy()))

def pin_to_dashboard(title: str, fig) -> None:
    fig_json = json.loads(fig.to_json())
    titles   = [c["title"] for c in st.session_state.dashboard]
    entry    = {"title": title, "fig_json": fig_json}
    if title in titles:
        st.session_state.dashboard[titles.index(title)] = entry
    else:
        st.session_state.dashboard.append(entry)

def pin_button(key: str, title: str, fig) -> None:
    already = any(c["title"] == title for c in st.session_state.dashboard)
    label   = "✅ Saved to Dashboard" if already else "📌 Save to Dashboard"
    if st.button(label, key=key):
        pin_to_dashboard(title, fig)
        st.toast(f"📌 '{title}' pinned!", icon="✅")
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ◈ DataLens")
    st.markdown("---")

    source = st.radio("Load data from",
                      ["Sample dataset","Upload file","URL"],
                      label_visibility="collapsed")
    if source == "Upload file":
        f = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
        if f and st.button("Load file", type="primary"):
            try:
                st.session_state.df = (pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f))
                st.session_state.history = []
                st.success(f"Loaded {f.name}"); st.rerun()
            except Exception as e:
                st.error(f"Read error: {e}")
    elif source == "URL":
        url_in = st.text_input("CSV URL", placeholder="https://example.com/data.csv", label_visibility="collapsed")
        if st.button("Load URL", type="primary") and url_in:
            try:
                st.session_state.df = pd.read_csv(url_in)
                st.session_state.history = []
                st.success("Loaded from URL"); st.rerun()
            except Exception as e:
                st.error(f"Load error: {e}")
    else:
        choice = st.selectbox("Dataset", list(SAMPLE_DATA.keys()))
        if st.button("Load dataset", type="primary"):
            st.session_state.df = SAMPLE_DATA[choice]()
            st.session_state.history = []
            st.success(f"Loaded: {choice}"); st.rerun()

    if st.session_state.df is None:
        st.info("Load a dataset to begin.")
        st.stop()

    df = get_df()
    q  = quality_score(df)
    nc = num_cols(df)
    cc = cat_cols(df)
    qc = "#22c55e" if q>80 else "#f0b429" if q>60 else "#ef4444"

    st.markdown("---")
    st.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#384a68'>"
        f"{len(df):,} rows  ×  {len(df.columns)} cols</span><br>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:{qc}'>"
        f"quality {q}/100</span>", unsafe_allow_html=True)

    st.markdown("---")
    palette_name = st.selectbox("Palette", list(PALETTES.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🤖 Local AI (Ollama)**")
    _ol_ok  = ollama_available()
    if _ol_ok:
        st.markdown('<span class="badge badge-green">● Ollama running</span>', unsafe_allow_html=True)
        _models = list_ollama_models()
        if _models:
            ollama_model = st.selectbox("Model", _models, label_visibility="collapsed")
        else:
            st.markdown('<span class="badge badge-red">No models pulled</span>', unsafe_allow_html=True)
            st.caption("Run: `ollama pull mistral`")
            ollama_model = ""
    else:
        st.markdown('<span class="badge badge-red">○ Ollama offline</span>', unsafe_allow_html=True)
        st.caption("Start: `ollama serve`")
        ollama_model = ""

    st.markdown("---")
    n_pinned = len(st.session_state.dashboard)
    st.markdown(
        f'<span class="badge badge-amber">📌 {n_pinned} chart{"s" if n_pinned!=1 else ""} pinned</span>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN HEADER + METRICS
# ══════════════════════════════════════════════════════════════════
df = get_df(); nc = num_cols(df); cc = cat_cols(df)

st.markdown('<p class="page-title">Data<span>Lens</span></p>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:JetBrains Mono,monospace;font-size:11px;"
    "color:#243352;margin-top:-4px;margin-bottom:18px'>"
    "smart visualization toolkit  ·  v4.1</p>", unsafe_allow_html=True)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Rows",       f"{len(df):,}")
m2.metric("Columns",    len(df.columns))
m3.metric("Quality",    f"{quality_score(df)}/100")
m4.metric("Missing",    f"{int(df.isna().sum().sum()):,}")
m5.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
_dash_lbl = f"📊 Dashboard ({len(st.session_state.dashboard)})" \
            if st.session_state.dashboard else "📊 Dashboard"

TABS = st.tabs([
    "📋 Overview", "📈 Charts", "💬 Ask Data", "🤖 Ask AI",
    "🔄 Pivot", "🔵 Clustering", "📐 Trendlines", "🧮 Calc Columns",
    "🧹 Clean", "🔔 Alerts", "🗺️ Maps", "📤 Export", _dash_lbl,
])


# ══════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with TABS[0]:
    st.markdown("### Dataset Preview")
    n_rows = st.slider("Rows to show", 5, min(200, len(df)), 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("### Column Info")
    st.dataframe(pd.DataFrame([{
        "Column": c, "Dtype": str(df[c].dtype), "Kind": col_kind(df[c]),
        "Non-Null": int(df[c].count()), "Missing %": round(df[c].isna().mean()*100,1),
        "Unique": int(df[c].nunique()),
    } for c in df.columns]), use_container_width=True, hide_index=True)

    if nc:
        st.markdown("### Numeric Statistics")
        st.dataframe(df[nc].describe().round(2), use_container_width=True)

    miss = {c: round(df[c].isna().mean()*100,1) for c in df.columns if df[c].isna().any()}
    if miss:
        st.markdown("### Missing Values")
        fig_m = px.bar(x=list(miss.keys()), y=list(miss.values()),
                       labels={"x":"Column","y":"Missing %"},
                       color=list(miss.values()), color_continuous_scale="Oranges",
                       template="plotly_dark")
        fig_m.update_layout(height=260, showlegend=False, coloraxis_showscale=False, **CHART_BG)
        st.plotly_chart(fig_m, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 1 — CHARTS  (3 sub-tabs)
# ══════════════════════════════════════════════════════════════════
with TABS[1]:
    st.markdown("### Chart Builder")
    UV, BV, MV = st.tabs(["📊 Univariate", "🔀 Bivariate", "🌐 Multivariate"])

    # ── Univariate ──────────────────────────────────────────────
    with UV:
        st.caption("One column — distribution, counts, proportions.")
        u_x     = st.selectbox("Column", df.columns.tolist(), key="uv_x")
        kind    = col_kind(df[u_x])
        allowed = UNIVARIATE_CHARTS.get(kind, [])
        st.markdown(
            f'<span class="badge">{kind}</span> → '
            + " ".join(f'<span class="badge">{c}</span>' for c in allowed),
            unsafe_allow_html=True)
        if not allowed:
            st.warning("No chart types for this column kind.")
        else:
            u_chart = st.selectbox("Chart type", allowed, key="uv_chart")
            c1,c2,c3 = st.columns(3)
            needs_y = u_chart in ("Line Chart","Area Chart","Bar Chart")
            u_y = u_col = None; u_agg = "sum"
            if needs_y:
                u_y = c1.selectbox("Y axis (numeric)", nc, key="uv_y") if nc else None
            else:
                y_raw = c1.selectbox("Y axis (optional)", ["(none)"]+nc, key="uv_yopt")
                u_y   = None if y_raw=="(none)" else y_raw
            cr    = c2.selectbox("Colour by", ["(none)"]+cc, key="uv_col")
            u_col = None if cr=="(none)" else cr
            if u_y:
                u_agg = c3.selectbox("Aggregation", ["sum","mean","count","min","max"], key="uv_agg")
            if st.button("Generate", type="primary", key="uv_btn"):
                fig = render_chart(df, u_chart, u_x, u_y, u_col, u_agg, palette_name)
                if fig:
                    st.session_state["_last_uv"] = (fig, f"{u_chart}: {u_x}")
            if "_last_uv" in st.session_state:
                fig, title = st.session_state["_last_uv"]
                st.plotly_chart(fig, use_container_width=True)
                pin_button("uv_pin", title, fig)

    # ── Bivariate ───────────────────────────────────────────────
    with BV:
        st.caption("Two columns — relationships and comparisons.")
        cols_all = df.columns.tolist()
        b1, b2   = st.columns(2)
        bv_x = b1.selectbox("X column", cols_all, key="bv_x")
        bv_y = b2.selectbox("Y column", cols_all, index=min(1,len(cols_all)-1), key="bv_y")
        if bv_x == bv_y:
            st.warning("X and Y must be different.")
        else:
            kx = col_kind(df[bv_x]); ky = col_kind(df[bv_y])
            allowed_bv = BIVARIATE_CHARTS.get(f"{kx}|{ky}", [])
            st.markdown(
                f'<span class="badge">{bv_x} ({kx})</span> × '
                f'<span class="badge">{bv_y} ({ky})</span> → '
                + (" ".join(f'<span class="badge">{c}</span>' for c in allowed_bv)
                   if allowed_bv else '<span class="badge badge-red">no match</span>'),
                unsafe_allow_html=True)
            if not allowed_bv:
                st.warning("No charts defined for this column-type pair.")
            else:
                bv_chart = st.selectbox("Chart type", allowed_bv, key="bv_chart")
                c1, c2   = st.columns(2)
                cr       = c1.selectbox("Colour by", ["(none)"]+cc, key="bv_col")
                bv_col   = None if cr=="(none)" else cr
                bv_agg   = c2.selectbox("Aggregation", ["mean","sum","count","min","max"], key="bv_agg")
                if st.button("Generate", type="primary", key="bv_btn"):
                    fig = render_chart(df, bv_chart, bv_x, bv_y, bv_col, bv_agg, palette_name)
                    if fig:
                        st.session_state["_last_bv"] = (fig, f"{bv_chart}: {bv_x} × {bv_y}")
                if "_last_bv" in st.session_state:
                    fig, title = st.session_state["_last_bv"]
                    st.plotly_chart(fig, use_container_width=True)
                    pin_button("bv_pin", title, fig)

    # ── Multivariate ────────────────────────────────────────────
    with MV:
        st.caption("All (or many) columns at once — structure and correlations.")
        mv_chart = st.selectbox("Chart type", MULTIVARIATE_CHARTS, key="mv_chart")
        cr       = st.selectbox("Colour by (optional)", ["(none)"]+cc, key="mv_col")
        mv_col   = None if cr=="(none)" else cr
        if st.button("Generate", type="primary", key="mv_btn"):
            fig = render_chart(df, mv_chart, df.columns[0], None, mv_col, "mean", palette_name)
            if fig:
                st.session_state["_last_mv"] = (fig, mv_chart)
        if "_last_mv" in st.session_state:
            fig, title = st.session_state["_last_mv"]
            st.plotly_chart(fig, use_container_width=True)
            pin_button("mv_pin", title, fig)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — ASK DATA  (keyword NLQ)
# ══════════════════════════════════════════════════════════════════
with TABS[2]:
    st.markdown("### Ask a Question About Your Data")
    st.caption("Keyword-based NLQ — works fully offline. No AI required.")

    query = st.text_input("Your question",
                          placeholder="e.g. Show total Sales by Region", key="nlq_input")
    if st.button("Run", type="primary", key="nlq_btn") and query:
        q_l = query.lower()
        agg_fn = ("mean" if any(w in q_l for w in ["average","avg","mean"])
                  else "count" if "count" in q_l
                  else "max" if "max" in q_l
                  else "min" if "min" in q_l
                  else "sum")
        chart_t = ("Line Chart" if any(w in q_l for w in ["trend","line","over","time"])
                   else "Scatter Plot" if any(w in q_l for w in ["scatter","vs","versus"])
                   else "Pie Chart"    if any(w in q_l for w in ["pie","share","proportion"])
                   else "Histogram"    if any(w in q_l for w in ["distribution","histogram"])
                   else "Bar Chart")
        top_m = re.search(r"top\s+(\d+)", q_l)
        top_n = int(top_m.group(1)) if top_m else None

        found = [c for c in sorted(df.columns, key=len, reverse=True) if c.lower() in q_l]
        y_c   = next((c for c in found if c in nc), None) or (nc[0] if nc else None)
        x_c   = next((c for c in found if c != y_c), None) or (cc[0] if cc else None)

        if not x_c:
            st.error("Could not identify columns — mention a column name.")
        else:
            plot_df = df.copy()
            if top_n and y_c:
                grp     = plot_df.groupby(x_c, observed=True)[y_c].agg(agg_fn).reset_index()
                plot_df = grp.sort_values(y_c, ascending=False).head(top_n)
            st.caption(f"x=`{x_c}` · y=`{y_c}` · agg=`{agg_fn}` · chart=`{chart_t}`")
            fig = render_chart(plot_df, chart_t, x_c, y_c, None, agg_fn, palette_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                pin_button("nlq_pin", f"Ask: {query[:40]}", fig)
                with st.expander("Data table"):
                    show = [c for c in [x_c,y_c] if c and c in plot_df.columns]
                    st.dataframe(plot_df[show].head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ASK AI  (Ollama — FIX: placeholder + streaming)
# ══════════════════════════════════════════════════════════════════
with TABS[3]:
    st.markdown("### Ask AI — Local LLM via Ollama")
    st.caption("Runs on your machine. No API key, no cloud, fully private.")

    if not ollama_available():
        st.error(
            "**Ollama is not running.**\n\n"
            "**Step 1 — Install:** `curl -fsSL https://ollama.ai/install.sh | sh`\n\n"
            "**Step 2 — Pull a model:** `ollama pull mistral`  (or `llama3`, `phi3`, `gemma2`)\n\n"
            "**Step 3 — Start:** `ollama serve`\n\n"
            "Then reload this page."
        )
    elif not ollama_model:
        st.warning(
            "No models found.\n\n"
            "Pull one in your terminal:\n"
            "`ollama pull mistral`  or  `ollama pull llama3`  or  `ollama pull phi3`"
        )
    else:
        # ── Shared context sent with every request ──────────────
        col_summary = "\n".join(
            f"  {c}: {col_kind(df[c])} (dtype={df[c].dtype}, "
            f"unique={df[c].nunique()}, missing={df[c].isna().sum()})"
            for c in df.columns
        )
        stats_txt  = df[nc].describe().round(2).to_string() if nc else "No numeric columns."
        sample_txt = df.head(5).fillna("").astype(str).to_string(index=False)
        system_ctx = (
            f"You are an expert data analyst assistant.\n"
            f"Dataset: {len(df)} rows × {len(df.columns)} columns.\n\n"
            f"COLUMNS:\n{col_summary}\n\n"
            f"NUMERIC STATS:\n{stats_txt}\n\n"
            f"SAMPLE (first 5 rows):\n{sample_txt}\n\n"
            "Answer concisely and accurately. Name chart types explicitly when recommending them."
        )

        ai_mode = st.radio("Mode", ["💬 Chat","📊 Generate Chart from description"],
                           horizontal=True, key="ai_mode")

        # ── Chat mode ──────────────────────────────────────────
        if ai_mode == "💬 Chat":
            ai_q = st.text_area(
                "Your question",
                placeholder=(
                    "• What are the main trends in this dataset?\n"
                    "• Which columns are most correlated?\n"
                    "• Are there data quality issues I should fix?\n"
                    "• What chart best shows Revenue over time?"
                ),
                height=110, key="ai_chat_q",
            )
            st.markdown("**Quick questions:**")
            suggestions = [
                "Summarise the key statistics",
                "What are the main trends?",
                "Any data quality issues?",
                "Which columns to visualise first?",
            ]
            sc = st.columns(len(suggestions))
            picked = ""
            for i, s in enumerate(suggestions):
                if sc[i].button(s, key=f"sug_{i}", use_container_width=True):
                    picked = s
            question = picked or ai_q

            # FIX: replaced st.spinner (blocks silently) with st.empty()
            # placeholder that shows a visible status message while streaming.
            if (st.button("Ask", type="primary", key="ai_chat_btn") or picked) and question:
                placeholder = st.empty()
                placeholder.info(
                    f"🤖 **{ollama_model}** is thinking…  "
                    f"*(streaming — this can take 30–90 s for larger models)*"
                )
                answer = ollama_chat(ollama_model, system_ctx, question)
                placeholder.empty()
                st.markdown(
                    f'<div class="ai-bubble"><div class="ai-label">🤖 {ollama_model}</div>'
                    f'{answer}</div>', unsafe_allow_html=True)

        # ── Chart generation mode ──────────────────────────────
        else:
            chart_q = st.text_input(
                "Describe the chart you want",
                placeholder="e.g. Show total Profit by Region as a bar chart",
                key="ai_chart_q",
            )
            if st.button("Generate Chart", type="primary", key="ai_chart_btn") and chart_q:
                col_list = ", ".join(f"{c}({col_kind(df[c])})" for c in df.columns)
                spec_sys = (
                    f"You are a data analyst. Dataset columns: {col_list}.\n"
                    "Respond ONLY with valid JSON, no explanation:\n"
                    '{"chart_type":"<type>","x_col":"<col>","y_col":"<col or null>",'
                    '"color_by":"<col or null>","aggregation":"sum|mean|count|min|max",'
                    '"top_n":<int or null>}\n'
                    "Valid chart types: Bar Chart, Horizontal Bar, Pie Chart, Treemap, "
                    "Histogram, Box Plot, Violin Plot, Line Chart, Area Chart, Scatter Plot, "
                    "Grouped Bar, Stacked Bar, Correlation Heatmap, Pair Plot, 3D Scatter."
                )
                # FIX: placeholder instead of st.spinner
                ph = st.empty()
                ph.info(
                    f"🤖 **{ollama_model}** building chart spec…  "
                    f"*(may take up to 60 s)*"
                )
                raw = ollama_chat(ollama_model, spec_sys, chart_q)
                ph.empty()

                spec = extract_json(raw)
                if not spec:
                    st.error(f"Could not parse chart spec. Model responded:\n\n{raw}")
                else:
                    for fld in ["x_col","y_col","color_by"]:
                        if spec.get(fld) and spec[fld] not in df.columns:
                            spec[fld] = None
                    x_c = spec.get("x_col") or df.columns[0]
                    y_c = spec.get("y_col")
                    cb  = spec.get("color_by")
                    ct  = spec.get("chart_type","Bar Chart")
                    ag  = spec.get("aggregation","sum")
                    tn  = spec.get("top_n")
                    plot_df = df.copy()
                    if tn and y_c and y_c in df.columns:
                        grp = plot_df.groupby(x_c,observed=True)[y_c].agg(ag).reset_index()
                        plot_df = grp.sort_values(y_c, ascending=False).head(int(tn))
                    with st.expander("🔍 LLM chart spec"):
                        st.json(spec)
                    fig = render_chart(plot_df, ct, x_c, y_c, cb, ag, palette_name)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        pin_button("ai_chart_pin", f"AI: {chart_q[:40]}", fig)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — PIVOT
# ══════════════════════════════════════════════════════════════════
with TABS[4]:
    st.markdown("### Pivot Table")
    p1,p2,p3,p4 = st.columns(4)
    rows_p = p1.multiselect("Rows",    df.columns.tolist(), default=[df.columns[0]])
    cols_p = p2.multiselect("Columns", df.columns.tolist())
    vals_p = p3.multiselect("Values",  nc, default=nc[:1] if nc else [])
    agg_p  = p4.selectbox("Aggregation", ["sum","mean","count","min","max","median"])
    if st.button("Build Pivot", type="primary") and rows_p and vals_p:
        try:
            col_arg = cols_p[0] if len(cols_p)==1 else (cols_p or None)
            val_arg = vals_p[0] if len(vals_p)==1 else vals_p
            pivot   = pd.pivot_table(df, index=rows_p, columns=col_arg, values=val_arg,
                                     aggfunc=agg_p, margins=True, margins_name="Total",
                                     observed=True).round(2)
            if isinstance(pivot.columns, pd.MultiIndex):
                pivot.columns = ["_".join(str(s) for s in c).strip("_") for c in pivot.columns]
            else:
                pivot.columns = [str(c) for c in pivot.columns]
            st.dataframe(pivot.reset_index().style.background_gradient(cmap="Blues", axis=None),
                         use_container_width=True)
        except Exception as e:
            st.error(f"Pivot error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — CLUSTERING
# ══════════════════════════════════════════════════════════════════
with TABS[5]:
    st.markdown("### K-Means Clustering")
    if not SKLEARN:
        st.error("Install scikit-learn: `pip install scikit-learn`")
    elif len(nc) < 2:
        st.warning("Need ≥2 numeric columns.")
    else:
        c1,c2 = st.columns(2)
        feats  = c1.multiselect("Feature columns", nc, default=nc[:min(3,len(nc))])
        k_val  = c2.slider("Clusters (K)", 2, 8, 3)
        vx     = c1.selectbox("Plot X", nc, key="cl_x")
        vy     = c2.selectbox("Plot Y", nc, index=min(1,len(nc)-1), key="cl_y")
        scale  = st.checkbox("Standardise features", value=True)
        if st.button("Run Clustering", type="primary") and len(feats) >= 2:
            try:
                sub    = df[feats].dropna()
                X      = StandardScaler().fit_transform(sub) if scale else sub.values
                labels = KMeans(n_clusters=k_val, random_state=42, n_init="auto").fit_predict(X)
                out    = df.loc[sub.index].copy()
                out["Cluster"] = [f"C{l+1}" for l in labels]
                p_list = get_pal(palette_name)
                fig_s  = apply_bg(px.scatter(out, x=vx, y=vy, color="Cluster",
                                             color_discrete_sequence=p_list, template="plotly_dark",
                                             title=f"{vx} vs {vy} — Clusters"))
                sizes           = out["Cluster"].value_counts().reset_index()
                sizes.columns   = ["Cluster","Count"]
                fig_b           = apply_bg(px.bar(sizes, x="Cluster", y="Count", color="Cluster",
                                                  color_discrete_sequence=p_list, template="plotly_dark",
                                                  title="Cluster Sizes"))
                fig_b.update_layout(showlegend=False)
                ca,cb = st.columns(2)
                with ca:
                    st.plotly_chart(fig_s, use_container_width=True)
                    pin_button("cl_pin", f"Cluster: {vx}×{vy} k={k_val}", fig_s)
                with cb:
                    st.plotly_chart(fig_b, use_container_width=True)
                st.markdown("### Cluster Means")
                st.dataframe(out.groupby("Cluster",observed=True)[feats].mean().round(2),
                             use_container_width=True)
            except Exception as e:
                st.error(f"Clustering error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — TRENDLINES
# ══════════════════════════════════════════════════════════════════
with TABS[6]:
    st.markdown("### Advanced Trendlines")
    if len(nc) < 2:
        st.warning("Need ≥2 numeric columns.")
    else:
        c1,c2,c3 = st.columns(3)
        tx    = c1.selectbox("X column", nc, key="tl_x")
        ty    = c2.selectbox("Y column", nc, index=min(1,len(nc)-1), key="tl_y")
        ttype = c3.selectbox("Type", ["Linear","Polynomial","Exponential","All"])
        deg   = st.slider("Polynomial degree", 2, 5, 2) if ttype=="Polynomial" else 3

        if st.button("Fit Trendline", type="primary"):
            sub = df[[tx,ty]].dropna()
            if len(sub) < 4:
                st.error("Need ≥4 non-null rows.")
            else:
                x_v = sub[tx].values.astype(float); y_v = sub[ty].values.astype(float)
                xs  = np.linspace(x_v.min(), x_v.max(), 300)
                p_l = get_pal(palette_name)
                denom = max(float(np.sum((y_v-y_v.mean())**2)), 1e-10)
                r2v: dict = {}
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=list(x_v), y=list(y_v), mode="markers", name="Data",
                                           marker=dict(color=p_l[0], size=5, opacity=0.6)))
                def _add(xs_,ys_,name,color,dash="solid"):
                    fig_t.add_trace(go.Scatter(x=list(xs_),y=list(ys_),mode="lines",name=name,
                                               line=dict(color=color,width=2,dash=dash)))
                if ttype in ("Linear","All"):
                    c_=np.polyfit(x_v,y_v,1); r2=1-np.sum((y_v-np.polyval(c_,x_v))**2)/denom
                    r2v["Linear"]=round(float(r2),4)
                    _add(xs,np.polyval(c_,xs),f"Linear R²={r2:.3f}",p_l[1] if len(p_l)>1 else "#EF4444")
                if ttype in ("Polynomial","All"):
                    d=deg if ttype=="Polynomial" else 3; c_=np.polyfit(x_v,y_v,d)
                    r2=1-np.sum((y_v-np.polyval(c_,x_v))**2)/denom
                    r2v[f"Poly deg-{d}"]=round(float(r2),4)
                    _add(xs,np.polyval(c_,xs),f"Poly deg-{d} R²={r2:.3f}",p_l[2] if len(p_l)>2 else "#F59E0B","dash")
                if ttype in ("Exponential","All") and (y_v>0).all():
                    try:
                        lc=np.polyfit(x_v,np.log(y_v),1); a,b=float(np.exp(lc[1])),float(lc[0])
                        yp=a*np.exp(b*x_v); r2=1-np.sum((y_v-yp)**2)/denom
                        r2v["Exponential"]=round(float(r2),4)
                        _add(xs,a*np.exp(b*xs),f"Exp R²={r2:.3f}",p_l[3] if len(p_l)>3 else "#10B981","dot")
                    except Exception: pass
                apply_bg(fig_t, h=440)
                fig_t.update_layout(xaxis_title=tx, yaxis_title=ty, template="plotly_dark",
                                    legend=dict(orientation="h",y=1.05))
                st.plotly_chart(fig_t, use_container_width=True)
                pin_button("tl_pin", f"Trendline ({ttype}): {tx}×{ty}", fig_t)
                if r2v:
                    rc=st.columns(len(r2v))
                    for i,(k,v) in enumerate(r2v.items()): rc[i].metric(k, v)


# ══════════════════════════════════════════════════════════════════
# TAB 7 — CALCULATED COLUMNS
# ══════════════════════════════════════════════════════════════════
with TABS[7]:
    st.markdown("### Add a Calculated Column")
    st.caption("Use column names in the formula. `np` is available.")
    col_name_new = st.text_input("New column name", placeholder="e.g. Profit_Margin")
    formula_in   = st.text_input("Formula", placeholder="e.g. Profit / Sales * 100")
    if nc:
        st.markdown(
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:#384a68'>"
            "Numeric: " + " · ".join(nc) + "</span>", unsafe_allow_html=True)
    if st.button("Add Column", type="primary") and col_name_new and formula_in:
        if col_name_new in df.columns:
            st.error(f"'{col_name_new}' already exists.")
        else:
            ns: dict = {}
            for c in df.columns:
                ns[c]=df[c]; ns[c.replace(" ","_")]=df[c]
            ns["np"]=np
            formula_clean=re.sub(r"`([^`]+)`",lambda m:m.group(1).replace(" ","_"),formula_in)
            try:
                result=eval(formula_clean,{"__builtins__":{}},ns)  # noqa
                save_snapshot(f"Add col: {col_name_new}")
                st.session_state.df[col_name_new]=result
                st.success(f"Added '{col_name_new}'"); st.rerun()
            except Exception as e:
                st.error(f"Formula error: {e}")
    st.markdown("---"); st.markdown("### Current Columns")
    st.dataframe(pd.DataFrame([{
        "Column":c,"Kind":col_kind(df[c]),"Dtype":str(df[c].dtype),"Unique":int(df[c].nunique())
    } for c in df.columns]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 8 — CLEAN
# ══════════════════════════════════════════════════════════════════
with TABS[8]:
    st.markdown("### Data Cleaning")
    s1,s2,s3,s4 = st.tabs(["Duplicates","Missing","Rename / Drop","Type Cast"])
    with s1:
        n_dup=int(df.duplicated().sum()); st.metric("Duplicate rows", n_dup)
        if n_dup>0:
            if st.button("Remove duplicates", type="primary"):
                save_snapshot("Remove duplicates")
                st.session_state.df=df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {n_dup} rows"); st.rerun()
        else: st.success("No duplicates.")
    with s2:
        miss_c=[c for c in df.columns if df[c].isna().any()]
        if not miss_c: st.success("No missing values.")
        else:
            c1,c2=st.columns(2)
            col_f=c1.selectbox("Column",miss_c)
            strat=c2.selectbox("Strategy",["Mean","Median","Mode","Forward fill","Drop rows","Custom"])
            custom=st.text_input("Custom value") if strat=="Custom" else None
            if st.button("Apply", type="primary"):
                save_snapshot(f"Fill {col_f}"); df2=st.session_state.df.copy()
                try:
                    if strat=="Mean":         df2[col_f]=df2[col_f].fillna(df2[col_f].mean())
                    elif strat=="Median":     df2[col_f]=df2[col_f].fillna(df2[col_f].median())
                    elif strat=="Mode":
                        mv=df2[col_f].mode()
                        if not mv.empty: df2[col_f]=df2[col_f].fillna(mv[0])
                    elif strat=="Forward fill": df2[col_f]=df2[col_f].ffill()
                    elif strat=="Drop rows":    df2=df2.dropna(subset=[col_f])
                    elif strat=="Custom":
                        try: df2[col_f]=df2[col_f].fillna(float(custom))
                        except: df2[col_f]=df2[col_f].fillna(custom)
                    st.session_state.df=df2; st.success("Done"); st.rerun()
                except Exception as e: st.error(f"Error: {e}")
    with s3:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Rename**")
            old_c=st.selectbox("Column",df.columns.tolist(),key="ren_old")
            new_c=st.text_input("New name",key="ren_new")
            if st.button("Rename") and new_c:
                if new_c in df.columns: st.error(f"'{new_c}' exists.")
                else:
                    save_snapshot(f"Rename {old_c}")
                    st.session_state.df=df.rename(columns={old_c:new_c})
                    st.success("Renamed"); st.rerun()
        with c2:
            st.markdown("**Drop**")
            drop_c=st.multiselect("Columns to drop",df.columns.tolist())
            if st.button("Drop") and drop_c:
                save_snapshot(f"Drop {drop_c}")
                st.session_state.df=df.drop(columns=drop_c)
                st.success("Dropped"); st.rerun()
    with s4:
        c1,c2=st.columns(2)
        cast_col=c1.selectbox("Column",df.columns.tolist(),key="cast_col")
        cast_type=c2.selectbox("New type",["int64","float64","str","datetime","category"])
        if st.button("Cast type", type="primary"):
            save_snapshot(f"Cast {cast_col}"); df2=st.session_state.df.copy()
            try:
                if cast_type=="datetime":  df2[cast_col]=pd.to_datetime(df2[cast_col],errors="coerce")
                elif cast_type=="int64":   df2[cast_col]=pd.to_numeric(df2[cast_col],errors="coerce").astype("Int64")
                elif cast_type=="float64": df2[cast_col]=pd.to_numeric(df2[cast_col],errors="coerce")
                elif cast_type=="str":     df2[cast_col]=df2[cast_col].astype(str)
                elif cast_type=="category":df2[cast_col]=df2[cast_col].astype("category")
                st.session_state.df=df2; st.success("Cast applied"); st.rerun()
            except Exception as e: st.error(f"Cast error: {e}")
    if st.session_state.history:
        st.markdown("---"); st.markdown("### Undo History")
        for i,(label,_) in enumerate(reversed(st.session_state.history)):
            ri=len(st.session_state.history)-1-i
            c1,c2=st.columns([4,1])
            c1.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#384a68'>{label}</span>",
                        unsafe_allow_html=True)
            if c2.button("Restore",key=f"undo_{ri}"):
                st.session_state.df=st.session_state.history[ri][1].copy()
                st.success(f"Restored: {label}"); st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 9 — ALERTS
# ══════════════════════════════════════════════════════════════════
with TABS[9]:
    st.markdown("### Smart Alerts & Anomaly Detection")
    if st.button("Scan Dataset", type="primary"):
        alerts=[]
        for col in nc:
            data=df[col].dropna()
            if len(data)<4: continue
            Q1,Q3=data.quantile(.25),data.quantile(.75); IQR=Q3-Q1
            if IQR==0: continue
            out=data[(data<Q1-1.5*IQR)|(data>Q3+1.5*IQR)]
            if len(out):
                pct=len(out)/len(data)*100
                alerts.append({"Severity":"Critical" if pct>10 else "Warning","Column":col,
                               "Issue":"Outliers (IQR)","Detail":f"{len(out)} values ({pct:.1f}%) outside IQR fence"})
            sk=data.skew()
            if abs(sk)>2:
                alerts.append({"Severity":"Warning","Column":col,"Issue":"High skewness",
                               "Detail":f"skew={sk:.2f} — consider log transform"})
        for col in df.columns:
            pct=df[col].isna().mean()*100
            if pct>30: alerts.append({"Severity":"Critical","Column":col,"Issue":"High missing %","Detail":f"{pct:.1f}%"})
            elif pct>5: alerts.append({"Severity":"Warning","Column":col,"Issue":"Missing values","Detail":f"{pct:.1f}%"})
        n_dup=int(df.duplicated().sum())
        if n_dup: alerts.append({"Severity":"Warning","Column":"Dataset","Issue":"Duplicates","Detail":f"{n_dup} rows"})
        sc=Counter(a["Severity"] for a in alerts)
        c1,c2,c3=st.columns(3)
        c1.metric("Critical",sc.get("Critical",0)); c2.metric("Warnings",sc.get("Warning",0)); c3.metric("Total",len(alerts))
        if alerts: st.dataframe(pd.DataFrame(alerts),use_container_width=True,hide_index=True)
        else: st.success("Dataset looks clean! ✅")
    else:
        st.info("Press **Scan Dataset** to check for outliers, skewness, and missing values.")


# ══════════════════════════════════════════════════════════════════
# TAB 10 — MAPS
# ══════════════════════════════════════════════════════════════════
with TABS[10]:
    st.markdown("### Maps & Geospatial")
    cols_all=df.columns.tolist()
    lat_c=next((c for c in cols_all if c.lower() in ("lat","latitude")),None)
    lon_c=next((c for c in cols_all if c.lower() in ("lon","lng","longitude")),None)
    iso_c=next((c for c in cols_all if c.lower() in ("iso","iso_alpha","iso3","country_code")),None)
    map_type=st.selectbox("Map type",["Scatter map","Bubble map","Choropleth (world)"])
    if map_type in ("Scatter map","Bubble map"):
        c1,c2,c3,c4=st.columns(4)
        lat=c1.selectbox("Latitude",cols_all,index=cols_all.index(lat_c) if lat_c else 0)
        lon=c2.selectbox("Longitude",cols_all,index=cols_all.index(lon_c) if lon_c else 0)
        hover=c3.selectbox("Hover label",["(none)"]+cols_all)
        size=(c4.selectbox("Bubble size",["(none)"]+nc) if map_type=="Bubble map" else "(none)")
        style=st.selectbox("Map style",["open-street-map","carto-positron","carto-darkmatter"])
        if st.button("Draw Map", type="primary"):
            try:
                sub=df.dropna(subset=[lat,lon])
                kw=dict(lat=lat,lon=lon,zoom=1,height=500,mapbox_style=style,
                        hover_name=None if hover=="(none)" else hover)
                if map_type=="Bubble map" and size!="(none)": kw["size"]=size; kw["size_max"]=40
                fig=px.scatter_mapbox(sub,**kw)
                fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
                st.plotly_chart(fig,use_container_width=True)
                pin_button("map_pin",f"Map ({map_type})",fig)
            except Exception as e: st.error(f"Map error: {e}")
    elif map_type=="Choropleth (world)":
        str_cols=[c for c in cols_all if c not in nc]
        if not str_cols: st.warning("No string columns for ISO codes.")
        elif not nc: st.warning("No numeric columns.")
        else:
            c1,c2=st.columns(2)
            iso=c1.selectbox("ISO-3 col",str_cols,index=str_cols.index(iso_c) if iso_c and iso_c in str_cols else 0)
            val=c2.selectbox("Value col",nc); cscale=st.selectbox("Colour scale",["Viridis","Blues","Reds","Greens","Plasma"])
            if st.button("Draw Choropleth", type="primary"):
                try:
                    sub=df.dropna(subset=[iso,val])
                    fig=px.choropleth(sub,locations=iso,color=val,color_continuous_scale=cscale,
                                      height=500,title=f"{val} by Country")
                    fig.update_geos(showcoastlines=True,showland=True,showocean=True,
                                    landcolor="#1a2535",oceancolor="#080c17")
                    fig.update_layout(margin=dict(r=0,t=40,l=0,b=0),
                                      paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8899bb"))
                    st.plotly_chart(fig,use_container_width=True)
                    pin_button("choro_pin",f"Choropleth: {val}",fig)
                except Exception as e: st.error(f"Choropleth error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 11 — EXPORT
# ══════════════════════════════════════════════════════════════════
with TABS[11]:
    st.markdown("### Export Your Data")
    e1,e2,e3=st.columns(3)
    with e1:
        st.markdown("**CSV**")
        buf=io.StringIO(); df.to_csv(buf,index=False)
        st.download_button("⬇ Download CSV", buf.getvalue().encode(),
                           f"data_{datetime.now().strftime('%Y%m%d')}.csv","text/csv",use_container_width=True)
    with e2:
        st.markdown("**Excel (multi-sheet)**")
        buf_xl=io.BytesIO()
        try:
            with pd.ExcelWriter(buf_xl,engine="openpyxl") as w:
                df.to_excel(w,sheet_name="Data",index=False)
                if nc: df[nc].describe().round(3).to_excel(w,sheet_name="Statistics")
                pd.DataFrame({"Column":df.columns,"Kind":[col_kind(df[c]) for c in df.columns],
                               "Missing %":(df.isna().mean()*100).round(1).values,
                               "Unique":df.nunique().values}).to_excel(w,sheet_name="Quality",index=False)
            st.download_button("⬇ Download Excel",buf_xl.getvalue(),
                               f"report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        except Exception as e: st.error(f"Excel error: {e}")
    with e3:
        st.markdown("**HTML Report**")
        include_dash = False
        if st.session_state.dashboard:
            include_dash = st.checkbox("Embed dashboard charts", value=True)
        stats_h = df[nc].describe().round(2).to_html(border=0) if nc else ""
        dash_html=""
        if include_dash and st.session_state.dashboard:
            blocks=[]
            for item in st.session_state.dashboard:
                try:
                    fig_o=pio.from_json(json.dumps(item["fig_json"]))
                    blocks.append(f"<div class='cb'><h3>{item['title']}</h3>"
                                  f"{fig_o.to_html(full_html=False,include_plotlyjs=False)}</div>")
                except Exception: pass
            if blocks:
                dash_html=("<h2>Dashboard Charts</h2>"
                           "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
                           +"".join(blocks))
        html_out=f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>DataLens Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;
     background:#0b0f1a;color:#c8d0e0;font-size:13px;padding:0 20px}}
h1{{color:#f0b429;border-bottom:2px solid #1e2740;padding-bottom:8px}}
h2{{color:#8899bb;margin-top:28px;font-size:14px;text-transform:uppercase;letter-spacing:.06em}}
h3{{color:#f0b429;margin-top:16px;font-size:13px}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th{{background:#111827;color:#f0b429;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
td{{padding:6px 12px;border-bottom:1px solid #1a2238;font-size:12px;font-family:monospace}}
tr:nth-child(even) td{{background:#0f1524}}
.cb{{margin-bottom:28px;border:1px solid #1e2740;border-radius:10px;padding:16px}}
</style></head><body>
<h1>◈ DataLens Report</h1>
<p style="font-family:monospace;font-size:11px;color:#384a68">
{datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(df):,} rows × {len(df.columns)} cols · Quality: {quality_score(df)}/100
</p>{dash_html}<h2>Statistics</h2>{stats_h}
<h2>Data Preview — first 50 rows</h2>{df.head(50).to_html(index=False,border=0)}
</body></html>"""
        st.download_button("⬇ Download HTML", html_out.encode(),
                           f"report_{datetime.now().strftime('%Y%m%d')}.html","text/html",use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 12 — DASHBOARD  📊
# ══════════════════════════════════════════════════════════════════
with TABS[12]:
    st.markdown("### My Dashboard")
    st.caption(
        "Charts pinned with **📌 Save to Dashboard** from any tab appear here.  "
        "Each chart is downloadable as a standalone HTML file."
    )

    if not st.session_state.dashboard:
        st.info(
            "Your dashboard is empty.\n\n"
            "Go to any chart tab, generate a chart, then click **📌 Save to Dashboard** below it."
        )
    else:
        hc,_ = st.columns([2,5])
        with hc:
            if st.button("🗑 Clear all", key="dash_clear"):
                st.session_state.dashboard=[]; st.rerun()

        st.markdown(
            f'<span class="badge badge-amber">📌 {len(st.session_state.dashboard)} '
            f'chart{"s" if len(st.session_state.dashboard)!=1 else ""} pinned</span>',
            unsafe_allow_html=True)
        st.markdown("---")

        to_remove: list[int]=[]
        items=st.session_state.dashboard

        for idx in range(0,len(items),2):
            cols=st.columns(2)
            for offset,col in enumerate(cols):
                ci=idx+offset
                if ci>=len(items): break
                item=items[ci]
                with col:
                    st.markdown(
                        f'<div class="dash-card-title">📌 {item["title"]}</div>',
                        unsafe_allow_html=True)
                    try:
                        fig=pio.from_json(json.dumps(item["fig_json"]))
                        apply_bg(fig)
                        st.plotly_chart(fig,use_container_width=True,key=f"dash_chart_{ci}")
                    except Exception as e:
                        st.warning(f"Cannot render: {e}")
                    try:
                        fig_o=pio.from_json(json.dumps(item["fig_json"]))
                        chart_html=fig_o.to_html(full_html=True,include_plotlyjs=True)
                        st.download_button(
                            "⬇ Download chart",chart_html.encode(),
                            file_name=f"{item['title'].replace(' ','_')}.html",
                            mime="text/html",key=f"dl_{ci}",use_container_width=True)
                    except Exception: pass
                    if st.button("✖ Remove",key=f"dash_rm_{ci}"):
                        to_remove.append(ci)
            st.markdown("---")

        for ci in sorted(to_remove,reverse=True):
            st.session_state.dashboard.pop(ci)
        if to_remove: st.rerun()