"""
Smart Data Visualizer — FastAPI Backend  v3.0
---------------------------------------------
Install:
    pip install fastapi uvicorn pandas numpy plotly openpyxl scikit-learn python-multipart

Run:
    uvicorn api_server:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import io
import re
import json
from datetime import datetime
from typing import Optional, Literal, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except ImportError:
    SKLEARN = False

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Smart Data Visualizer API",
    description="REST API for data analysis, visualization, cleaning, and export.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict] = {}

PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold":   px.colors.qualitative.Bold,
    "Safe":   px.colors.qualitative.Safe,
    "Dark24": px.colors.qualitative.Dark24,
}


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────
class SessionResponse(BaseModel):
    session_id: str
    rows: int
    columns: int
    quality: float
    column_names: list[str]


class ChartRequest(BaseModel):
    chart_type: Literal["Bar","Line","Scatter","Histogram","Box","Pie","Area","Heatmap"]
    x_col: str
    y_col: Optional[str] = None
    color_by: Optional[str] = None
    aggregation: Literal["sum","mean","count","min","max"] = "sum"
    palette: str = "Plotly"
    dark_mode: bool = False


class NLQRequest(BaseModel):
    query: str
    palette: str = "Plotly"
    dark_mode: bool = False


class PivotRequest(BaseModel):
    rows: list[str]
    columns: list[str] = []
    values: list[str]
    aggregation: Literal["sum","mean","count","min","max","median"] = "sum"


class ClusterRequest(BaseModel):
    features: list[str]
    k: int = Field(3, ge=2, le=8)
    plot_x: str
    plot_y: str
    standardize: bool = True
    palette: str = "Plotly"
    dark_mode: bool = False


class TrendlineRequest(BaseModel):
    x_col: str
    y_col: str
    trend_type: Literal["Linear","Polynomial","Exponential","All"] = "Linear"
    poly_degree: int = Field(3, ge=2, le=5)
    palette: str = "Plotly"
    dark_mode: bool = False


class CalcColumnRequest(BaseModel):
    column_name: str
    formula: str


class FillMissingRequest(BaseModel):
    column: str
    strategy: Literal["Mean","Median","Mode","Forward fill","Drop rows","Custom"]
    custom_value: Optional[str] = None


class RenameRequest(BaseModel):
    old_name: str
    new_name: str


class CastTypeRequest(BaseModel):
    column: str
    new_type: Literal["int64","float64","str","datetime","category"]


class MapRequest(BaseModel):
    map_type: Literal["Scatter map","Bubble map","Choropleth (world)"]
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None
    iso_col: Optional[str] = None
    value_col: Optional[str] = None
    hover_col: Optional[str] = None
    size_col: Optional[str] = None
    map_style: str = "open-street-map"
    color_scale: str = "Viridis"


class DashboardChart(BaseModel):
    title: str
    fig_json: dict


class HtmlExportRequest(BaseModel):
    dashboard_charts: list[DashboardChart] = []


class DropColumnsRequest(BaseModel):
    columns: list[str]


# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────
SAMPLE_DATA = {
    "Sales": lambda: pd.DataFrame({
        "Month":     ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Sales":     [12000,15000,13500,18000,21000,19500,22000,24000,20000,17000,25000,28000],
        "Profit":    [2400,3200,2700,4000,5000,4500,5200,6100,4800,3900,6500,7800],
        "Customers": [120,145,138,167,189,175,195,212,181,160,230,260],
        "Region":    ["North","South"]*6,
        "Category":  ["A","B","A","C","B","A","C","B","A","C","B","A"],
    }),
    "Students": lambda: pd.DataFrame({
        "Student":   [f"S{i:02d}" for i in range(1,31)],
        "Math":      list(np.random.RandomState(42).randint(60,100,30)),
        "Science":   list(np.random.RandomState(43).randint(55,100,30)),
        "English":   list(np.random.RandomState(44).randint(65,100,30)),
        "Study_Hrs": list(np.random.RandomState(45).uniform(1,8,30).round(1)),
        "Grade":     list(np.random.RandomState(46).choice(["A","B","C","D"],30,p=[.3,.4,.2,.1])),
        "School":    list(np.random.RandomState(47).choice(["Public","Private"],30)),
    }),
    "E-Commerce": lambda: pd.DataFrame({
        "Date":    [str(d.date()) for d in pd.date_range("2024-01-01",periods=90,freq="D")],
        "Revenue": list(np.cumsum(np.random.RandomState(42).normal(1000,200,90))+50000),
        "Orders":  list(np.random.RandomState(43).randint(50,300,90)),
        "AOV":     list(np.random.RandomState(44).uniform(30,120,90).round(2)),
        "Returns": list(np.random.RandomState(45).randint(0,30,90)),
        "Channel": list(np.random.RandomState(46).choice(["Organic","Paid","Email","Social"],90)),
    }),
    "World Cities": lambda: pd.DataFrame({
        "City":       ["New York","London","Tokyo","Paris","Sydney","Dubai","Singapore","Toronto","Berlin","Mumbai"],
        "Country":    ["USA","UK","Japan","France","Australia","UAE","Singapore","Canada","Germany","India"],
        "ISO":        ["USA","GBR","JPN","FRA","AUS","ARE","SGP","CAN","DEU","IND"],
        "Lat":        [40.71,51.51,35.68,48.85,-33.87,25.20,1.35,43.65,52.52,19.08],
        "Lon":        [-74.00,-0.13,139.69,2.35,151.21,55.27,103.82,-79.38,13.40,72.88],
        "Population": [8.3,9.0,13.9,2.1,5.3,3.3,5.9,2.9,3.7,20.7],
        "GDP_Index":  [100,82,91,78,75,95,88,80,76,45],
        "Region":     ["Americas","Europe","Asia","Europe","Oceania","Middle East","Asia","Americas","Europe","Asia"],
    }),
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return _sessions[session_id]


def _get_df(session_id: str) -> pd.DataFrame:
    return _get_session(session_id)["df"]


def _save_snap(session: dict, label: str) -> None:
    session.setdefault("history", [])
    session["history"] = session["history"][-9:]
    session["history"].append((label, session["df"].copy()))


def _quality(df: pd.DataFrame) -> float:
    s = 100 - (df.isna().mean().mean() * 50) - (df.duplicated().mean() * 30)
    return round(min(100.0, max(0.0, s)), 1)


def _num_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() < 20]


def _pal(name: str) -> list[str]:
    return PALETTES.get(name, px.colors.qualitative.Plotly)


def _tmpl(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"


def _fig_json(fig: Any) -> dict:
    return json.loads(fig.to_json())


def _session_meta(sid: str, df: pd.DataFrame) -> dict:
    return {
        "session_id":   sid,
        "rows":         len(df),
        "columns":      len(df.columns),
        "quality":      _quality(df),
        "column_names": df.columns.tolist(),
    }


# ─────────────────────────────────────────────
# ROUTES — INFO / DATA LOADING
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {"message": "Smart Data Visualizer API", "docs": "/docs", "version": "3.0.0"}


@app.get("/samples", tags=["Data Loading"])
def list_samples():
    return {"datasets": list(SAMPLE_DATA.keys())}


@app.post("/sessions/sample", tags=["Data Loading"], response_model=SessionResponse)
def load_sample(
    dataset: str = Query(...),
    session_id: str = Query("default"),
):
    if dataset not in SAMPLE_DATA:
        raise HTTPException(400, f"Unknown dataset. Choose from: {list(SAMPLE_DATA.keys())}")
    df = SAMPLE_DATA[dataset]()
    _sessions[session_id] = {"df": df, "history": []}
    return _session_meta(session_id, df)


@app.post("/sessions/upload", tags=["Data Loading"], response_model=SessionResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Query("default"),
):
    contents = await file.read()
    fname = file.filename or ""
    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(400, "Only CSV and Excel files are supported.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {e}")
    _sessions[session_id] = {"df": df, "history": []}
    return _session_meta(session_id, df)


@app.post("/sessions/url", tags=["Data Loading"], response_model=SessionResponse)
def load_from_url(url: str = Query(...), session_id: str = Query("default")):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise HTTPException(422, f"Failed to fetch URL: {e}")
    _sessions[session_id] = {"df": df, "history": []}
    return _session_meta(session_id, df)


@app.delete("/sessions/{session_id}", tags=["Data Loading"])
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


# ─────────────────────────────────────────────
# ROUTES — OVERVIEW
# ─────────────────────────────────────────────
@app.get("/sessions/{session_id}/overview", tags=["Overview"])
def overview(session_id: str, rows: int = Query(10, ge=1, le=500)):
    df = _get_df(session_id)
    nc = _num_cols(df)
    col_info = [
        {
            "column":      c,
            "type":        str(df[c].dtype),
            "non_null":    int(df[c].count()),
            "missing_pct": round(df[c].isna().mean() * 100, 1),
            "unique":      int(df[c].nunique()),
        }
        for c in df.columns
    ]
    numeric_stats   = df[nc].describe().round(2).to_dict() if nc else {}
    missing_by_col  = {c: round(df[c].isna().mean()*100, 1) for c in df.columns if df[c].isna().any()}
    preview = df.head(rows).fillna("").astype(str).to_dict(orient="records")
    return {
        **_session_meta(session_id, df),
        "missing_total":     int(df.isna().sum().sum()),
        "duplicates":        int(df.duplicated().sum()),
        "column_info":       col_info,
        "numeric_stats":     numeric_stats,
        "missing_by_column": missing_by_col,
        "preview":           preview,
    }


# ─────────────────────────────────────────────
# ROUTES — CHARTS
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/charts", tags=["Charts"])
def generate_chart(session_id: str, req: ChartRequest):
    df  = _get_df(session_id)
    nc  = _num_cols(df)
    p   = _pal(req.palette)
    t   = _tmpl(req.dark_mode)
    x, y, col = req.x_col, req.y_col, req.color_by

    if x not in df.columns:
        raise HTTPException(400, f"Column '{x}' not found.")
    if y and y not in df.columns:
        raise HTTPException(400, f"Column '{y}' not found.")
    if col and col not in df.columns:
        raise HTTPException(400, f"Column '{col}' not found.")

    fig = None
    try:
        if req.chart_type == "Bar":
            if y:
                grp_cols = [x] + ([col] if col else [])
                gdf = df.groupby(grp_cols, observed=True)[y].agg(req.aggregation).reset_index()
                fig = px.bar(gdf, x=x, y=y, color=col or None,
                             barmode="group", color_discrete_sequence=p, template=t)
            else:
                vc = df[x].value_counts().reset_index()
                vc.columns = [x, "count"]
                fig = px.bar(vc, x=x, y="count", color_discrete_sequence=p, template=t)

        elif req.chart_type == "Line":
            if not y:
                raise HTTPException(400, "Y column required for Line chart.")
            fig = px.line(df.sort_values(x), x=x, y=y, color=col,
                          markers=True, color_discrete_sequence=p, template=t)

        elif req.chart_type == "Scatter":
            if not y:
                raise HTTPException(400, "Y column required for Scatter chart.")
            fig = px.scatter(df, x=x, y=y, color=col,
                             trendline="ols" if not col else None,
                             color_discrete_sequence=p, template=t)

        elif req.chart_type == "Histogram":
            fig = px.histogram(df, x=x, color=col, nbins=30,
                               color_discrete_sequence=p, template=t)

        elif req.chart_type == "Box":
            y_use = y if y else x
            fig = px.box(df, x=col or None, y=y_use, color=col or None,
                         color_discrete_sequence=p, template=t)

        elif req.chart_type == "Pie":
            vc = df[x].value_counts().head(10)
            fig = px.pie(names=vc.index, values=vc.values, hole=0.35,
                         color_discrete_sequence=p)

        elif req.chart_type == "Area":
            if not y:
                raise HTTPException(400, "Y column required for Area chart.")
            fig = px.area(df.sort_values(x), x=x, y=y, color=col,
                          color_discrete_sequence=p, template=t)

        elif req.chart_type == "Heatmap":
            if not nc:
                raise HTTPException(400, "No numeric columns for Heatmap.")
            corr = df[nc].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, template=t)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Chart generation error: {e}")

    if fig is None:
        raise HTTPException(422, "Could not generate chart with given parameters.")

    fig.update_layout(height=460)
    return {"figure": _fig_json(fig), "chart_type": req.chart_type}


# ─────────────────────────────────────────────
# ROUTES — NLQ
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/ask", tags=["Ask Data"])
def ask_data(session_id: str, req: NLQRequest):
    df = _get_df(session_id)
    nc = _num_cols(df)
    cc = _cat_cols(df)
    q  = req.query.lower()
    p  = _pal(req.palette)
    t  = _tmpl(req.dark_mode)

    agg_fn = "sum"
    if any(w in q for w in ["average","avg","mean"]): agg_fn = "mean"
    elif "count" in q:                                 agg_fn = "count"
    elif "max"   in q:                                 agg_fn = "max"
    elif "min"   in q:                                 agg_fn = "min"

    chart_t = "bar"
    if any(w in q for w in ["trend","line","over","time"]): chart_t = "line"
    elif any(w in q for w in ["scatter","vs","versus"]):    chart_t = "scatter"
    elif any(w in q for w in ["pie","share","proportion"]): chart_t = "pie"
    elif any(w in q for w in ["distribution","histogram"]): chart_t = "histogram"

    top_m = re.search(r"top\s+(\d+)", q)
    top_n = int(top_m.group(1)) if top_m else None

    found = [c for c in sorted(df.columns, key=len, reverse=True) if c.lower() in q]
    y_c   = next((c for c in found if c in nc), None) or (nc[0] if nc else None)
    x_c   = next((c for c in found if c != y_c), None) or (cc[0] if cc else None)

    fig = None
    result_data = None
    try:
        if chart_t in ("bar","line","pie") and x_c and y_c:
            grp = df.groupby(x_c, observed=True)[y_c].agg(agg_fn).reset_index()
            grp = grp.sort_values(y_c, ascending=False)
            if top_n:
                grp = grp.head(top_n)
            result_data = grp.to_dict(orient="records")
            if chart_t == "bar":
                fig = px.bar(grp, x=x_c, y=y_c, color=x_cs,
                             color_discrete_sequence=p, template=t,
                             title=f"{agg_fn.title()} of {y_c} by {x_c}")
            elif chart_t == "line":
                fig = px.line(grp, x=x_c, y=y_c, markers=True,
                              color_discrete_sequence=p, template=t)
            elif chart_t == "pie":
                fig = px.pie(grp, names=x_c, values=y_c, hole=0.35,
                             color_discrete_sequence=p)
        elif chart_t == "scatter" and x_c and y_c:
            fig = px.scatter(df, x=x_c, y=y_c, trendline="ols",
                             color_discrete_sequence=p, template=t)
            result_data = df[[x_c, y_c]].describe().round(2).to_dict()
        elif chart_t == "histogram" and x_c:
            fig = px.histogram(df, x=x_c, nbins=30,
                               color_discrete_sequence=p, template=t)
        elif agg_fn == "count" and x_c:
            vc = df[x_c].value_counts().reset_index()
            vc.columns = [x_c, "count"]
            if top_n:
                vc = vc.head(top_n)
            fig = px.bar(vc, x=x_c, y="count", color=x_c,
                         color_discrete_sequence=p, template=t)
            result_data = vc.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(422, f"NLQ error: {e}")

    if not fig:
        raise HTTPException(422, "Could not resolve query — try mentioning a column name.")

    fig.update_layout(height=420, showlegend=False)
    return {
        "query":    req.query,
        "resolved": {"x": x_c, "y": y_c, "aggregation": agg_fn, "chart": chart_t, "top_n": top_n},
        "figure":   _fig_json(fig),
        "data":     result_data,
    }


# ─────────────────────────────────────────────
# ROUTES — PIVOT
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/pivot", tags=["Pivot"])
def pivot_table(session_id: str, req: PivotRequest):
    df      = _get_df(session_id)
    missing = [c for c in req.rows + req.columns + req.values if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    try:
        col_arg = req.columns[0] if len(req.columns) == 1 else (req.columns or None)
        val_arg = req.values[0]  if len(req.values)  == 1 else req.values
        pivot   = pd.pivot_table(
            df, index=req.rows, columns=col_arg, values=val_arg,
            aggfunc=req.aggregation, margins=True, margins_name="Total",
            observed=True,
        ).round(2)
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ["_".join(str(s) for s in c).strip("_") for c in pivot.columns]
        else:
            pivot.columns = [str(c) for c in pivot.columns]
        return {"pivot": pivot.reset_index().fillna("").to_dict(orient="records"),
                "index_names": req.rows}
    except Exception as e:
        raise HTTPException(422, f"Pivot error: {e}")


# ─────────────────────────────────────────────
# ROUTES — CLUSTERING
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/cluster", tags=["Clustering"])
def cluster(session_id: str, req: ClusterRequest):
    if not SKLEARN:
        raise HTTPException(501, "scikit-learn not installed.")
    df = _get_df(session_id)
    all_needed = list(dict.fromkeys(req.features + [req.plot_x, req.plot_y]))
    missing    = [c for c in all_needed if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    if len(req.features) < 2:
        raise HTTPException(400, "Need at least 2 feature columns.")

    p   = _pal(req.palette)
    t   = _tmpl(req.dark_mode)
    sub = df[req.features].dropna()
    X   = StandardScaler().fit_transform(sub) if req.standardize else sub.values

    try:
        labels = KMeans(n_clusters=req.k, random_state=42, n_init="auto").fit_predict(X)
    except Exception as e:
        raise HTTPException(422, f"Clustering error: {e}")

    out            = df.loc[sub.index].copy()
    out["Cluster"] = [f"C{lbl+1}" for lbl in labels]

    fig_scatter = px.scatter(out, x=req.plot_x, y=req.plot_y, color="Cluster",
                             color_discrete_sequence=p, template=t,
                             title=f"{req.plot_x} vs {req.plot_y} — clusters")
    sizes         = out["Cluster"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig_sizes     = px.bar(sizes, x="Cluster", y="Count", color="Cluster",
                           color_discrete_sequence=p, template=t, title="Cluster sizes")
    fig_sizes.update_layout(showlegend=False)

    cluster_means = (out.groupby("Cluster", observed=True)[req.features]
                       .mean().round(2).reset_index()
                       .to_dict(orient="records"))

    return {
        "scatter_figure": _fig_json(fig_scatter),
        "sizes_figure":   _fig_json(fig_sizes),
        "cluster_means":  cluster_means,
        "labeled_rows":   out.fillna("").astype(str).to_dict(orient="records"),
    }


# ─────────────────────────────────────────────
# ROUTES — TRENDLINES
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/trendline", tags=["Trendlines"])
def trendline(session_id: str, req: TrendlineRequest):
    df = _get_df(session_id)
    for c in [req.x_col, req.y_col]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.x_col, req.y_col]].dropna()
    if len(sub) < 4:
        raise HTTPException(422, "Need at least 4 non-null rows for trendline fitting.")

    x  = sub[req.x_col].values.astype(float)
    y  = sub[req.y_col].values.astype(float)
    xs = np.linspace(x.min(), x.max(), 300)
    p  = _pal(req.palette)
    t  = _tmpl(req.dark_mode)

    denom: float = max(float(np.sum((y - y.mean()) ** 2)), 1e-10)
    r2_values: dict[str, float] = {}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="markers", name="Data",
                             marker=dict(color=p[0], size=5, opacity=0.6)))

    def _add_line(xs_: Any, ys_: Any, name: str, color: str, dash: str = "solid") -> None:
        fig.add_trace(go.Scatter(x=list(xs_), y=list(ys_), mode="lines", name=name,
                                 line=dict(color=color, width=2, dash=dash)))

    if req.trend_type in ("Linear","All"):
        c_ = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - np.polyval(c_, x))**2) / denom
        r2_values["linear"] = round(float(r2), 4)
        _add_line(xs, np.polyval(c_, xs), f"Linear  R²={r2:.3f}",
                  p[1] if len(p)>1 else "#EF4444")

    if req.trend_type in ("Polynomial","All"):
        d  = req.poly_degree if req.trend_type == "Polynomial" else 3
        c_ = np.polyfit(x, y, d)
        r2 = 1 - np.sum((y - np.polyval(c_, x))**2) / denom
        r2_values[f"polynomial_deg{d}"] = round(float(r2), 4)
        _add_line(xs, np.polyval(c_, xs), f"Poly deg-{d}  R²={r2:.3f}",
                  p[2] if len(p)>2 else "#F59E0B", "dash")

    if req.trend_type in ("Exponential","All") and (y > 0).all():
        try:
            lc   = np.polyfit(x, np.log(y), 1)
            a, b = float(np.exp(lc[1])), float(lc[0])
            yp   = a * np.exp(b * x)
            r2   = 1 - np.sum((y - yp)**2) / denom
            r2_values["exponential"] = round(float(r2), 4)
            _add_line(xs, a * np.exp(b * xs), f"Exponential  R²={r2:.3f}",
                      p[3] if len(p)>3 else "#10B981", "dot")
        except Exception:
            pass

    fig.update_layout(height=440, template=t,
                      xaxis_title=req.x_col, yaxis_title=req.y_col,
                      legend=dict(orientation="h", y=1.05))
    return {"figure": _fig_json(fig), "r2_values": r2_values}


# ─────────────────────────────────────────────
# ROUTES — COLUMNS
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/columns/calc", tags=["Columns"])
def add_calc_column(session_id: str, req: CalcColumnRequest):
    session = _get_session(session_id)
    df      = session["df"]
    if req.column_name in df.columns:
        raise HTTPException(400, f"Column '{req.column_name}' already exists.")
    ns: dict[str, Any] = {}
    for c in df.columns:
        ns[c] = df[c]
        ns[c.replace(" ","_")] = df[c]
    ns.update({"np": np, "pd": pd})
    formula = re.sub(r"`([^`]+)`", lambda m: m.group(1).replace(" ","_"), req.formula)
    try:
        result = eval(formula, {"__builtins__": {}}, ns)  # noqa: S307
    except Exception as e:
        raise HTTPException(422, f"Formula error: {e}")
    _save_snap(session, f"Add col: {req.column_name}")
    session["df"][req.column_name] = result
    return _session_meta(session_id, session["df"])


@app.post("/sessions/{session_id}/columns/drop", tags=["Columns"])
def drop_columns(session_id: str, req: DropColumnsRequest):
    session = _get_session(session_id)
    df      = session["df"]
    missing = [c for c in req.columns if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    _save_snap(session, f"Drop {req.columns}")
    session["df"] = df.drop(columns=req.columns)
    return _session_meta(session_id, session["df"])


@app.patch("/sessions/{session_id}/columns/rename", tags=["Columns"])
def rename_column(session_id: str, req: RenameRequest):
    session = _get_session(session_id)
    df      = session["df"]
    if req.old_name not in df.columns:
        raise HTTPException(400, f"Column '{req.old_name}' not found.")
    if req.new_name in df.columns:
        raise HTTPException(400, f"Column '{req.new_name}' already exists.")
    _save_snap(session, f"Rename {req.old_name} → {req.new_name}")
    session["df"] = df.rename(columns={req.old_name: req.new_name})
    return _session_meta(session_id, session["df"])


@app.patch("/sessions/{session_id}/columns/cast", tags=["Columns"])
def cast_column(session_id: str, req: CastTypeRequest):
    session = _get_session(session_id)
    df      = session["df"].copy()
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    try:
        _save_snap(session, f"Cast {req.column} → {req.new_type}")
        if req.new_type == "datetime":   df[req.column] = pd.to_datetime(df[req.column], errors="coerce")
        elif req.new_type == "int64":    df[req.column] = pd.to_numeric(df[req.column], errors="coerce").astype("Int64")
        elif req.new_type == "float64":  df[req.column] = pd.to_numeric(df[req.column], errors="coerce")
        elif req.new_type == "str":      df[req.column] = df[req.column].astype(str)
        elif req.new_type == "category": df[req.column] = df[req.column].astype("category")
        session["df"] = df
    except Exception as e:
        raise HTTPException(422, str(e))
    return _session_meta(session_id, session["df"])


# ─────────────────────────────────────────────
# ROUTES — CLEANING
# ─────────────────────────────────────────────
@app.delete("/sessions/{session_id}/clean/duplicates", tags=["Cleaning"])
def remove_duplicates(session_id: str):
    session = _get_session(session_id)
    df      = session["df"]
    n       = int(df.duplicated().sum())
    _save_snap(session, "Remove duplicates")
    session["df"] = df.drop_duplicates().reset_index(drop=True)
    return {"removed": n, **_session_meta(session_id, session["df"])}


@app.post("/sessions/{session_id}/clean/missing", tags=["Cleaning"])
def fill_missing(session_id: str, req: FillMissingRequest):
    session = _get_session(session_id)
    df      = session["df"].copy()
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    _save_snap(session, f"Fill {req.column} ({req.strategy})")
    if req.strategy == "Mean":
        df[req.column] = df[req.column].fillna(df[req.column].mean())
    elif req.strategy == "Median":
        df[req.column] = df[req.column].fillna(df[req.column].median())
    elif req.strategy == "Mode":
        mode_vals = df[req.column].mode()
        if not mode_vals.empty:
            df[req.column] = df[req.column].fillna(mode_vals[0])
    elif req.strategy == "Forward fill":
        df[req.column] = df[req.column].ffill()
    elif req.strategy == "Drop rows":
        df = df.dropna(subset=[req.column])
    elif req.strategy == "Custom":
        if not req.custom_value:
            raise HTTPException(400, "custom_value required for Custom strategy.")
        try:    df[req.column] = df[req.column].fillna(float(req.custom_value))
        except ValueError: df[req.column] = df[req.column].fillna(req.custom_value)
    session["df"] = df
    return _session_meta(session_id, session["df"])


# ─────────────────────────────────────────────
# ROUTES — ALERTS
# ─────────────────────────────────────────────
@app.get("/sessions/{session_id}/alerts", tags=["Alerts"])
def scan_alerts(session_id: str):
    df     = _get_df(session_id)
    alerts = []
    for col in _num_cols(df):
        data = df[col].dropna()
        if len(data) < 4:
            continue
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR     = Q3 - Q1
        if IQR == 0:
            continue
        out = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        if len(out):
            pct = len(out) / len(data) * 100
            alerts.append({"severity": "critical" if pct>10 else "warning",
                            "column": col, "issue": "Outliers (IQR)",
                            "detail": f"{len(out)} values ({pct:.1f}%) outside IQR fence"})
        sk = data.skew()
        if abs(sk) > 2:
            alerts.append({"severity": "warning", "column": col, "issue": "High skewness",
                            "detail": f"skew = {sk:.2f} — consider log transform"})
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 30:
            alerts.append({"severity": "critical", "column": col,
                            "issue": "High missing %", "detail": f"{pct:.1f}% missing"})
        elif pct > 5:
            alerts.append({"severity": "warning", "column": col,
                            "issue": "Missing values", "detail": f"{pct:.1f}% missing"})
    n_dup = int(df.duplicated().sum())
    if n_dup:
        alerts.append({"severity": "warning", "column": "Dataset",
                        "issue": "Duplicates", "detail": f"{n_dup} duplicate rows"})
    return {"total": len(alerts),
            "critical": sum(1 for a in alerts if a["severity"]=="critical"),
            "warnings": sum(1 for a in alerts if a["severity"]=="warning"),
            "alerts": alerts}


# ─────────────────────────────────────────────
# ROUTES — MAPS
# ─────────────────────────────────────────────
@app.post("/sessions/{session_id}/maps", tags=["Maps"])
def generate_map(session_id: str, req: MapRequest):
    df = _get_df(session_id)
    try:
        if req.map_type in ("Scatter map","Bubble map"):
            if not req.lat_col or not req.lon_col:
                raise HTTPException(400, "lat_col and lon_col are required.")
            for c in [req.lat_col, req.lon_col]:
                if c not in df.columns:
                    raise HTTPException(400, f"Column '{c}' not found.")
            sub = df.dropna(subset=[req.lat_col, req.lon_col])
            kw: dict[str, Any] = dict(lat=req.lat_col, lon=req.lon_col,
                                      hover_name=req.hover_col or None,
                                      zoom=1, height=500, mapbox_style=req.map_style)
            if req.map_type == "Bubble map" and req.size_col:
                kw["size"] = req.size_col
                kw["size_max"] = 40
            fig = px.scatter_mapbox(sub, **kw)
            fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))

        elif req.map_type == "Choropleth (world)":
            if not req.iso_col or not req.value_col:
                raise HTTPException(400, "iso_col and value_col are required.")
            for c in [req.iso_col, req.value_col]:
                if c not in df.columns:
                    raise HTTPException(400, f"Column '{c}' not found.")
            sub = df.dropna(subset=[req.iso_col, req.value_col])
            fig = px.choropleth(sub, locations=req.iso_col, color=req.value_col,
                                color_continuous_scale=req.color_scale,
                                height=500, title=f"{req.value_col} by country")
            fig.update_geos(showcoastlines=True, showland=True, showocean=True,
                            landcolor="#F5F5F5", oceancolor="#EBF4FF")
            fig.update_layout(margin=dict(r=0,t=40,l=0,b=0))
        else:
            raise HTTPException(400, f"Unknown map_type: {req.map_type}")

        return {"figure": _fig_json(fig), "map_type": req.map_type}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Map error: {e}")


# ─────────────────────────────────────────────
# ROUTES — HISTORY
# ─────────────────────────────────────────────
@app.get("/sessions/{session_id}/history", tags=["History"])
def list_history(session_id: str):
    session = _get_session(session_id)
    return {"snapshots": [{"index": i, "label": lbl}
                          for i, (lbl, _) in enumerate(session.get("history", []))]}


@app.post("/sessions/{session_id}/history/{index}/restore", tags=["History"])
def restore_snapshot(session_id: str, index: int):
    session = _get_session(session_id)
    history = session.get("history", [])
    if index < 0 or index >= len(history):
        raise HTTPException(404, f"Snapshot index {index} not found.")
    session["df"] = history[index][1].copy()
    return {"restored": history[index][0], **_session_meta(session_id, session["df"])}


# ─────────────────────────────────────────────
# ROUTES — EXPORT
# ─────────────────────────────────────────────
@app.get("/sessions/{session_id}/export/csv", tags=["Export"])
def export_csv(session_id: str):
    df  = _get_df(session_id)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"data_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/sessions/{session_id}/export/excel", tags=["Export"])
def export_excel(session_id: str):
    df  = _get_df(session_id)
    nc  = _num_cols(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        if nc:
            df[nc].describe().round(3).to_excel(writer, sheet_name="Statistics")
        pd.DataFrame({"Column": df.columns,
                      "Missing %": (df.isna().mean()*100).round(1).values,
                      "Unique": df.nunique().values}
                     ).to_excel(writer, sheet_name="Quality", index=False)
    buf.seek(0)
    filename = f"report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/sessions/{session_id}/export/html", tags=["Export"])
def export_html(session_id: str, req: HtmlExportRequest = Body(default=HtmlExportRequest())):
    df      = _get_df(session_id)
    nc      = _num_cols(df)
    stats_h = df[nc].describe().round(2).to_html(border=0, classes="tbl") if nc else ""

    dashboard_html = ""
    if req.dashboard_charts:
        blocks = []
        for item in req.dashboard_charts:
            try:
                fig_obj    = pio.from_json(json.dumps(item.fig_json))
                chart_html = fig_obj.to_html(full_html=False, include_plotlyjs=False,
                                             config={"responsive": True})
                blocks.append(f"<div class='cb'><h3>{item.title}</h3>{chart_html}</div>")
            except Exception:
                blocks.append(f"<p>Could not render: {item.title}</p>")
        dashboard_html = ("<h2>Dashboard Charts</h2>"
                          "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
                          + "".join(blocks))

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;
     color:#1a1a2e;font-size:13px;padding:0 20px;background:#f8f9ff}}
h1{{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:8px;font-size:22px}}
h2{{color:#16213e;margin-top:32px;font-size:16px}}
h3{{color:#0f3460;margin-top:16px;font-size:14px}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th{{background:#0f3460;color:#fff;padding:8px 12px;text-align:left;font-size:12px}}
td{{padding:6px 12px;border-bottom:1px solid #e2e8f0;font-size:12px}}
tr:nth-child(even) td{{background:#f1f5f9}}
.cb{{margin-bottom:28px;border:1px solid #e2e8f0;border-radius:10px;padding:16px;background:#fff}}
</style></head><body>
<h1>Data Report</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
{len(df):,} rows &times; {len(df.columns)} cols &nbsp;·&nbsp; Quality {_quality(df)}/100</p>
{dashboard_html}
<h2>Statistics</h2>{stats_h}
<h2>Data Preview — first 50 rows</h2>
{df.head(50).to_html(index=False, border=0, classes="tbl")}
</body></html>"""

    filename = f"report_{datetime.now().strftime('%Y%m%d')}.html"
    return StreamingResponse(iter([html]), media_type="text/html",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})