# 📊 Smart Data Visualizer

> An advanced, AI-augmented data analytics platform built with Streamlit and Plotly. Upload any CSV and instantly explore it through 25+ interactive charts, automated statistical analysis, a data quality scorer, and optional LLM-powered insights .

---

## ✨ Features

### 🏠 Home — Dataset Overview
- Auto-classifies every column as **numerical, categorical, datetime, or text**
- Displays row/column counts, missing data %, and per-column stats in a clean summary table
- Head / tail / random preview with adjustable row count
- One-click **AI-generated dataset summary** (requires Hugging Face token)

### 📊 Visualizations — 25+ Interactive Charts
Charts are organized into three categories and dynamically enabled/disabled based on what column types your data contains:

| Category | Available Charts |
|----------|-----------------|
| **Single Variable** | Bar, Horizontal Bar, Pie, Treemap, Histogram, Box Plot, Violin, Density, Line (time), Area (time), Bar by Period |
| **Two Variables** | Scatter, Line, Hexbin, 2D Density, Grouped/Stacked Bar, Heatmap (counts), Box (grouped), Violin (grouped), Strip Plot, Bar (aggregated), Line (datetime × numeric), Area (datetime × numeric) |
| **Multiple Variables** | Pair Plot, Correlation Heatmap, 3D Scatter, Parallel Coordinates |

Every chart comes with rule-based insights and optional AI-enhanced commentary. Charts can be pinned to a **personal dashboard** for side-by-side comparison.

### 🤖 AI Insights
- Five analysis modes: **Full Analysis, Trends & Patterns, Data Quality, Recommendations, Statistical Summary**
- Powered by Hugging Face Inference API (`google/flan-t5-base` by default, with `flan-t5-large` and `facebook/bart-large-cnn` as fallbacks)
- Gracefully falls back to rule-based analysis when the API is unavailable
- **Analysis history** panel shows the last 5 runs with timestamps

### 📈 Advanced Analytics
Four dedicated sub-tabs:
- **Correlation** — heatmap + pairwise correlation matrix for all numerical columns
- **Trends** — rolling averages and trend lines for time-series data
- **Outliers** — IQR-based outlier detection with highlighted data points
- **Distribution** — skewness, kurtosis, and per-column distribution summaries

### 📋 Data Quality Assessment
- Composite **quality score (0–100)** deducting for missing values, duplicates, and constant columns
- Missing-values bar chart sorted by severity
- Actionable recommendations (impute / drop / flag)
- Optional AI-powered quality narrative 

### ⚙️ Settings
- Toggle auto type-detection, missing value handling, and duplicate removal
- Export processed data as **CSV, Excel, or JSON**

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-data-visualizer.git
cd smart-data-visualizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Hugging Face for AI insights

```bash
# Get a free token at https://huggingface.co/settings/tokens
export HUGGINGFACE_TOKEN="hf_your_token_here"
pip install huggingface-hub
```

> The app works fully without a Hugging Face token — all visualizations, statistics, and rule-based insights are available out of the box. The token only unlocks the LLM-generated narratives on the AI Insights and Data Quality pages.

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📂 Project Structure

```
smart-data-visualizer/
├── app.py              # Full application — all pages, charts, and AI logic
├── graph_mapping.py    # Chart type → Plotly renderer mapping
└── README.md
```

---

## 🧩 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `plotly` | Interactive charting |
| `openpyxl` | Excel export |
| `huggingface-hub` | *(Optional)* LLM-powered insights |

Install everything at once:

```bash
pip install streamlit pandas numpy plotly openpyxl huggingface-hub
```

---

## 🗂️ Sample Datasets

No CSV handy? The app ships with three built-in sample datasets you can load from the sidebar:

| Dataset | Description |
|---------|-------------|
| **Sales Data** | 6-month sales, customer counts, and region breakdown |
| **Student Performance** | Grades across Math, Science, and English for 8 students |
| **Stock Prices** | 30 days of simulated price, volume, high, and low data |



---

## 💡 Upgrade Ideas

- **Semantic column type detection** — use an embedding model to infer column meaning beyond dtype
- **Natural language queries** — "show me sales by region" → auto-generate the right chart
- **Persistent dashboard** — save pinned charts to disk between sessions
- **Database connectors** — load data directly from PostgreSQL, BigQuery, or S3
- **Scheduled reports** — auto-generate and email a PDF summary on a cron schedule
- **Deploy on Streamlit Cloud** — push to GitHub and connect at [share.streamlit.io](https://share.streamlit.io)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss the proposal.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---
