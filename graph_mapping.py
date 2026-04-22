# Graph rules for univariate analysis
GRAPH_RULES = {
    "categorical": [
        "Bar Chart",
        "Horizontal Bar Chart",
        "Pie Chart",
        "Treemap"
    ],
    "numerical": [
        "Histogram",
        "Box Plot",
        "Violin Plot",
        "Density Plot"
    ],
    "datetime": [
        "Line Chart",
        "Area Chart",
        "Bar Chart (by period)"
    ],
    "boolean": [
        "Bar Chart",
        "Pie Chart"
    ],
    "text": [
        "Bar Chart (Top N)",
        "Word Cloud"
    ],
    "unknown": []
}

# Graph rules for bivariate analysis (x_type, y_type)
BIVARIATE_GRAPH_RULES = {
    "categorical|categorical": [
        "Grouped Bar Chart",
        "Stacked Bar Chart",
        "Heatmap (Counts)"
    ],
    "categorical|numerical": [
        "Box Plot",
        "Violin Plot",
        "Strip Plot",
        "Bar Chart (Aggregated)"
    ],
    "numerical|categorical": [
        "Box Plot",
        "Violin Plot",
        "Strip Plot",
        "Bar Chart (Aggregated)"
    ],
    "numerical|numerical": [
        "Scatter Plot",
        "Line Plot",
        "Hexbin Plot",
        "2D Density Plot"
    ],
    "datetime|numerical": [
        "Line Chart",
        "Area Chart"
    ]
}

# Graph rules for multivariate analysis
MULTIVARIATE_GRAPH_RULES = [
    "Pair Plot",
    "Correlation Heatmap",
    "Parallel Coordinates Plot",
    "3D Scatter Plot"
]


def get_col_type(series):
    """Classify a pandas Series into one of the graph_mapping types."""
    import pandas as pd
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if series.dtype == object:
        # rough heuristic: average word count > 3 → text
        avg_words = series.dropna().astype(str).str.split().str.len().mean()
        if avg_words and avg_words > 3:
            return "text"
        return "categorical"
    return "unknown"


def get_univariate_charts(series):
    """Return allowed chart types for a single column."""
    col_type = get_col_type(series)
    return GRAPH_RULES.get(col_type, [])


def get_bivariate_charts(x_series, y_series):
    """Return allowed chart types for an (x, y) pair."""
    xt = get_col_type(x_series)
    yt = get_col_type(y_series)
    key = f"{xt}|{yt}"
    return BIVARIATE_GRAPH_RULES.get(key, [])