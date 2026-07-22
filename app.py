import math
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules


ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "groceries-groceries.csv",
    ROOT / "groceries - groceries.csv",
]


def format_itemset(items):
    return ", ".join(sorted(items)) if isinstance(items, (set, frozenset)) else str(items)


@st.cache_data(show_spinner=False)
def load_transactions():
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find the groceries dataset in the workspace.")

    raw = pd.read_csv(data_path)
    if "Item(s)" in raw.columns:
        raw = raw.drop(columns=["Item(s)"])

    transactions = []
    for _, row in raw.iterrows():
        cleaned = [str(val).strip() for val in row if pd.notna(val) and str(val).strip()]
        if len(cleaned) > 1:
            transactions.append(cleaned)

    return transactions


@st.cache_data(show_spinner=False)
def build_mining_results(min_support: float, min_confidence: float):
    transactions = load_transactions()

    all_items = sorted({item for basket in transactions for item in basket})
    basket_matrix = []
    for basket in transactions:
        basket_matrix.append([True if item in basket else False for item in all_items])

    item_frame = pd.DataFrame(basket_matrix, columns=all_items, dtype=bool)

    frequent_itemsets = apriori(item_frame, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)

    item_counts = Counter(item for basket in transactions for item in basket)
    item_counts_series = pd.Series(item_counts).sort_values(ascending=False)

    basket_sizes = [len(basket) for basket in transactions]
    basket_size_series = pd.Series(basket_sizes)

    return {
        "transactions": transactions,
        "item_frame": item_frame,
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "item_counts": item_counts_series,
        "basket_size_series": basket_size_series,
    }


def prepare_rules_for_display(rules_df: pd.DataFrame) -> pd.DataFrame:
    display_df = rules_df.copy()
    display_df["antecedents"] = display_df["antecedents"].apply(format_itemset)
    display_df["consequents"] = display_df["consequents"].apply(format_itemset)
    return display_df


def prepare_itemsets_for_display(itemsets_df: pd.DataFrame) -> pd.DataFrame:
    display_df = itemsets_df.copy()
    display_df["itemsets"] = display_df["itemsets"].apply(format_itemset)
    return display_df


def split_items(item_text) -> list[str]:
    return [item.strip() for item in str(item_text).split(",") if item.strip()]


def style_plotly_figure(fig: go.Figure):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title_font=dict(color="#f8fafc"),
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#cbd5e1")
    fig.update_yaxes(showgrid=False, zeroline=False, color="#cbd5e1")
    return fig


def plot_support_confidence_bubble(rules_df: pd.DataFrame):
    if rules_df.empty:
        return None

    chart_df = prepare_rules_for_display(rules_df[["support", "confidence", "lift", "antecedents", "consequents"]]).copy()
    chart_df = chart_df.head(20)
    chart_df["rule_label"] = [f"Rule {i + 1}" for i in range(len(chart_df))]
    chart_df["bubble_size"] = chart_df["lift"] * 20

    fig = px.scatter(
        chart_df,
        x="confidence",
        y="support",
        size="bubble_size",
        color="lift",
        hover_name="rule_label",
        hover_data={"antecedents": True, "consequents": True, "support": True, "confidence": True, "lift": True},
        title="Support vs confidence bubble chart",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig = style_plotly_figure(fig)
    return fig


def plot_rule_network(rules_df: pd.DataFrame):
    if rules_df.empty:
        return None

    display_rules = prepare_rules_for_display(rules_df)
    nodes = []
    edges = []

    for _, row in display_rules.iterrows():
        antecedents = split_items(row["antecedents"])
        consequents = split_items(row["consequents"])
        for antecedent in antecedents:
            nodes.append(antecedent)
            for consequent in consequents:
                nodes.append(consequent)
                edges.append((antecedent, consequent))

    unique_nodes = list(dict.fromkeys(nodes))
    if not unique_nodes:
        return None

    positions = {}
    for idx, node in enumerate(unique_nodes):
        angle = idx / max(1, len(unique_nodes)) * 2 * math.pi
        positions[node] = (math.cos(angle), math.sin(angle))

    edge_x, edge_y = [], []
    for source, target in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=4, color="rgba(96, 165, 250, 0.95)"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(width=4, color="rgba(96, 165, 250, 0.95)"),
    )

    for source, target in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            continue
        angle = math.atan2(dy, dx)
        arrow_len = 0.14
        arrow_x = [x1 - arrow_len * math.cos(angle), x1]
        arrow_y = [y1 - arrow_len * math.sin(angle), y1]
        fig.add_trace(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                line=dict(width=3.5, color="rgba(248, 113, 113, 1.0)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x = [positions[node][0] for node in unique_nodes]
    node_y = [positions[node][1] for node in unique_nodes]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=unique_nodes,
            textposition="top center",
            marker=dict(
                size=24,
                color=["#4C78A8" if i % 2 == 0 else "#F58518" for i in range(len(unique_nodes))],
                line=dict(width=2, color="#1F4E79"),
                symbol="circle",
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig = style_plotly_figure(fig)
    fig.update_layout(
        title="Rule network graph",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def plot_top_consequents(rules_df: pd.DataFrame):
    if rules_df.empty:
        return None

    display_rules = prepare_rules_for_display(rules_df)
    consequent_counter = Counter(item for row in display_rules["consequents"] for item in split_items(row))
    consequent_df = pd.DataFrame(consequent_counter.items(), columns=["item", "count"])
    consequent_df = consequent_df.sort_values(by="count", ascending=False).head(10)
    fig = px.bar(
        consequent_df,
        x="item",
        y="count",
        color="count",
        title="Top consequent items",
        template="plotly_dark",
    )
    fig = style_plotly_figure(fig)
    return fig


def plot_top_uncommon_pairs(rules_df: pd.DataFrame, top_n: int):
    if rules_df.empty:
        return None

    pair_rules = rules_df[
        rules_df["antecedents"].apply(lambda items: len(items) == 1)
        & rules_df["consequents"].apply(lambda items: len(items) == 1)
    ].copy()
    if pair_rules.empty:
        return None

    pair_rules = pair_rules.sort_values(by=["support", "lift", "confidence"], ascending=[True, False, False]).head(top_n)
    pair_rules["antecedent"] = pair_rules["antecedents"].apply(lambda items: next(iter(items)))
    pair_rules["consequent"] = pair_rules["consequents"].apply(lambda items: next(iter(items)))
    pair_rules["pair_label"] = pair_rules.apply(lambda row: f"{row['antecedent']} → {row['consequent']}", axis=1)

    fig = px.bar(
        pair_rules,
        x="support",
        y="pair_label",
        orientation="h",
        color="lift",
        color_continuous_scale="Plasma",
        hover_data={"support": ":.3f", "confidence": ":.2f", "lift": ":.2f"},
        title="Top uncommon pair items",
        labels={"support": "Support", "pair_label": "Item pair", "lift": "Lift"},
        template="plotly_dark",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig = style_plotly_figure(fig)
    return fig


st.set_page_config(page_title="Groceries Association Rules", page_icon="🛒", layout="wide")
st.markdown(
    """
    <style>
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.3); }
            50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.6); }
            100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.3); }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        body, .main, .stApp {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #0f172a, #1a0a2e) !important;
            background-size: 400% 400% !important;
            animation: gradientShift 15s ease infinite !important;
            color: #e2e8f0;
        }
        .block-container, .element-container, .stMarkdown, .stButton {
            background: rgba(15, 23, 42, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(148, 163, 184, 0.12) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            border-radius: 18px !important;
            animation: fadeInUp 0.6s ease-out !important;
        }
        .stApp header {
            background: rgba(15, 23, 42, 0.5) !important;
            backdrop-filter: blur(8px) !important;
        }
        .stButton>button {
            background: linear-gradient(135deg, #1e293b, #334155) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 10px !important;
            padding: 0.8rem 1rem !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #312e81, #4338ca) !important;
            border-color: rgba(99, 102, 241, 0.6) !important;
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
        }
        .stButton>button:active {
            transform: translateY(0) scale(0.98) !important;
        }
        .stApp h1 {
            background: linear-gradient(135deg, #f8fafc, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: "Inter", sans-serif;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .stApp h2 {
            background: linear-gradient(135deg, #e2e8f0, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: "Inter", sans-serif;
            font-weight: 600;
            letter-spacing: -0.3px;
            padding-top: 0.5rem;
        }
        .stApp h3, .stApp h4 {
            color: #f8fafc;
            font-family: "Inter", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(30, 27, 75, 0.9)) !important;
            backdrop-filter: blur(16px) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.08) !important;
        }
        section[data-testid="stSidebar"] .block-container,
        section[data-testid="stSidebar"] .element-container {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.15) !important;
        }
        section[data-testid="stSidebar"] .stSlider label {
            color: #cbd5e1 !important;
        }
        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(8px) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        .stDataFrame td, .stDataFrame th, .stTable td, .stTable th {
            background: rgba(15, 23, 42, 0.4) !important;
            color: #e2e8f0 !important;
            border-color: rgba(148, 163, 184, 0.08) !important;
            padding: 12px 16px !important;
        }
        .stDataFrame th {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.8), rgba(15, 23, 42, 0.8)) !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
        }
        .stDataFrame tr:hover td {
            background: rgba(99, 102, 241, 0.08) !important;
        }
        .metric-container, .stMetric, .stMetric>div {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.6), rgba(15, 23, 42, 0.6)) !important;
            backdrop-filter: blur(8px) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(99, 102, 241, 0.15) !important;
            border-radius: 16px !important;
            padding: 16px !important;
            animation: glowPulse 3s ease-in-out infinite !important;
        }
        .stMetric label {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stSlider div[data-baseweb="slider"] div {
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        }
        .stSlider div[data-baseweb="slider"] div[role="slider"] {
            background: #818cf8 !important;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.5) !important;
        }
        .st-emotion-cache-1y4p8pa {
            padding: 2rem 1.5rem !important;
        }
        .st-emotion-cache-183lzff {
            font-family: "Inter", "Segoe UI", sans-serif !important;
        }
        iframe[title="plotly"] {
            border-radius: 12px !important;
            animation: fadeInUp 0.8s ease-out !important;
        }
        .stCaption {
            color: #94a3b8 !important;
        }
        div.stTabs button {
            background: rgba(15, 23, 42, 0.6) !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        div.stTabs button:hover {
            background: rgba(99, 102, 241, 0.1) !important;
            color: #e2e8f0 !important;
        }
        div.stTabs button[aria-selected="true"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(124, 58, 237, 0.2)) !important;
            color: #f8fafc !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
        }
        .stAlert {
            background: rgba(30, 27, 75, 0.6) !important;
            backdrop-filter: blur(8px) !important;
            border: 1px solid rgba(99, 102, 241, 0.15) !important;
            color: #e2e8f0 !important;
        }
        .st-bb, .st-bc {
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        }
        .st-cx {
            color: #818cf8 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; animation: fadeInUp 0.6s ease-out;">
        <h1 style="
            background: linear-gradient(135deg, #f8fafc, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0;
        ">Groceries Frequent Pattern Mining Dashboard</h1>
        <p style="
            color: #94a3b8;
            font-size: 1rem;
            margin-top: 0.5rem;
            letter-spacing: 0.3px;
        ">Interactive exploration of frequent itemsets and association rules from the grocery transactions dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 2.5rem; animation: float 3s ease-in-out infinite;">🛒</div>
            <h3 style="
                background: linear-gradient(135deg, #f8fafc, #818cf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0.5rem 0 0 0;
            ">Mining Controls</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    min_support = st.slider("Minimum support", 0.005, 0.05, 0.02, 0.005)
    min_confidence = st.slider("Minimum confidence", 0.1, 0.9, 0.3, 0.05)
    top_n = st.slider("Top N items / rules", 5, 58, 10, 1)

    st.markdown(
        """
        <hr style="border-color: rgba(148, 163, 184, 0.1); margin: 1rem 0;">
        <div style="
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(124, 58, 237, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        ">
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0; line-height: 1.5;">
                Dataset: <a href="https://www.kaggle.com/datasets/irfanasrullah/groceries" style="color: #818cf8; text-decoration: none;">Groceries Dataset on Kaggle</a>
            </p>
            <p style="color: #64748b; font-size: 0.75rem; margin: 0.5rem 0 0 0; line-height: 1.4;">
                Adjust the thresholds below to explore how itemsets and rules change in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

results = build_mining_results(min_support=min_support, min_confidence=min_confidence)

frequent_itemsets = results["frequent_itemsets"]
rules = results["rules"]
item_counts = results["item_counts"]
basket_size_series = results["basket_size_series"]

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", len(results["transactions"]))
col2.metric("Frequent itemsets", len(frequent_itemsets))
col3.metric("Association rules", len(rules))

st.subheader("Association rules")
if rules.empty:
    st.info("No rules matched the selected thresholds. Try lowering support or confidence.")
else:
    display_rules = prepare_rules_for_display(rules[["antecedents", "consequents", "support", "confidence", "lift"]]).head(top_n)
    st.dataframe(display_rules, width="stretch")

st.subheader("Visualizations")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    top_items = item_counts.head(top_n).reset_index()
    top_items.columns = ["item", "count"]
    fig_bar = px.bar(
        top_items,
        x="item",
        y="count",
        color="count",
        title=f"Top {top_n} frequent items",
        template="plotly_dark",
    )
    fig_bar = style_plotly_figure(fig_bar)
    st.plotly_chart(fig_bar, width="stretch")

with chart_col2:
    fig_bubble = plot_support_confidence_bubble(rules.head(top_n))
    if fig_bubble is not None:
        st.plotly_chart(fig_bubble, width="stretch")
    else:
        st.info("No bubble chart available for the current threshold.")

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    fig_network = plot_rule_network(rules.head(top_n))
    if fig_network is not None:
        st.plotly_chart(fig_network, width="stretch")
    else:
        st.info("No rule network available for the current threshold.")

with chart_col4:
    fig_consequents = plot_top_consequents(rules.head(top_n))
    if fig_consequents is not None:
        st.plotly_chart(fig_consequents, width="stretch")
    else:
        st.info("No consequent chart available for the current threshold.")

chart_row2_col1, chart_row2_col2 = st.columns(2)
with chart_row2_col1:
    fig_hist = px.histogram(
        basket_size_series.to_frame(name="basket_size"),
        x="basket_size",
        nbins=10,
        title="Distribution of basket sizes",
        template="plotly_dark",
    )
    fig_hist = style_plotly_figure(fig_hist)
    st.plotly_chart(fig_hist, width="stretch")

with chart_row2_col2:
    if rules.empty:
        st.info("No scatter plot available for the current threshold.")
    else:
        scatter_data = prepare_rules_for_display(rules[["confidence", "lift", "antecedents", "consequents"]]).head(top_n)
        scatter_data["bubble_size"] = scatter_data["lift"] * 40
        fig_scatter = px.scatter(
            scatter_data,
            x="confidence",
            y="lift",
            size="bubble_size",
            color="lift",
            color_continuous_scale="Plasma",
            hover_data=["antecedents", "consequents", "confidence", "lift"],
            title="Confidence vs lift",
            labels={"confidence": "Confidence", "lift": "Lift"},
            template="plotly_dark",
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.8)")))
        fig_scatter = style_plotly_figure(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")

chart_row3_col1, chart_row3_col2 = st.columns([1, 1])
with chart_row3_col1:
    if rules.empty:
        st.info("No pie chart available for the current threshold.")
    else:
        antecedent_counter = Counter(item for antecedents in rules["antecedents"] for item in antecedents)
        antecedent_df = pd.DataFrame(antecedent_counter.items(), columns=["item", "count"])
        antecedent_df = antecedent_df.sort_values(by="count", ascending=False).head(top_n)
        fig_pie = px.pie(
            antecedent_df,
            names="item",
            values="count",
            title="Top antecedent items",
            template="plotly_dark",
        )
        fig_pie = style_plotly_figure(fig_pie)
        st.plotly_chart(fig_pie, width="stretch")

with chart_row3_col2:
    fig_top_pairs = plot_top_uncommon_pairs(rules, top_n)
    if fig_top_pairs is not None:
        st.plotly_chart(fig_top_pairs, width="stretch")
    else:
        st.info("No top uncommon pairs available for the current threshold.")

st.subheader("Frequent itemsets preview")
if frequent_itemsets.empty:
    st.info("No frequent itemsets reached the selected support threshold.")
else:
    frequent_preview = prepare_itemsets_for_display(frequent_itemsets.sort_values(by=["support", "itemsets"], ascending=[False, True]).head(top_n))
    st.dataframe(frequent_preview, width="stretch")
