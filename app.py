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
        body, .main, .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        .block-container, .element-container, .stMarkdown, .stButton {
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25) !important;
            border-radius: 18px !important;
        }
        .stButton>button {
            background: #1f2937 !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 0.8rem 1rem !important;
            transition: background-color 0.2s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #334155 !important;
            transform: translateY(-1px);
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #f8fafc;
            font-family: "Inter", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: #0f172a !important;
        }
        section[data-testid="stSidebar"] .block-container,
        section[data-testid="stSidebar"] .element-container {
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
        }
        .stDataFrame, .stTable, .css-1lcbmhc, .css-1v3fvcr {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
        }
        .stDataFrame td, .stDataFrame th, .stTable td, .stTable th {
            background: #0f172a !important;
            color: #e2e8f0 !important;
        }
        .metric-container, .stMetric, .stMetric>div {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #e2e8f0 !important;
        }
        .css-1d391kg, .css-1lcbmhc {
            background: transparent !important;
        }
        .css-1y0tads, .css-1r6slb0 {
            color: #e2e8f0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Groceries Frequent Pattern Mining Dashboard")
st.caption("Interactive exploration of frequent itemsets and association rules from the grocery transactions dataset")

with st.sidebar:
    st.header("Mining controls")
    min_support = st.slider("Minimum support", 0.005, 0.05, 0.02, 0.005)
    min_confidence = st.slider("Minimum confidence", 0.1, 0.9, 0.3, 0.05)
    top_n = st.slider("Top N items / rules", 5, 58, 10, 1)

    st.markdown("---")
    st.caption("Dataset: [Groceries Dataset on Kaggle](https://www.kaggle.com/datasets/irfanasrullah/groceries). This dashboard uses the same grocery transaction workflow from the notebook and updates the results as the thresholds change.")

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
    styled_rules = display_rules.style.set_properties(**{
        'background-color': '#0f172a',
        'color': '#e2e8f0',
        'border-color': '#334155',
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#111827'), ('color', '#e2e8f0'), ('border-color', '#334155')]}
    ])
    st.dataframe(styled_rules, width="stretch")

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
    styled_preview = frequent_preview.style.set_properties(**{
        'background-color': '#0f172a',
        'color': '#e2e8f0',
        'border-color': '#334155',
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#111827'), ('color', '#e2e8f0'), ('border-color', '#334155')]}
    ])
    st.dataframe(styled_preview, width="stretch")
