# Plan: Streamlit Dashboard for Frequent Pattern Mining

TL;DR: Turn the grocery transaction mining workflow from the notebook into an interactive Streamlit dashboard that lets users change support and confidence values and immediately see how the mined itemsets and association rules change.

## 1. Data preparation
- Review the notebook workflow and reuse its FPGrowth-based approach for frequent itemsets and association rules.
- Fix the dataset loading issue by using the correct CSV file name and handling missing values consistently.
- Convert the CSV into transaction rows suitable for mining and cache the cleaned dataset for the dashboard.

## 2. Mining logic
- Reuse the notebook’s support/confidence-based mining logic with configurable slider inputs.
- Compute frequent itemsets and association rules dynamically when the user changes the controls.
- Prepare summary metrics such as the number of itemsets, number of rules, and top-ranked rules by lift or confidence.

## 3. Streamlit dashboard UI
- Build a single-page dashboard with a sidebar for support, confidence, and top-N display controls.
- Show an overview section with KPI cards for the number of frequent itemsets, number of rules, and top item frequency.
- Add a rules table that updates as the sliders change.
- Add several visualizations:
  - Bar chart for top 10 frequent items
  - Histogram for item frequency distribution
  - Scatter plot for confidence vs lift
  - Pie chart for the most common antecedents or consequents

## 4. Implementation structure
- Create a Streamlit app entry point that loads the cleaned dataset and runs the mining logic.
- Keep the preprocessing and mining steps separate from the UI so the dashboard remains easier to maintain.
- Add a dependency list and clear local run instructions.
- Place deployment-related files in the deployment folder.

## 5. Verification
- Run the app locally and confirm that changing support and confidence updates the table and charts.
- Check that the dashboard loads successfully with the provided groceries dataset.
- Validate that the visualizations render correctly and that the app remains responsive for the expected dataset size.

## Relevant files
- [Lab_Work_FrequentPatternMining_ID_2023_2_60_101.ipynb](Lab_Work_FrequentPatternMining_ID_2023_2_60_101.ipynb) — main source of mining logic and dataset preparation approach.
- [groceries - groceries.csv](groceries%20-%20groceries.csv) — source dataset for the dashboard.
- deployment/ — location for deployment assets and instructions.

## Scope decisions
- The first version will focus on one interactive dashboard page with core visualizations and configurable mining thresholds.
- The dashboard will use the same grocery dataset and FPGrowth-style workflow from the notebook.
- If performance becomes an issue, the app will cache the cleaned transactions and mined results to reduce reprocessing time.
