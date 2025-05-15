import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("📊 Exploratory Data Analysis")

# --- Load CSV/Excel files ---
def load_dataframe(file):
    file_type = file.name.split(".")[-1].lower()
    if file_type == "csv":
        return {file.name: pd.read_csv(file)}
    elif file_type in ["xlsx", "xls"]:
        xl = pd.ExcelFile(file)
        return {f"{file.name} - {sheet}": xl.parse(sheet) for sheet in xl.sheet_names}
    return {}

# --- EDA functions ---
def show_data_overview(df):
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🧮 Data Info")
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

def show_summary_stats(df):
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))

def plot_univariate(df):
    st.subheader("🔍 Univariate Distribution")
    col = st.selectbox("Select a column", df.columns, key="univariate")
    if df[col].dtype in ["float", "int"]:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Frequency of {col}")
        st.pyplot(fig)

def plot_multiline(df):
    st.subheader("📉 Line Plot")
    x_col = st.selectbox("X-axis (e.g. time/index)", df.columns, key="line_x")
    y_cols = st.multiselect("Y-axis numeric columns", df.select_dtypes(include=["float", "int"]).columns, key="line_y")
    if x_col and y_cols:
        fig, ax = plt.subplots()
        for col in y_cols:
            ax.plot(df[x_col], df[col], label=col)
        ax.set_title(f"{' & '.join(y_cols)} over {x_col}")
        ax.legend()
        st.pyplot(fig)

def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=["float", "int"])
    if num_df.shape[1] < 2:
        st.info("Not enough numeric features for correlation heatmap.")
        return
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(1.5 * len(num_df.columns), 8))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def filter_dataframe(df):
    st.subheader("🔍 Optional Row Filter")
    cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cols:
        options = df[col].dropna().unique().tolist()
        selection = st.multiselect(f"Filter `{col}`", options, default=options)
        df = df[df[col].isin(selection)]
    return df

def download_button(df, filename, label):
    csv = df.to_csv(index=False)
    b64 = csv.encode("utf-8")
    st.download_button(label=label, data=b64, file_name=filename, mime="text/csv")

# ======================
# SIDEBAR: FILE UPLOAD
# ======================
st.sidebar.header("Upload Data")
uploads = st.sidebar.file_uploader(
    "Upload CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True
)

dataframes = {}
if uploads:
    for file in uploads:
        dataframes.update(load_dataframe(file))
else:
    st.warning("Please upload one or more CSV or Excel files.")
    st.stop()

# ======================
# JOINING SECTION
# ======================
st.subheader("🔗 Optional: Join Two Datasets")

st.markdown("""
**Join Types (in simple terms):**
- **Inner Join**: Only rows with matching keys in both datasets are kept  
- **Left Join**: All rows from the left dataset, plus matches from the right  
- **Right Join**: All rows from the right dataset, plus matches from the left  
- **Outer Join**: All rows from both datasets, matching where possible
""")

df_keys = list(dataframes.keys())
left_name = st.selectbox("Select Left Dataset", df_keys, key="left_ds")
right_name = st.selectbox("Select Right Dataset", df_keys, key="right_ds")

if left_name != right_name:
    left_df = dataframes[left_name]
    right_df = dataframes[right_name]
    common_cols = list(set(left_df.columns) & set(right_df.columns))

    if common_cols:
        join_cols = st.multiselect("Select column(s) to join on", common_cols, key="join_cols")
        join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"], key="join_type")

        if join_cols and st.button("Join Datasets"):
            try:
                joined_df = pd.merge(left_df, right_df, on=join_cols, how=join_type)
                join_label = f"[Joined] {left_name} x {right_name}"
                dataframes[join_label] = joined_df
                st.success(f"{join_type.title()} join complete. Dataset '{join_label}' added.")
                st.dataframe(joined_df.head())
                # Save to CSV
                st.write("💾 Download Joined Dataset")
                download_button(joined_df, f"{join_label}.csv", "📥 Download as CSV")
            except Exception as e:
                st.error(f"Join failed: {e}")
    else:
        st.info("No common columns found to join on.")
else:
    st.info("Please select two different datasets.")

# ======================
# SELECT DATASET FOR EDA
# ======================
st.subheader("📂 Choose Dataset for EDA")
selected_df_name = st.selectbox("Available datasets (raw or joined)", list(dataframes.keys()), key="eda_ds")
df = dataframes[selected_df_name]

# ======================
# COLUMN AND ROW FILTER
# ======================
st.subheader("🔧 Column & Row Selection")
all_columns = df.columns.tolist()
selected_columns = st.multiselect("Select columns for analysis", all_columns, default=all_columns, key="eda_cols")

if not selected_columns:
    st.warning("Please select at least one column.")
    st.stop()

df = df[selected_columns]
df = filter_dataframe(df)

# --- Save Selected Subset ---
st.markdown("💾 Download Selected Dataset")
download_button(df, f"{selected_df_name}_selected.csv", "📥 Download Filtered CSV")

# ======================
# EDA ON SELECTED DF
# ======================
show_data_overview(df)
show_summary_stats(df)
plot_univariate(df)
plot_multiline(df)
plot_correlation_heatmap(df)
