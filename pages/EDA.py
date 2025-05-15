import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("📊 Exploratory Data Analysis")

# --- Load CSV or Excel file ---
def load_data(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "csv":
            return pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# --- Show raw data, types, nulls ---
def show_data_overview(df):
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🧮 Data Info")
    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.write("**Missing Values:**")
    st.write(df.isnull().sum())


# --- Summary statistics ---
def show_summary_stats(df):
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))


# --- Univariate chart (histogram or bar) ---
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
        ax.set_ylabel("Count")
        st.pyplot(fig)


# --- Multivariate line plot ---
def plot_multiline(df):
    st.subheader("📉 Line Plot")
    time_col = st.selectbox("Select X-axis (usually time or index)", df.columns, key="line_x")
    line_cols = st.multiselect("Select numeric columns for Y-axis", df.select_dtypes(include=["float", "int"]).columns, key="line_y")

    if time_col and line_cols:
        fig, ax = plt.subplots()
        for col in line_cols:
            ax.plot(df[time_col], df[col], label=col)
        ax.set_title(f"{' vs '.join(line_cols)} over {time_col}")
        ax.set_xlabel(time_col)
        ax.set_ylabel("Values")
        ax.legend(title="Features")
        st.pyplot(fig)


# --- Correlation heatmap ---
def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=["float", "int"])
    if num_df.shape[1] < 2:
        return
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# --- Sidebar: file upload ---
st.sidebar.header("Data Options")
upload = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

# --- Load dataset ---
if upload:
    df = load_data(upload)
    if df is not None:
        st.success("Data uploaded successfully.")
else:
    st.info("No file uploaded. Using example dataset: `penguins`.")
    df = sns.load_dataset("penguins")

# --- Feature selection ---
if df is not None:
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to analyze", all_columns, default=all_columns)

    if not selected_columns:
        st.warning("Please select at least one column.")
        st.stop()
    df = df[selected_columns]

    # --- Analysis sections ---
    show_data_overview(df)
    show_summary_stats(df)
    plot_univariate(df)
    plot_multiline(df)
    plot_correlation_heatmap(df)

