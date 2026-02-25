import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard (RFM + KMeans)")

api_url = st.text_input("FastAPI URL", value="http://api:8000")

uploaded = st.file_uploader("Upload a CSV file (RFM table or raw transactions)", type=["csv"])
st.caption("Accepted: either already-computed RFM per customer OR raw invoices to compute RFM.")

# ----------------------------
# Helpers
# ----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def has_rfm_cols(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return ("recency" in cols and "frequency" in cols and ("monetary_value" in cols or "monetary" in cols))

def compute_rfm_from_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns commonly used in Online Retail-like datasets:
    CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice
    """
    df = df.copy()
    # Common cleanups
    # Remove cancellations if InvoiceNo exists (credit notes usually start with 'C')
    if "InvoiceNo" in df.columns:
        df = df[~df["InvoiceNo"].astype(str).str.contains("C", na=False)]

    required = ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transaction mode: {missing}")

    # Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])

    # Numeric cleanup
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # Remove weird negatives/zeros
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Reference date = max invoice date + 1 day
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Frequency: unique invoices if InvoiceNo exists; else count rows per customer
    if "InvoiceNo" in df.columns:
        freq_series = df.groupby("CustomerID")["InvoiceNo"].nunique()
    else:
        freq_series = df.groupby("CustomerID").size()

    monetary = df.groupby("CustomerID")["TotalPrice"].sum()
    last_date = df.groupby("CustomerID")["InvoiceDate"].max()
    recency = (ref_date - last_date).dt.days

    rfm = pd.DataFrame({
        "CustomerID": recency.index,
        "recency": recency.values.astype(float),
        "frequency": freq_series.reindex(recency.index).values.astype(float),
        "monetary_value": monetary.reindex(recency.index).values.astype(float)
    })

    # Basic NA safety
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["recency", "frequency", "monetary_value"])
    return rfm

def standardize_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts RFM table with flexible names:
    Recency/Frequency/Monetary or recency/frequency/monetary_value
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(name_options):
        for n in name_options:
            if n in cols_lower:
                return cols_lower[n]
        return None

    r_col = pick(["recency"])
    f_col = pick(["frequency"])
    m_col = pick(["monetary_value", "monetary"])

    if not (r_col and f_col and m_col):
        raise ValueError("Could not find RFM columns.")

    out = pd.DataFrame({
        "recency": pd.to_numeric(df[r_col], errors="coerce"),
        "frequency": pd.to_numeric(df[f_col], errors="coerce"),
        "monetary_value": pd.to_numeric(df[m_col], errors="coerce"),
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["recency"] >= 0) & (out["frequency"] >= 0) & (out["monetary_value"] >= 0)]
    return out

def predict_clusters_batch(rfm_df: pd.DataFrame, api_url: str) -> list[int]:
    payload = rfm_df[["recency", "frequency", "monetary_value"]].to_dict(orient="records")
    r = requests.post(f"{api_url}/predict_batch", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["clusters"]

def plot_feature_by_cluster(df: pd.DataFrame, feature: str, title: str):
    # Barplot per cluster (clear + standard for RFM segmentation)
    cluster_means = df.groupby("cluster")[feature].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="cluster", y=feature, data=cluster_means, palette
                    =sns.color_palette("Set2", n_colors=cluster_means["cluster"].nunique()), ax=ax)
    ax.set_title(title)
    st.pyplot(fig)



# ----------------------------
# Main flow
# ----------------------------
if uploaded:
    df_raw = pd.read_csv(uploaded)
    df_raw = normalize_cols(df_raw)

    st.subheader("Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    mode = "RFM table" if has_rfm_cols(df_raw) else "Transactions (compute RFM)"
    st.info(f"Detected input type: **{mode}**")

    try:
        if has_rfm_cols(df_raw):
            rfm = standardize_rfm_table(df_raw)
            # if CustomerID exists, keep it for display
            if "CustomerID" in df_raw.columns:
                rfm.insert(0, "CustomerID", df_raw["CustomerID"].values[: len(rfm)])
        else:
            rfm = compute_rfm_from_transactions(df_raw)

        st.subheader("Computed / Loaded RFM (per customer)")
        st.dataframe(rfm.head(20), use_container_width=True)

        # Predict clusters
        clusters = predict_clusters_batch(rfm, api_url)
        rfm_out = rfm.copy()
        rfm_out["cluster"] = clusters

        st.subheader("Segmented Customers")
        st.dataframe(rfm_out.head(50), use_container_width=True)

        # Basic segment stats
        st.subheader("Segment Summary")
        summary = rfm_out.groupby("cluster")[["recency", "frequency", "monetary_value"]].mean().round(2)
        st.dataframe(summary, use_container_width=True)

        # 3 plots (R, F, M)
        st.subheader("RFM Distributions by Cluster")
        c1, c2, c3 = st.columns(3)
        with c1:
            plot_feature_by_cluster(rfm_out, "recency", "Recency by Cluster")
        with c2:
            plot_feature_by_cluster(rfm_out, "frequency", "Frequency by Cluster")
        with c3:
            plot_feature_by_cluster(rfm_out, "monetary_value", "Monetary Value by Cluster")

        # Download segmented output
        st.subheader("Download Output")
        csv_bytes = rfm_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download segmented RFM CSV", data=csv_bytes, file_name="segmented_rfm.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()
else:
    st.warning("Upload a CSV to begin.")