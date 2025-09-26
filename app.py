import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load models ---
clf = joblib.load("models/clf_pipeline.joblib")
reg = joblib.load("models/reg_pipeline.joblib")
kmeans = joblib.load("models/kmeans.joblib")

st.title("🛒 Customer Conversion Demo App")

st.write("Upload a dataset (CSV or Excel) to get predictions for conversion, revenue, and customer clustering.")

# --- File uploader ---
uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# --- Helper to align dataset columns ---
def align_df(df, required_cols, num_cols):
    df2 = df.copy()
    for c in required_cols:
        if c not in df2.columns:
            if c in num_cols:
                df2[c] = 0
            else:
                df2[c] = "missing"
    return df2[required_cols].copy()

# --- When file is uploaded ---
if uploaded is not None:
    # Read uploaded file
    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)

    st.write("### Preview uploaded data", raw.head())

    # Try to extract column lists from pipeline
    try:
        preproc = clf.named_steps["preproc"]
        num_cols, cat_cols = [], []
        for name, _, cols in preproc.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
    except:
        # fallback (update if needed)
        num_cols = ["YEAR", "MONTH", "DAY", "SESSION_ID", "LOCATION", "PRICE", "PRICE2", "PAGE"]
        cat_cols = ["COUNTRY", "PAGE1", "PAGE2", "COLOUR", "MODEL_PHOTOGRAPHY"]

    feature_cols = num_cols + cat_cols

    # Align uploaded file to training schema
    X_for_models = align_df(raw, feature_cols, num_cols)

    # Convert numeric safely
    for c in num_cols:
        if c in X_for_models.columns:
            X_for_models[c] = pd.to_numeric(X_for_models[c], errors="coerce").fillna(0)

    # --- Predictions ---
    preds_converted = clf.predict(X_for_models)
    preds_revenue = reg.predict(X_for_models)

    # Clustering: preprocess then cluster
    X_transformed = clf.named_steps["preproc"].transform(X_for_models)
    clusters = kmeans.predict(X_transformed)

    # Combine with original data
    out_df = raw.copy().reset_index(drop=True)
    out_df["Predicted_Converted"] = preds_converted
    out_df["Predicted_Revenue"] = np.round(preds_revenue, 2)
    out_df["Cluster"] = clusters

    st.write("### Predictions", out_df.head())

    # Download button
    st.download_button(
        "Download Predictions",
        out_df.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
    )
