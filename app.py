import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    df = pd.read_csv(
        "Customer_Segmentation_Dataset.csv",
        low_memory=False,
        on_bad_lines="skip",
        skip_blank_lines=True
    )
    df = df.dropna(subset=["Income"])
    df = df.drop_duplicates(subset="ID")
    df = df.drop(columns="ID")
    df["Age"] = datetime.now().year - df["Year_Birth"]
    df = df.drop(columns="Year_Birth")
    df["Education"] = df["Education"].map({
        "Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4
    })
    df = df.drop(df[df["Marital_Status"].isin(["Alone", "Absurd", "YOLO"])].index)
    df["Marital_Status"] = df["Marital_Status"].map({
        "Married": 0, "Together": 1, "Single": 2, "Divorced": 3, "Widow": 4
    })
    df_raw = df.copy()
    df["Income"] = (df["Income"] - df["Income"].mean()) / df["Income"].std()
    df["Total_Youth"] = df["Kidhome"] + df["Teenhome"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    today = pd.to_datetime("today")
    df["Tenure_Months"] = (
        (today.year - df["Dt_Customer"].dt.year) * 12
        + (today.month - df["Dt_Customer"].dt.month)
    )
    df = df.drop(columns="Dt_Customer")
    def recency_binning(x):
        if x <= 30: return "Very Recent"
        if x <= 60: return "Recent"
        if x <= 90: return "Less Recent"
        return "Inactive"
    df["Recency"] = df["Recency"].apply(recency_binning).map({
        "Very Recent": 3, "Recent": 2, "Less Recent": 1, "Inactive": 0
    })
    for col in [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds", "Age"
    ]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    def categorize(v, t): 
        if v <= t[0]: return "Low"
        if v <= t[1]: return "Medium"
        if v <= t[2]: return "High"
        return "Very High"
    mapping = {
        "Deal_Sensitivity": ("NumDealsPurchases", [0,2,5]),
        "WebPurchaseCategory_Bin": ("NumWebPurchases", [2,5,9]),
        "CatalogPurchaseCategory": ("NumCatalogPurchases", [2,5,8]),
        "StorePurchaseCategory": ("NumStorePurchases", [2,5,9]),
        "WebVisitsCategory": ("NumWebVisitsMonth", [2,5,8])
    }
    for new_col,(orig,t) in mapping.items():
        df[new_col] = df[orig].apply(lambda x: categorize(x, t)).map({
            "Low":0, "Medium":1, "High":2, "Very High":3
        })
    df = df.drop(columns=["Z_Revenue", "Z_CostContact"], errors="ignore")
    df_raw = df_raw.drop(columns=["Z_Revenue", "Z_CostContact"], errors="ignore")
    return df, df_raw

st.title("Customer Segmentation Clustering")
df, df_raw = load_and_preprocess_data()

X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=24)
X_pca = pca.fit_transform(X_scaled)

st.write("Explained variance ratio by component:")
st.write(pca.explained_variance_ratio_)

k = st.slider("Select number of clusters (k)", 2, 10, 4)

inertia = []
for i in range(1, 11):
    inertia.append(KMeans(n_clusters=i, random_state=42).fit(X_pca).inertia_)
plt.figure()
plt.plot(range(1, 11), inertia, "bo-")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
st.pyplot(plt.gcf())
plt.clf()

kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
labels = kmeans.labels_
df["Cluster"] = labels
df_raw["Cluster"] = labels

cluster_modes = (
    df_raw
    .groupby("Cluster")
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)
st.write("Cluster mode profiles:")
st.dataframe(cluster_modes)

st.write("PCA scatter plot of first two components:")
plt.figure()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis")
st.pyplot(plt.gcf())
plt.clf()

st.write("Number of customers per cluster:")
st.bar_chart(df["Cluster"].value_counts().sort_index())
