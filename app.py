
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data():
    df = pd.read_csv("Customer_Segmentation_Dataset.csv", low_memory=False, on_bad_lines='skip', skip_blank_lines=True)
    df = df.dropna(subset=['Income'])
    df = df.drop_duplicates(subset='ID')
    df = df.drop(columns='ID')
    df['Age'] = datetime.now().year - df['Year_Birth']
    df = df.drop(columns='Year_Birth')

    education_order = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    df['Education'] = df['Education'].map(education_order)

    to_drop = df[df['Marital_Status'].isin(['Alone', 'Absurd', 'YOLO'])].index
    df = df.drop(to_drop)
    mapping = {
        'Married': 0,
        'Together': 1,
        'Single': 2,
        'Divorced': 3,
        'Widow': 4
    }
    df['Marital_Status'] = df['Marital_Status'].map(mapping)

    df['Income'] = (df['Income'] - df['Income'].mean()) / df['Income'].std()
    df['Total_Youth'] = df['Kidhome'] + df['Teenhome']
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'],dayfirst=True)
    today = pd.to_datetime('today')
    df['Tenure_Months'] = (today.year - df['Dt_Customer'].dt.year) * 12 + (today.month - df['Dt_Customer'].dt.month)
    df.drop(columns=['Dt_Customer'], inplace=True)

    def recency_binning(days):
        if days <= 30:
            return 'Very Recent'
        elif days <= 60:
            return 'Recent'
        elif days <= 90:
            return 'Less Recent'
        else:
            return 'Inactive'

    df['Recency'] = df['Recency'].apply(recency_binning)
    map_rec = {'Very Recent': 3, 'Recent': 2, 'Less Recent': 1, 'Inactive': 0}
    df['Recency'] = df['Recency'].map(map_rec)

    features_to_scale = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", 
                         "MntSweetProducts", "MntGoldProds", "Age"]

    for col in features_to_scale:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    def categorize(value, thresholds=[2, 5, 9]):
        if value <= thresholds[0]: return 'Low'
        elif value <= thresholds[1]: return 'Medium'
        elif value <= thresholds[2]: return 'High'
        else: return 'Very High'

    cols_map = {
        'Deal_Sensitivity': ('NumDealsPurchases', [0, 2, 5]),
        'WebPurchaseCategory_Bin': ('NumWebPurchases', [2, 5, 9]),
        'CatalogPurchaseCategory': ('NumCatalogPurchases', [2, 5, 8]),
        'StorePurchaseCategory': ('NumStorePurchases', [2, 5, 9]),
        'WebVisitsCategory': ('NumWebVisitsMonth', [2, 5, 8]),
    }

    for new_col, (orig_col, thresh) in cols_map.items():
        df[new_col] = df[orig_col].apply(lambda x: categorize(x, thresh))
        ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
        df[new_col] = df[new_col].map(ordinal_map)

    df.drop(columns=["Z_Revenue", "Z_CostContact"], inplace=True, errors='ignore')
    return df


st.title("Customer Segmentation Clustering")
df = load_and_preprocess_data()

k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

features = df.select_dtypes(include=['float64', 'int64'])
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

st.write("### Cluster Profiling")
cluster_profile = df.groupby('Cluster').mean(numeric_only=True)
st.dataframe(cluster_profile)

# Visualize with pairplot or barplot
st.write("### Cluster Visualization (Age vs Income)")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Income', hue='Cluster', palette='viridis')
st.pyplot(plt)

st.write("### Number of Customers per Cluster")
st.bar_chart(df['Cluster'].value_counts().sort_index())
