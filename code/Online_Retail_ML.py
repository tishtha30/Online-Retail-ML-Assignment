import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv("D:/CU/7072_ML/Assignment/OnlineRetail.csv", encoding="utf-8-sig")
print("Original shape:", df.shape)
print(df.columns.tolist())

df.drop_duplicates(inplace=True)
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[df['Country'] == 'United Kingdom']

print("Rows left:", df.shape)

# Create customer features
df['Total'] = df['Quantity'] * df['UnitPrice']
max_date = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce').max()

features = (
    df.groupby('CustomerID')
      .agg(Total_Spend=('Total','sum'),
           Orders=('InvoiceNo','nunique'),
           Quantity=('Quantity','sum'),
           Last_Date=('InvoiceDate','max'))
      .reset_index()
)

features['Recency'] = (max_date - pd.to_datetime(features['Last_Date'], dayfirst=True, errors='coerce')).dt.days
features['Avg_Basket'] = features['Quantity'] / features['Orders']

print(features.head())
print(features.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Scale and Cluster
X = features[['Total_Spend','Orders','Quantity','Recency','Avg_Basket']]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=0)
features['Cluster'] = kmeans.fit_predict(X_scaled)

print(features.groupby('Cluster')[['Total_Spend','Orders','Recency','Avg_Basket']].mean())

# Cluster metrics
X_eval = StandardScaler().fit_transform(features[['Total_Spend','Orders','Quantity','Recency']])
labels_km = features['Cluster']

sil_km  = silhouette_score(X_eval, labels_km)
dbi_km  = davies_bouldin_score(X_eval, labels_km)

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X_eval)
sil_agg = silhouette_score(X_eval, labels_agg)
dbi_agg = davies_bouldin_score(X_eval, labels_agg)

gmm = GaussianMixture(n_components=4, random_state=0)
labels_gmm = gmm.fit_predict(X_eval)
sil_gmm = silhouette_score(X_eval, labels_gmm)
dbi_gmm = davies_bouldin_score(X_eval, labels_gmm)

print("Silhouette  |  DB-Index")
print(f"K-Means:        {sil_km:.3f}  |  {dbi_km:.3f}")
print(f"Agglomerative:  {sil_agg:.3f}  |  {dbi_agg:.3f}")
print(f"GMM:            {sil_gmm:.3f}  |  {dbi_gmm:.3f}")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
monthly = df.groupby('YearMonth')['Total'].sum().reset_index()

# Monthly trend
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(monthly['YearMonth'], monthly['Total'])
ax.set_title("Monthly Spend Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Total Spend")
plt.xticks(rotation=40, ha='right')
ax.grid(True, linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.savefig("monthly_spend.png", dpi=220)
plt.show()

# Silhouette per cluster
sil_samples = silhouette_samples(X_eval, labels_km)
sil_df = pd.DataFrame({'Cluster': labels_km, 'Sil': sil_samples})
sil_means = sil_df.groupby('Cluster')['Sil'].mean()

fig, ax = plt.subplots(figsize=(5.5,4))
ax.bar(sil_means.index.astype(str), sil_means.values)
ax.set_title("Mean Silhouette by Cluster (K-Means)")
ax.set_xlabel("Cluster")
ax.set_ylabel("Mean Silhouette")
ax.grid(axis='y', linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.savefig("silhouette_per_cluster.png", dpi=220)
plt.show()

# Isolation Forest
iso = IsolationForest(random_state=0, contamination=0.02)
features['Anomaly'] = iso.fit_predict(X_eval)

anom = features[features['Anomaly']==-1]
print("Anomalies found:", len(anom))
print(anom[['CustomerID','Total_Spend','Orders','Recency']].head())

top20 = anom.head(20)

# Cluster bar chart
cluster_profile = (
    features.groupby('Cluster')[['Total_Spend','Orders','Recency']]
    .mean()
    .round(2)
)

fig, ax = plt.subplots(figsize=(6,4))
cluster_profile.plot(kind='bar', ax=ax)
ax.set_title("Average Behaviour per Cluster")
ax.set_ylabel("Average Values")
ax.grid(axis='y', linestyle='--', linewidth=0.5)
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig("cluster_profile.png", dpi=220)
plt.show()

# PCA Plots
p = PCA(n_components=2)
Z = p.fit_transform(X_eval)

# K-Means PCA
fig, ax = plt.subplots(figsize=(6,4))
sc = ax.scatter(Z[:,0], Z[:,1], c=labels_km, s=35, cmap='viridis', edgecolors='k', linewidths=0.3)
ax.set_title("K-Means Clusters (2-D PCA View)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True, linestyle='--', linewidth=0.5)
fig.colorbar(sc)
fig.tight_layout()
fig.savefig("clusters_pca.png", dpi=220)
plt.show()

# Agglomerative PCA
fig, ax = plt.subplots(figsize=(6,4))
sc = ax.scatter(Z[:,0], Z[:,1], c=labels_agg, s=35, cmap='plasma', edgecolors='k', linewidths=0.3)
ax.set_title("Agglomerative Clustering (2-D PCA View)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True, linestyle='--', linewidth=0.5)
fig.colorbar(sc)
fig.tight_layout()
fig.savefig("agglo_pca.png", dpi=220)
plt.show()

# GMM PCA
fig, ax = plt.subplots(figsize=(6,4))
sc = ax.scatter(Z[:,0], Z[:,1], c=labels_gmm, s=35, cmap='cividis', edgecolors='k', linewidths=0.3)
ax.set_title("Gaussian Mixture Model (2-D PCA View)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True, linestyle='--', linewidth=0.5)
fig.colorbar(sc)
fig.tight_layout()
fig.savefig("gmm_pca.png", dpi=220)
plt.show()

# Correlation heatmap
corr = features[['Total_Spend','Orders','Quantity','Recency']].corr()

fig, ax = plt.subplots(figsize=(5.5,4.5))
sns.heatmap(corr, annot=True, cmap='YlGnBu', linewidths=.5, ax=ax)
ax.set_title("Correlation Heatmap (Customer Features)")
fig.tight_layout()
fig.savefig("corr_heatmap.png", dpi=220)
plt.show()

# Radar charts
radar_features = ['Total_Spend','Orders','Quantity','Recency','Avg_Basket']
cluster_radar = features.groupby('Cluster')[radar_features].mean()
norm_radar = (cluster_radar - cluster_radar.min()) / (cluster_radar.max() - cluster_radar.min())

labels = radar_features
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

for idx, row in norm_radar.iterrows():
    stats = row.tolist()
    stats += stats[:1]
    ang = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(ang, stats, linewidth=2)
    ax.fill(ang, stats, alpha=0.25)
    ax.set_title(f"Cluster {idx} Profile (Normalised)")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"])
    plt.ylim(0,1)
    fig.tight_layout()
    fig.savefig(f"cluster_{idx}_radar_normalised.png", dpi=220)
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Prepare data
clf_features = features[['Total_Spend','Orders','Quantity','Recency','Avg_Basket']]
clf_target = features['Cluster']

X_train, X_test, y_train, y_test = train_test_split(
    clf_features, clf_target, test_size=0.25, random_state=0, stratify=clf_target
)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5.8,4.2), dpi=200)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix — Cluster Classification")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig("cluster_confusion_matrix.png", dpi=250)
plt.show()

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=clf_features.columns)

plt.figure(figsize=(5.8,4.2), dpi=200)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig("cluster_feature_importance.png", dpi=250)
plt.show()

# Insights
features.to_csv("features_raw.csv", index=False)

X_df = pd.DataFrame(X_eval, columns=['Total_Spend','Orders','Quantity','Recency'])
X_df.to_csv("features_scaled.csv", index=False)

features[['CustomerID','Cluster']].to_csv("clusters_kmeans.csv", index=False)
top20.to_csv("isolation_forest_top20.csv", index=False)
