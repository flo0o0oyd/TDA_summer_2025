from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import oat_python as oat
import os

# config

MIN_RELEVANCE = 0.7
MIN_YEAR = 1920
MAX_YEAR = 2021
MIN_CONCEPT_FREQ = 0.0001
MAX_CONCEPT_FREQ = 0.001
MAX_DIM = 1
DATA_DIR = "cache"
YEARS_GRID = np.linspace(0, 1, 100)
INV_COUNTS_GRID = np.linspace(0, 1, 40)
n_clusters = 3

# Data processing functions

def dataprocess(df):
    df = df[df['relevance_mean'] >= MIN_RELEVANCE]
    df = df[df['year'] >= MIN_YEAR]
    num_articles = df['article_id'].nunique()
    concept_freq = df.groupby('concept').transform('size') / num_articles
    df = df[(concept_freq >= MIN_CONCEPT_FREQ) & (concept_freq <= MAX_CONCEPT_FREQ)]
    return df[['article_id', 'concept', 'year']]

def conceptprocess(df):
    concepts = (
            df
                .sort_values('year')
                .groupby('concept')
                .agg(
                    year=('year', 'min'),
                    count=('article_id', 'nunique')
                )
                .reset_index()
        )

    concepts['norm_year'] = (concepts['year'] - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    concepts['inv_count'] = 1 / concepts['count']
    return(concepts)

def edgeprocess(df):
    edges = df.merge(df, on=['article_id', 'year'], suffixes=['_source', '_target'])
    edges = edges[edges['concept_source'] < edges['concept_target']]
    edges = edges.groupby(['concept_source', 'concept_target']).agg(
            year=('year', 'min'),
            count=('article_id', 'nunique')
        ).reset_index()
    edges['norm_year'] = (edges['year'] - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
    edges['inv_count'] = 1 / edges['count']
    return(edges)

def graphprocess(concepts,edges):
    G = nx.Graph()
    G.add_nodes_from([(c, {'norm_year': ny, 'inv_count': ic}) for c, ny, ic in zip(concepts['concept'], concepts['norm_year'], concepts['inv_count'])])
    G.add_edges_from([(u, v, {'norm_year': ny, 'inv_count': ic}) for u, v, ny, ic in zip(edges['concept_source'], edges['concept_target'], edges['norm_year'], edges['inv_count'])])
    return(G)

def processbetticurve(G):
    adj_year = nx.adjacency_matrix(G, weight='norm_year')
    adj_year.setdiag([d['norm_year'] for _, d in G.nodes(data=True)])
    adj_inv_count = nx.adjacency_matrix(G, weight='inv_count')
    adj_inv_count.setdiag([d['inv_count'] for _, d in G.nodes(data=True)])
    adj_year = adj_year.sorted_indices()
    betti_curves = np.empty((len(YEARS_GRID), len(INV_COUNTS_GRID), MAX_DIM + 1))
    return betti_curves

def runcrocker(G, YEARS_GRID, INV_COUNTS_GRID):
    adj_year = nx.adjacency_matrix(G, weight='norm_year')
    adj_year.setdiag([d['norm_year'] for _, d in G.nodes(data=True)])
    adj_inv_count = nx.adjacency_matrix(G, weight='inv_count')
    adj_inv_count.setdiag([d['inv_count'] for _, d in G.nodes(data=True)])
    adj_year = adj_year.sorted_indices()

    betti_curves = np.empty((len(YEARS_GRID), len(INV_COUNTS_GRID), MAX_DIM + 1))

    for i, c in enumerate(INV_COUNTS_GRID):
        # zero out things not included
        c_adj = adj_year.copy()
        c_adj[adj_inv_count > c] = 0
        c_adj.eliminate_zeros()

        c_adj.setdiag([d['norm_year'] for _, d in G.nodes(data=True)])
        c_adj = c_adj.sorted_indices()

        if c_adj.nnz == 0 or c_adj.shape[0] == 0:
            for d in range(MAX_DIM + 1):
                betti_curves[:, i, d] = 0
            continue

        try:
            factored = oat.rust.FactoredBoundaryMatrixVr(c_adj, MAX_DIM)
            homology = factored.homology(False, False)

            for d in range(MAX_DIM + 1):
                dim_homology = homology[homology['dimension'] == d]
                betti_curves[:, i, d] = ((dim_homology['birth'].values <= YEARS_GRID[:, None]) &
                                        (dim_homology['death'].values > YEARS_GRID[:, None])).sum(axis=1)
        except Exception as e:
            print(f"OAT error at inv_count={c:.3f}: {e}")
            for d in range(MAX_DIM + 1):
                betti_curves[:, i, d] = 0

    return betti_curves

def mainfunc(df, YEARS_GRID, INV_COUNTS_GRID):
    df = dataprocess(df)
    concepts = conceptprocess(df)
    edges = edgeprocess(df)
    G = graphprocess(concepts, edges)
    betti_curves = runcrocker(G, YEARS_GRID, INV_COUNTS_GRID)
    return betti_curves

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
X_vecs = []
field_names = []

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    field = fname.replace(".csv", "").replace("_", " ")

    try:
        df = pd.read_csv(path)
        betti_curves = mainfunc(df, YEARS_GRID, INV_COUNTS_GRID)
        if betti_curves.shape != (100, 40, 2):
            print(f"{field} skipped: shape {betti_curves.shape}")
            continue

        vec = betti_curves.flatten()
        X_vecs.append(vec)
        field_names.append(field)
        print(f"{field} vectorized.")

    except Exception as e:
        print(f"Failed on {field}: {e}")

X = np.array(X_vecs)
print("All fields processed. Final shape:", X.shape)

X = np.array(X_vecs)
fields = np.array(field_names)
_, unique_indices = np.unique(fields, return_index=True)
X_unique = X[unique_indices]
fields_unique = fields[unique_indices]

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_unique)
cluster_df = pd.DataFrame({
    "field": fields_unique,
    "cluster": labels
}).sort_values("cluster")
print(cluster_df)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_unique)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=60)
plt.colorbar(scatter)
plt.grid(True)
plt.show()

Z = linkage(X_unique, method='ward', metric='euclidean')

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=fields_unique, leaf_rotation=90)
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()

hier_labels = fcluster(Z, t=3, criterion='maxclust')

hier_cluster_df = pd.DataFrame({
    "field": fields_unique,
    "cluster": hier_labels
}).sort_values("cluster")

print("\nHierarchical Clustering Results:")
print(hier_cluster_df)

print("\nComparison of K-means and Hierarchical Clustering:")
comparison_df = cluster_df.copy()
comparison_df["hier_cluster"] = hier_cluster_df["cluster"]
print(comparison_df)


