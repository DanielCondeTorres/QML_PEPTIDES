# ================================
# 8. Visualization of SVM decision boundary in PCA 2D
# ================================
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Necessary import for SVM
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
# ===============================
# 1. Read and clean Excel or CSV file
# ===============================

# Set the input file name (change as needed)
excel_file = "peptides_transformados.xlsx"

# Read the Excel file using pandas
# If the file has multiple sheets, you can specify sheet_name if needed
try:
    df_raw = pd.read_excel(excel_file, header=None)
    print(df_raw)
except Exception as e:
    print(f"Error reading the Excel file: {e}")


# X: peptide names (from the first row, skipping the first column)
X = df_raw.iloc[0, 1:].values

# y: features for each peptide (all rows except the first, all columns except the first)
y = df_raw.iloc[2:, 1:].values.T  # Transpose so each row is a peptide, each column a feature

print("Peptide names (X):", X)
print("Feature matrix (y):", y)
print("y shape (samples, features):", y.shape)

# --- Safely convert the feature matrix y to numeric ---
y_df = pd.DataFrame(y)
y_numeric = y_df.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, set errors to NaN

# Check for NaN values after conversion
if y_numeric.isnull().values.any():
    print("Warning: There are NaN values after conversion. Dropping rows with NaN.")
    y_numeric = y_numeric.dropna()  # Drop rows with any NaN values

# Convert back to numpy array if needed
y_numeric = y_numeric.values

# --- PCA Analysis ---
# y: feature matrix (each row is a peptide, each column is a feature)
# Make sure y is numeric
try:
    y_numeric = y.astype(float)
except Exception as e:
    print(f"Error converting features to float: {e}")
    y_numeric = y

from sklearn.decomposition import PCA
#Prueba# Fit PCA to the feature matrix
def plot_pca_variance(y_matrix, max_components=10):
    pca = PCA()
    y_pca = pca.fit_transform(y_matrix)
    sum_variance = np.cumsum(pca.explained_variance_ratio_)
    components = range(1, min(max_components, y_matrix.shape[1]) + 1)
    plt.figure(figsize=(8, 5))
    for i, s in enumerate(sum_variance[:len(components)], start=1):
        plt.annotate(f"{s:.3f}", xy=(i, s))
    plt.plot(components, sum_variance[:len(components)], "-o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return pca, y_pca

# Call the function to plot PCA variance
pca, y_pca = plot_pca_variance(y_numeric, max_components=10)

# --- Select number of components to explain at least 95% of the variance ---
explained_var = np.cumsum(pca.explained_variance_ratio_)
# Find the number of components needed for >= 95% variance
n_components_95 = np.argmax(explained_var >= 0.95) + 1
print(f"Number of components to explain at least 95% variance: {n_components_95}")

# Transform the data using the selected number of components
pca_95 = PCA(n_components=n_components_95)
y_pca_95 = pca_95.fit_transform(y_numeric)
print(f"Shape of PCA-reduced data: {y_pca_95.shape}")

# --- Show feature contributions (loadings) for each principal component ---
# Get feature names from the first column, skipping the first row
feature_names = df_raw.iloc[1:, 0].values

# Get the loadings matrix (components_): shape (n_components, n_features)
loadings = pca_95.components_

# For each principal component, print the top contributing features
for i, component in enumerate(loadings, start=1):
    # Get absolute value of loadings for sorting
    abs_loadings = np.abs(component)
    # Get indices of top 5 contributing features
    top_indices = abs_loadings.argsort()[::-1][:5]
    print(f"\nPrincipal Component {i} (explains {pca_95.explained_variance_ratio_[i-1]:.2%} variance):")
    for idx in top_indices:
        print(f"  Feature: {feature_names[idx]}, Loading: {component[idx]:.3f}")

# Create a list of principal component names: 'PC1', 'PC2', ..., up to n_components_95
pc_names = [f'PC{i+1}' for i in range(n_components_95)]

# Create a DataFrame with the PCA-reduced data and these column names
X_main = y_pca_95  # y_pca_95 is the PCA-reduced data
X_main_df = pd.DataFrame(X_main, columns=pc_names)

print(X_main_df.head())  # Show the first few rows of the PCA DataFrame

# Pairplot: each PC vs each other, histograms on the diagonal, no hue
sns.pairplot(X_main_df)
plt.suptitle('Pairplot of principal components', y=1.02)
plt.savefig('Pic.png', dpi=300, bbox_inches='tight')
plt.show()

# K-MEANS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Find the best number of clusters using silhouette score ---
k_range = range(2, 11)  # Try k from 2 to 10
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_main)
    score = silhouette_score(X_main, labels)
    silhouette_scores.append(score)

# Find the best k (the one with the highest silhouette score)
best_k = k_range[np.argmax(silhouette_scores)]
print(f"Best number of clusters (k) according to silhouette score: {best_k}")

# Plot silhouette scores for each k
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for different k in KMeans')
plt.grid(True)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
plt.legend()
plt.tight_layout()
plt.savefig('Silhouette_KMeans.png', dpi=300, bbox_inches='tight')
plt.show()

# Fit KMeans with the best number of clusters
kmeans_main = KMeans(n_clusters=best_k, random_state=42)
y_kmeans = kmeans_main.fit_predict(X_main)
sil_score_kmeans = np.round(silhouette_score(X_main, y_kmeans), 2)

print(f"Silhouette coefficient for the best KMeans model: {sil_score_kmeans}")

# ===============================
# Agglomerative Hierarchical Clustering and Comparison with KMeans
# ===============================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score

# --- Dendrogram and automatic optimal number of clusters selection ---
# Compute the linkage matrix for hierarchical clustering
Z = linkage(X_main, method='ward')
plt.figure(figsize=(10, 10))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distance')
plt.show()

# Find the optimal number of clusters from the dendrogram
# We use the largest distance gap in the last 10 merges
last = Z[-10:, 2]
max_gap = np.argmax(last[1:] - last[:-1]) + 1
optimal_clusters = len(last) - max_gap + 1
print(f"Optimal number of clusters from dendrogram: {optimal_clusters}")

# --- Agglomerative clustering with the optimal number of clusters ---
hc = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_main)

# Calculate silhouette score for Agglomerative clustering
sil_score_hc = np.round(silhouette_score(X_main, y_hc), 4)
print(f'Silhouette coefficient for Agglomerative clustering: {sil_score_hc}')

# Calculate cluster centers for Agglomerative (mean of points in each cluster, for PC1 and PC2)
agg_centers = np.array([X_main[y_hc == i, :2].mean(axis=0) for i in range(optimal_clusters)])

# ===============================
# Subplots: KMeans vs Agglomerative Clustering in PCA space
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
sns.set(style="whitegrid", context="notebook", font_scale=1.3)
palette_kmeans = sns.color_palette('tab10', n_colors=best_k)
palette_agg = sns.color_palette('tab10', n_colors=optimal_clusters)

# --- KMeans plot (PC1 vs PC2) ---
sns.scatterplot(
    x=X_main_df['PC1'], y=X_main_df['PC2'],
    hue=y_kmeans, palette=palette_kmeans, legend='full', s=120, edgecolor='black', alpha=0.85, ax=axes[0]
)
centers = kmeans_main.cluster_centers_[:, :2]
axes[0].scatter(
    centers[:, 0], centers[:, 1], c='gold', s=300, marker='*', edgecolor='black', label='Cluster center', zorder=10
)
axes[0].set_title(f'KMeans Clusters (k={best_k})', fontsize=16, weight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(title='Cluster', title_fontsize=13, fontsize=12, loc='best', frameon=True)

# --- Agglomerative plot (PC1 vs PC2) ---
sns.scatterplot(
    x=X_main_df['PC1'], y=X_main_df['PC2'],
    hue=y_hc, palette=palette_agg, legend='full', s=120, edgecolor='black', alpha=0.85, ax=axes[1]
)
axes[1].scatter(
    agg_centers[:, 0], agg_centers[:, 1], c='gold', s=300, marker='*', edgecolor='black', label='Cluster center', zorder=10
)
axes[1].set_title(f'Agglomerative Clusters (k={optimal_clusters})', fontsize=16, weight='bold')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(title='Cluster', title_fontsize=13, fontsize=12, loc='best', frameon=True)

plt.suptitle('PC1 vs PC2: KMeans vs Agglomerative Clustering', fontsize=20, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('PC1_vs_PC2_KMeans_vs_Agglomerative.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# Barplot comparing silhouette scores of KMeans and Agglomerative
# ===============================

# Prepare silhouette scores as percentages
silhouette_scores_dict = {
    'KMeans': sil_score_kmeans * 100,
    'Agglomerative': sil_score_hc * 100
}
silhouette_scores_df = pd.DataFrame(list(silhouette_scores_dict.items()), columns=['Model', 'Silhouette %'])

plt.figure(figsize=(7, 6))
sns.barplot(x='Model', y='Silhouette %', data=silhouette_scores_df, palette='viridis')
for i, row in silhouette_scores_df.iterrows():
    plt.text(i, row['Silhouette %'] + 1, f"{row['Silhouette %']:.2f}%", ha='center', fontsize=13, weight='bold')
plt.ylabel('Silhouette Score (%)', fontsize=14)
plt.xlabel('Clustering Model', fontsize=14)
plt.title('Comparison of Silhouette Scores for Clustering Models', fontsize=16, weight='bold')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('Silhouette_Score_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# Improved Quantum K-Means implementation and comparison
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class QuantumKMeansFidelity:
    def __init__(self, n_clusters=2, n_qubits=4, max_iter=100, random_state=None, shots=256, use_ibmq=False, ibmq_backend_name=None):
        """
        Quantum K-Means clustering using swap test fidelity as distance (Lloyd et al. 2013).
        Optionally, run the quantum circuit on an IBM Quantum backend if use_ibmq=True and ibmq_backend_name is provided.
        """
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.random_state = random_state
        self.shots = shots
        self.use_ibmq = use_ibmq
        self.ibmq_backend_name = ibmq_backend_name
        self.cluster_centers_ = None
        self.labels_ = None
        self.mean_ = None
        self.std_ = None

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = []
        centroids.append(X[np.random.randint(n_samples)])
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for idx, prob in enumerate(cumulative_probs):
                if r < prob:
                    centroids.append(X[idx])
                    break
        return np.array(centroids)

    def _swap_test_fidelity(self, x1, x2):
        """
        Quantum fidelity using the swap test (Lloyd et al. 2013).
        If use_ibmq is True and ibmq_backend_name is set, run on IBM Quantum backend.
        """
        from qiskit import transpile
        x1_norm = x1 / (np.linalg.norm(x1) + 1e-8)
        x2_norm = x2 / (np.linalg.norm(x2) + 1e-8)
        qc = QuantumCircuit(1 + 2*self.n_qubits, 1)
        qc.h(0)
        # Angle encoding for x1
        for i in range(self.n_qubits):
            qc.ry(x1_norm[i], i+1)
        # Angle encoding for x2
        for i in range(self.n_qubits):
            qc.ry(x2_norm[i], i+1+self.n_qubits)
        # Controlled swaps
        for i in range(self.n_qubits):
            qc.cswap(0, i+1, i+1+self.n_qubits)
        qc.h(0)
        qc.measure(0, 0)
        if self.use_ibmq and self.ibmq_backend_name is not None:
            # Try to import IBMQ only if needed
            try:
                from qiskit.providers.ibmq import IBMQ
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q')
                backend = provider.get_backend(self.ibmq_backend_name)
                transpiled = transpile(qc, backend)
                job = backend.run(transpiled, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                p0 = counts.get('0', 0) / self.shots
            except Exception as e:
                print(f"Error running on IBMQ backend or importing IBMQ: {e}. Falling back to AerSimulator.")
                sim = AerSimulator()
                job = sim.run(qc, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                p0 = counts.get('0', 0) / self.shots
        else:
            sim = AerSimulator()
            job = sim.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            p0 = counts.get('0', 0) / self.shots
        fidelity = 2 * p0 - 1  # Estimate of |<psi|phi>|^2
        distance = 1 - fidelity
        return distance

    def _assign_clusters(self, X, centroids):
        labels = np.zeros(len(X))
        for i, point in enumerate(X):
            distances = []
            for centroid in centroids:
                dist = self._swap_test_fidelity(point, centroid)
                distances.append(dist)
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = self.cluster_centers_[k] if self.cluster_centers_ is not None else np.random.rand(X.shape[1])
        return centroids

    def fit_predict(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        X_normalized = normalize(X, axis=1)
        centroids = self._initialize_centroids(X_normalized)
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X_normalized, centroids)
            new_centroids = self._update_centroids(X_normalized, labels)
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                print(f"Converged after {iteration + 1} iterations")
                break
            centroids = new_centroids
        self.cluster_centers_ = centroids * self.std_ + self.mean_
        self.labels_ = labels
        return labels.astype(int)

    def get_quantum_circuit_example(self, x1, x2):
        x1_norm = x1 / (np.linalg.norm(x1) + 1e-8)
        x2_norm = x2 / (np.linalg.norm(x2) + 1e-8)
        qc = QuantumCircuit(1 + 2*self.n_qubits, 1)
        qc.h(0)
        for i in range(self.n_qubits):
            qc.ry(x1_norm[i], i+1)
        for i in range(self.n_qubits):
            qc.ry(x2_norm[i], i+1+self.n_qubits)
        for i in range(self.n_qubits):
            qc.cswap(0, i+1, i+1+self.n_qubits)
        qc.h(0)
        qc.measure(0, 0)
        return qc

# Use the same data as for classical clustering (X_main)
print("=== Improved Quantum K-Means Clustering ===")
print(f"Dataset shape: {X_main.shape}")

# Find the best number of clusters using silhouette score
k_range = range(2, 8)  # Reduced range for computational efficiency
silhouette_scores_qk = []

print("\nTesting different numbers of clusters...")
for k in k_range:
    print(f"Testing k={k}...")
    q_kmeans = QuantumKMeansFidelity(n_clusters=k, n_qubits=min(4, X_main.shape[1]), max_iter=20, random_state=42)
    labels = q_kmeans.fit_predict(X_main)
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_main, labels)
        silhouette_scores_qk.append(score)
        print(f"k={k}, Silhouette score: {score:.4f}")
    else:
        silhouette_scores_qk.append(-1)
        print(f"k={k}, Invalid clustering (all points in one cluster)")

# Find the best k for Quantum KMeans
best_k_qk = k_range[np.argmax(silhouette_scores_qk)]
print(f"\nBest number of clusters (k) according to silhouette score (Quantum KMeans): {best_k_qk}")

# Fit Quantum KMeans with the best number of clusters
print(f"\nFitting Improved Quantum K-Means with k={best_k_qk}...")
q_kmeans_main = QuantumKMeansFidelity(n_clusters=best_k_qk, n_qubits=min(4, X_main.shape[1]), max_iter=30, random_state=42)
y_q_kmeans = q_kmeans_main.fit_predict(X_main)
# Only calculate silhouette score if there is more than one cluster
if len(np.unique(y_q_kmeans)) > 1:
    sil_score_q_kmeans = np.round(silhouette_score(X_main, y_q_kmeans), 4)
else:
    sil_score_q_kmeans = float('nan')
    print("Warning: Quantum KMeans produced only one cluster. Silhouette score is undefined.")
print(f"Silhouette coefficient for the best Quantum K-Means model: {sil_score_q_kmeans}")

# ===============================
# Compare silhouette scores for all clustering methods
# ===============================

silhouette_scores_compare = {
    'KMeans': sil_score_kmeans,
    'Agglomerative': sil_score_hc,
    'Quantum KMeans': sil_score_q_kmeans
}

plt.figure(figsize=(7, 6))
sns.barplot(x=list(silhouette_scores_compare.keys()), y=list(silhouette_scores_compare.values()), palette='viridis')
for i, (model, score) in enumerate(silhouette_scores_compare.items()):
    plt.text(i, score + 0.01, f"{score:.2f}", ha='center', fontsize=13, weight='bold')
plt.ylabel('Silhouette Score')
plt.xlabel('Clustering Model')
plt.title('Comparison of Silhouette Scores for Clustering Models')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('Silhouette_Score_Comparison_All.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# Plot clusters for all methods with centroids
# ===============================

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# KMeans
axes[0].set_title(f'KMeans Clusters (k={best_k})')
for i in range(best_k):
    cluster_points = X_main[y_kmeans == i]
    axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.7, s=50)
centers = kmeans_main.cluster_centers_
axes[0].scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=200, linewidths=3, label='Centroids')
axes[0].set_xlabel('PC1', labelpad=15)
axes[0].set_ylabel('PC2', labelpad=15)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Agglomerative
axes[1].set_title(f'Agglomerative Clusters (k={optimal_clusters})')
for i in range(optimal_clusters):
    cluster_points = X_main[y_hc == i]
    axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.7, s=50)
agg_centers = np.array([X_main[y_hc == i, :2].mean(axis=0) for i in range(optimal_clusters)])
axes[1].scatter(agg_centers[:, 0], agg_centers[:, 1], c='black', marker='*', s=200, linewidths=3, label='Centroids')
axes[1].set_xlabel('PC1', labelpad=15)
axes[1].set_ylabel('PC2', labelpad=15)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Quantum KMeans
axes[2].set_title(f'Quantum KMeans Clusters (k={best_k_qk})')
for i in range(best_k_qk):
    cluster_points = X_main[y_q_kmeans == i]
    axes[2].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.7, s=50)
centroids = q_kmeans_main.cluster_centers_
axes[2].scatter(centroids[:, 0], centroids[:, 1], c='black', marker='*', s=200, linewidths=3, label='Centroids')
axes[2].set_xlabel('PC1', labelpad=15)
axes[2].set_ylabel('PC2', labelpad=15)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Cluster Assignments for KMeans, Agglomerative, and Quantum KMeans', fontsize=18, weight='bold', y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Cluster_Comparison_All.png', dpi=300, bbox_inches='tight')
plt.show()

# Show example quantum circuit for distance calculation
print("\n=== Example Quantum Circuit for Distance Calculation ===")
example_circuit = q_kmeans_main.get_quantum_circuit_example(X_main[0], X_main[1])
plt.figure(figsize=(14, 8))
circuit_drawing = example_circuit.draw(output='mpl', style='iqx', fold=20)
plt.title('Quantum Circuit for Distance Calculation Between Two Data Points', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('Quantum_Distance_Circuit.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nQuantum circuit details:")
print(f"- Number of qubits: {example_circuit.num_qubits}")
print(f"- Number of gates: {len(example_circuit.data)}")
print(f"- Circuit depth: {example_circuit.depth()}")

print("\n" + "="*50)
print("QUANTUM K-MEANS SUMMARY")
print("="*50)
print(f"Best k: {best_k_qk}")
print(f"Silhouette Score: {sil_score_q_kmeans}")
print(f"Number of data points: {len(X_main)}")
print(f"Number of features: {X_main.shape[1]}")
print(f"Quantum circuit qubits used: {q_kmeans_main.n_qubits}")
print("="*50)

# ===============================
# Improved Classical and Quantum SVMs with Decision Boundaries
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# 1. Select the best clustering label
scores = {
    'KMeans': sil_score_kmeans,
    'Agglomerative': sil_score_hc,
    'Quantum KMeans': sil_score_q_kmeans
}
best_label_name = max(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else -np.inf)
print(f"Best clustering label: {best_label_name}")

if best_label_name == 'KMeans':
    y = y_kmeans
elif best_label_name == 'Agglomerative':
    y = y_hc
else:
    y = y_q_kmeans

X = X_main  # Features: PCs explaining 95% variance

# 2. Train/test split (filter classes with only one sample)
unique, counts = np.unique(y, return_counts=True)
valid_classes = unique[counts > 1]
mask = np.isin(y, valid_classes)
X_filtered = X[mask]
y_filtered = y[mask]

# Use all PCs that explain at least 95% of the variance for quantum SVM and visualization
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
)

# Normalize features to [0, pi] for quantum circuit
def normalize_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8) * np.pi

X_train_norm = normalize_features(X_train)
X_test_norm = normalize_features(X_test)

# 3. Classical SVM (RBF) for all PCs
svc_rbf = SVC(kernel='rbf', random_state=42)
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
disp_rbf = ConfusionMatrixDisplay(confusion_matrix=cm_rbf)
disp_rbf.plot(cmap='Greens')
plt.title('Confusion Matrix for Classical SVM (RBF Kernel)')
plt.show()

def plot_predictions(X, y_true, y_pred, title):
    """
    Plot PC1 vs PC2, coloring by predicted label, and marking correct/incorrect predictions.
    Circles: correct, X: incorrect.
    """
    X_vis = X[:, :2]
    correct = y_true == y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[correct, 0], X_vis[correct, 1], c=y_pred[correct], cmap='tab10', marker='o', edgecolor='k', s=100, label='Correct')
    plt.scatter(X_vis[~correct, 0], X_vis[~correct, 1], c=y_pred[~correct], cmap='tab10', marker='X', edgecolor='red', s=120, label='Incorrect')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_predictions(X_test[:, :2], y_test, y_pred_rbf, 'Classical SVM (RBF): PC1 vs PC2 (Correct vs Incorrect)')

# 4. Quantum SVM with PennyLane (ZZFeatureMap kernel) using all PCs
n_qubits = X_train.shape[1]
shots = 1024

dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

@qml.qnode(dev)
def zz_kernel_circuit(x1, x2):
    # ZZFeatureMap style
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(x1[i], wires=i)
    # Entanglement (ring)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    # Optionally add more entanglement layers here
    for i in range(n_qubits):
        qml.RZ(-x2[i], wires=i)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(X1, X2, verbose=False):
    """
    Compute the quantum kernel matrix between X1 and X2 with nested progress bars.
    Shows progress for both rows and columns, and prints percentage completed.
    """
    kernel_matrix = np.zeros((len(X1), len(X2)))
    total = len(X1) * len(X2)
    completed = 0
    outer = tqdm(enumerate(X1), total=len(X1), desc='Quantum Kernel (rows)', disable=not verbose)
    for i, x1 in outer:
        inner = tqdm(enumerate(X2), total=len(X2), desc=f'Quantum Kernel (cols) row {i+1}/{len(X1)}', leave=False, disable=not verbose)
        for j, x2 in inner:
            prob = zz_kernel_circuit(x1, x2)
            kernel_matrix[i, j] = np.abs(prob[0])
            completed += 1
            if verbose and completed % max(1, total // 20) == 0:
                percent = 100 * completed / total
                print(f"Quantum kernel computation: {percent:.1f}% done")
    return kernel_matrix

print('Computing quantum kernel for training...')
K_train = quantum_kernel(X_train_norm, X_train_norm, verbose=True)
print('Computing quantum kernel for testing...')
K_test = quantum_kernel(X_test_norm, X_train_norm, verbose=True)
print('pre-computed')
svc_quantum = SVC(kernel='precomputed')
print('Entra')
svc_quantum.fit(K_train, y_train)
print('Sal')
y_pred_quantum = svc_quantum.predict(K_test)
cm_quantum = confusion_matrix(y_test, y_pred_quantum)
disp_quantum = ConfusionMatrixDisplay(confusion_matrix=cm_quantum)
disp_quantum.plot(cmap='Blues')
plt.title('Confusion Matrix for Quantum SVM')
plt.show()

plot_predictions(X_test[:, :2], y_test, y_pred_quantum, 'Quantum SVM: PC1 vs PC2 (Correct vs Incorrect)')

# Draw the PennyLane quantum kernel circuit for the first training sample
fig, ax = plt.subplots(figsize=(10, 6))
drawer = qml.draw_mpl(zz_kernel_circuit, decimals=2)
drawer(X_train_norm[0], X_train_norm[0])
plt.title('Quantum Kernel Circuit (first training sample)')
plt.tight_layout()
plt.show()

# Decision boundary plotting function (for 2D visualization, use first two PCs)

def plot_decision_boundary(model, X, y, title, is_quantum=False, X_train_kernel=None):
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if is_quantum:
        # Normalize grid for quantum kernel
        grid_norm = normalize_features(grid)
        K_grid = quantum_kernel(grid_norm, X_train_kernel)
        Z = model.predict(K_grid)
    else:
        Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolor='k', s=80)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title='Class')
    plt.tight_layout()
    plt.show()

# Plot decision boundaries (for 2D visualization, use first two PCs)
plot_decision_boundary(svc_rbf, X_test[:, :2], y_test, 'Classical SVM (RBF) Decision Boundary')
plot_decision_boundary(svc_quantum, X_test_norm[:, :2], y_test, 'Quantum SVM (ZZFeatureMap) Decision Boundary', is_quantum=True, X_train_kernel=X_train_norm)
