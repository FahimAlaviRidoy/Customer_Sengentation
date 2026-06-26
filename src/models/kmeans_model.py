import mlflow
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_kmeans(X, k_range=range(2, 11)):
    best_score = -1
    best_k = 2
    print("  Finding optimal K for K-Means...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        mlflow.log_metric(f"kmeans_silhouette_k{k}", score)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"  Best K for K-Means: {best_k}")
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(X)
    
    mlflow.log_param("kmeans_best_k", best_k)
    mlflow.log_metric("kmeans_silhouette", best_score)
    mlflow.log_metric("kmeans_davies_bouldin", davies_bouldin_score(X, labels))
    return final_model, labels
