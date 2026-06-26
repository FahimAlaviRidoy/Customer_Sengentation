import mlflow
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_agglomerative(X, k_range=range(2, 11)):
    best_score = -1
    best_k = 2
    print("  Finding optimal K for Agglomerative...")
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        score = silhouette_score(X, labels)
        mlflow.log_metric(f"agg_silhouette_k{k}", score)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"  Best K for Agglomerative: {best_k}")
    final_model = AgglomerativeClustering(n_clusters=best_k)
    labels = final_model.fit_predict(X)
    
    mlflow.log_param("agg_best_k", best_k)
    mlflow.log_metric("agg_silhouette", best_score)
    mlflow.log_metric("agg_davies_bouldin", davies_bouldin_score(X, labels))
    return final_model, labels
