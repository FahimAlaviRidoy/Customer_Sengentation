import matplotlib.pyplot as plt
import seaborn as sns
from src.config import ARTIFACTS_DIR

def perform_eda(df):
    print("  Generating distribution plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(['recency', 'frequency', 'monetary']):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col.capitalize()}')
    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_DIR}/distributions.png")
    plt.close()
    
    print("  Generating correlation heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['recency', 'frequency', 'monetary']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('RFM Correlation Heatmap')
    plt.savefig(f"{ARTIFACTS_DIR}/correlation.png")
    plt.close()
