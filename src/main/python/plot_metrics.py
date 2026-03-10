import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set Seaborn theme
sns.set_theme(style="whitegrid", font_scale=1.2)

# Read input files
metrics_file = sys.argv[1]
roc_file = sys.argv[2]

metrics = pd.read_csv(metrics_file)
roc_df = pd.read_csv(roc_file)

# Preprocess ROC labels
y_true = roc_df["TrueLabel"].replace({1: 0, 2: 1})
y_score = roc_df["PredictedScore"]
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Setup figure and axes
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

# Metric plots
metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall']
colors = sns.color_palette("Set2", 4)

for i, (metric, color) in enumerate(zip(metric_names, colors)):
    axs[i].plot(metrics['Epoch'], metrics[metric], label=metric, color=color, linewidth=2)
    axs[i].set_title(f'Training {metric}')
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)

# ROC Curve
axs[4].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axs[4].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random baseline')
axs[4].set_xlim([0.0, 1.0])
axs[4].set_ylim([0.0, 1.05])
axs[4].set_xlabel('False Positive Rate')
axs[4].set_ylabel('True Positive Rate')
axs[4].set_title('ROC Curve')
axs[4].legend(loc="lower right")
axs[4].grid(True)

# Remove the unused last subplot
fig.delaxes(axs[5])

# Final layout adjustments
plt.tight_layout()
plt.savefig("src/data/all_metrics.png", dpi=300)
print("Saving plot to 'src/data/all_metrics.png'")
plt.show()
