import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_json(data, file_path, indent=4):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def heatmap(data, title):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_title(title)
    sns.heatmap(data, annot=True, ax=ax)
    fig.tight_layout()
    return fig

def get_time_str() -> str:
    return datetime.now().strftime("%Y-%b-%d-%H-%M-%S")