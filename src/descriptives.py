import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_missing_data_heatmap(data, rotate_xticks=45, directory: str = "assets/plots/descriptives"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    missing_data = data.isna()
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(missing_data, cbar=False, cmap='Greys', yticklabels=False)

    ax.set_title('')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Index')

    plt.xticks(rotation=rotate_xticks)
    plt.savefig(os.path.join(directory, "missings.png"))
    plt.show()
