import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

def get_dataset_paths(dataset):
    paths = {}
    if dataset == "Default":
        paths["Regular"] = "data/test_dataset_smaller_ims/"
        paths["Updated"] = "data/test_dataset_updated/"
    elif dataset == "Morning":
        paths["Test"] = "data/morning/morning_test/"
        paths["Train"] = "data/morning/morning_train/"
        paths["Val"] = "data/morning/morning_validation/"
    elif dataset == "Afternoon":
        paths["Test"] = "data/afternoon/afternoon_test/"
        paths["Train"] = "data/afternoon/afternoon_train/"
        paths["Val"] = "data/afternoon/afternoon_validation/"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return paths

def main(dataset="Default"):
    dataset_paths = get_dataset_paths(dataset)
    cols = ['distance_to_centerline_meters', 'downtrack_position_meters', 'heading_error_degrees']
    all_dfs = []
    n_datasets = len(dataset_paths) + 1

    # Use Pastel1 colormap
    pastel_cmap = cm.get_cmap('Pastel1', 9) # Using 9 allows the first three colors to be RBG
    row_colors = [pastel_cmap(i) for i in range(n_datasets)]

    fig, axs = plt.subplots(n_datasets, len(cols), figsize=(6 * len(cols), 3.5 * n_datasets), squeeze=False)
    # Add overall title
    fig.suptitle(f"{dataset} Dataset", fontsize=18)

    # Add pretty column headings (styled like row labels)
    for col_idx, col in enumerate(cols):
        ax = axs[0, col_idx]
        bbox = dict(boxstyle="round,pad=0.4", facecolor="lightgray", edgecolor="gray", alpha=0.8)
        fig.text(
            ax.get_position().x0 + ax.get_position().width / 2,
            0.9,  # near top of figure
            col,
            ha='center', va='bottom', fontsize=13, bbox=bbox
        )

    # Plot histograms for each dataset
    for row_idx, (label, path) in enumerate(dataset_paths.items()):
        csv_path = os.path.join(path, "labels.csv")
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

        for col_idx, col in enumerate(cols):
            axs[row_idx, col_idx].hist(df[col], bins=50, color=row_colors[row_idx], edgecolor='black')
            axs[row_idx, col_idx].set_facecolor('white')

        # Row label
        fig.text(
            0.035,
            axs[row_idx, 0].get_position().y0 + axs[row_idx, 0].get_position().height / 2,
            label,
            ha='left', va='center', rotation=90,
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=row_colors[row_idx], alpha=0.5)
        )

    # Combined row
    combined_df = pd.concat(all_dfs, ignore_index=True)
    row_idx = n_datasets - 1
    for col_idx, col in enumerate(cols):
        axs[row_idx, col_idx].hist(combined_df[col], bins=50, color=row_colors[row_idx], edgecolor='black')
        axs[row_idx, col_idx].set_facecolor('white')

    fig.text(
        0.035,
        axs[row_idx, 0].get_position().y0 + axs[row_idx, 0].get_position().height / 2,
        "Combined",
        ha='left', va='center', rotation=90,
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=row_colors[row_idx], alpha=0.5)
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    plt.show()

if __name__ == "__main__":
    main("Afternoon")
