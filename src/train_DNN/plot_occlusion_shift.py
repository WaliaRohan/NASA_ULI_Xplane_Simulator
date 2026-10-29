"""
    Plots how occlusion changes the DNN's (CTE, DTP) predictions and their errors,
    using the CSV saved by compare_occlusion.py / predict_occluded_DNN.py
"""

import os
import pandas
import matplotlib.pyplot as plt

# Config
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'
dataset_type = 'large_images' # large_images or downsampled
condition = 'afternoon' # 'afternoon', 'morning', 'night', 'overcast'
save_plot = True # also save the figure as a PNG in the results dir

# colors
CLEAN_COLOR = '#2a78d6'
OCC_COLOR = '#eb6834'
TRUE_COLOR = '#1baf7a'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e4e3df'


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_legend(ax, ncol):
    # legend sits above the axes so it never covers data
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.06), ncol=ncol, borderaxespad=0,
              handletextpad=0.2, labelcolor=TEXT_PRIMARY, fontsize=9)


def scatter(ax, x, y, color, label):
    ax.scatter(x, y, s=14, color=color, alpha=0.7, edgecolors='white', linewidths=0.4, label=label, zorder=3)


if __name__ == '__main__':

    results_dir = SCRATCH_DIR + '/test_DNN_taxinet_occlusion/'
    df = pandas.read_csv(results_dir + condition + '_' + dataset_type + '_occlusion_predictions.csv')

    fig, (ax_points, ax_cte_err, ax_dtp_err) = plt.subplots(1, 3, figsize=(19, 5.5))

    # (a) true labels, clean predictions and occluded predictions as separate point clouds
    scatter(ax_points, df['true_cte_norm'], df['true_dtp_norm'], TRUE_COLOR, 'True label')
    scatter(ax_points, df['pred_cte_norm'], df['pred_dtp_norm'], CLEAN_COLOR, 'Clean prediction')
    scatter(ax_points, df['pred_cte_norm_occ'], df['pred_dtp_norm_occ'], OCC_COLOR, 'Occluded prediction')
    style_axes(ax_points, '(a) True vs. predicted positions',
               'Cross-track error (normalized)', 'Downtrack position (normalized)')
    add_legend(ax_points, ncol=3)

    # (b), (c) absolute prediction error against the ground-truth value of the same output
    for ax, output, name, letter in [(ax_cte_err, 'cte', 'CTE', 'b'),
                                     (ax_dtp_err, 'dtp', 'DTP', 'c')]:
        true_col = 'true_' + output + '_norm'
        scatter(ax, df[true_col], (df['pred_' + output + '_norm'] - df[true_col]).abs(),
                CLEAN_COLOR, 'Clean prediction')
        scatter(ax, df[true_col], (df['pred_' + output + '_norm_occ'] - df[true_col]).abs(),
                OCC_COLOR, 'Occluded prediction')
        ax.set_ylim(bottom=0)
        style_axes(ax, f'({letter}) {name} prediction error vs. true {name}',
                   f'True {name} (normalized)', f'|predicted {name} − true {name}| (normalized)')
        add_legend(ax, ncol=2)

    fig.suptitle(f'Effect of center occlusion on TaxiNet predictions ({condition}, {len(df)} images)',
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout()

    # save before showing, since closing the window discards the figure
    if save_plot:
        out_png = results_dir + condition + '_' + dataset_type + '_occlusion_shift.png'
        fig.savefig(out_png, dpi=200, facecolor='white')
        print('Saved plot to: ', out_png)

    plt.show()
