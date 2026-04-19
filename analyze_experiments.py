import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np


# Filename parsing
def parse_filename(filename):
    pattern = r"exp_(?P<rows>\d+)x(?P<cols>\d+)_(?P<scenario>\w)_c(?P<clouds>\d+)_ex(?P<ex>\d+)_t(?P<thresh>[^ _]+)_m(?P<mins>\d+)"
    match = re.search(pattern, filename)
    if match:
        d = match.groupdict()
        return pd.Series({
            'rows': int(d['rows']),
            'cols': int(d['cols']),
            'total_cells': int(d['rows']) * int(d['cols']),
            'scenario': d['scenario'],
            'clouds': int(d['clouds']),
            'ex_factor': int(d['ex']),
            'mins': int(d['mins'])
        })
    return pd.Series()

def analyze_flood_results(csv_path):

    # Load
    df = pd.read_csv(csv_path)

    df = df[pd.to_numeric(df['runtime'], errors='coerce').notna()]

    metadata = df['input_file'].apply(parse_filename)
    df = pd.concat([df, metadata], axis=1)

    # Convert numerics
    numeric_cols = [
        'runtime', 'precision_loss', 'total_rain',
        'total_water', 'total_water_loss',
        'rows', 'cols', 'total_cells',
        'clouds', 'ex_factor', 'mins'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize precision loss
    df['rel_precision_loss'] = df['precision_loss'] / df['total_water']

    # Aggregate results
    config_cols = ['rows', 'clouds', 'ex_factor']

    avg_df = (
        df.groupby(['binary'] + config_cols)
        .agg(
            runtime_mean=('runtime', 'mean'),
            runtime_std=('runtime', 'std')
        )
        .reset_index()
    )

    # Speedup computation
    seq_runs = (
        avg_df[avg_df['binary'] == 'flood_seq']
        .set_index(config_cols)['runtime_mean']
    )

    def compute_speedup(row):
        key = tuple(row[c] for c in config_cols)
        if row['binary'] != 'flood_seq' and key in seq_runs.index:
            return seq_runs.loc[key] / row['runtime_mean']
        return np.nan

    avg_df['speedup'] = avg_df.apply(compute_speedup, axis=1)

    # Efficiency
    avg_df['cloud_efficiency'] = avg_df['speedup'] / avg_df['clouds']
    avg_df['grid_efficiency'] = avg_df['speedup'] / avg_df['rows']

    # Plot style
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update({
        'figure.figsize': (5.5, 4),
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        "legend.title_fontsize": 9,
    })

    # -----------------------------
    # FIGURE 1: Speedup vs Clouds
    # -----------------------------
    scaling = avg_df[avg_df['binary'] != 'flood_seq']

    fig, ax = plt.subplots()

    sns.lineplot(
        data=scaling,
        x='clouds', y='speedup',
        hue='binary',
        errorbar='sd',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log', base=2)
    #ax.set_title('Speedup vs Clouds')
    ax.set_ylabel('Speedup vs Seq')
    ax.set_xlabel('Clouds (log2)')
    ax.legend()

    fig.tight_layout()
    fig.savefig('fig_speedup_vs_clouds.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 2: Speedup vs Grid Size
    # -----------------------------
    fig, ax = plt.subplots()

    sns.lineplot(
        data=scaling,
        x='rows', y='speedup',
        hue='binary',
        errorbar='sd',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log', base=2)
    #ax.set_title('Speedup vs Grid Size')
    ax.set_xlabel('Grid Size (log2)')
    ax.set_ylabel('Speedup vs Seq')
    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_speedup_vs_grid.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 3: Efficiency
    # -----------------------------
    fig, ax = plt.subplots()

    sns.lineplot(
        data=scaling,
        x='clouds', y='cloud_efficiency',
        hue='binary',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log', base=2)
    #ax.set_title('Parallel Efficiency')
    ax.set_ylabel('Efficiency (Speedup / Clouds)')
    ax.set_xlabel('Clouds')
    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_cloud_efficiency.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()

    sns.lineplot(
        data=scaling,
        x='rows', y='grid_efficiency',
        hue='binary',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log', base=2)
    #ax.set_title('Parallel Efficiency')
    ax.set_ylabel('Efficiency (Speedup / Grid Size)')
    ax.set_xlabel('Grid Size')
    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_grid_efficiency.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 4: AoS vs SoA comparison
    # -----------------------------
    comp = avg_df.pivot_table(
        index=['rows', 'clouds'],
        columns='binary',
        values='runtime_mean'
    ).reset_index()

    comp['soa_vs_aos'] = comp['flood_cuda'] / comp['flood_cuda_soa']

    fig, ax = plt.subplots()

    sns.lineplot(
        data=comp,
        x='clouds', y='soa_vs_aos',
        hue='rows',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log', base=2)
    #ax.set_title('SoA vs AoS ( >1 = SoA faster )')
    ax.set_ylabel('AoS / SoA Runtime')
    ax.set_xlabel('Clouds')

    fig.tight_layout()
    fig.savefig('fig_soa_vs_aos.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 5: Runtime scaling (log-log)
    # -----------------------------
    fig, ax = plt.subplots()

    sns.lineplot(
        data=df,
        x='total_cells', y='runtime',
        hue='binary',
        marker='o',
        ax=ax
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    #ax.set_title('Runtime Scaling (log-log)')
    ax.set_xlabel('Total Cells')
    ax.set_ylabel('Runtime (Seconds)')

    fig.tight_layout()
    fig.savefig('fig_runtime_scaling.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 6: Precision loss (normalized)
    # -----------------------------
    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x='total_rain',
        y='rel_precision_loss',
        hue='binary',
        alpha=0.5,
        ax=ax
    )

    #ax.set_title('Relative Precision Loss vs Rain')
    ax.set_xlabel('Total Rain')
    ax.set_ylabel('Relative Precision Loss')

    fig.tight_layout()
    fig.savefig('fig_precision_loss.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 7: Runtime distribution
    # -----------------------------
    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x='binary',
        y='runtime',
        ax=ax
    )

    #ax.set_title('Runtime Distribution')
    ax.set_xlabel('Implementation')
    ax.set_ylabel('Runtime (Seconds)')

    fig.tight_layout()
    fig.savefig('fig_runtime_distribution.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 8: Correlation heatmap
    # -----------------------------

    corr_df = df.select_dtypes(include=[np.number]).drop(columns=['ex_factor', 'mins'], errors='ignore')
    corr_pearson = corr_df.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_pearson, cmap='coolwarm', annot=False, ax=ax)

    #ax.set_title('Correlation Heatmap')

    fig.tight_layout()
    fig.savefig('fig_correlation_pearson.pdf')
    plt.close(fig)

    corr_spearman = corr_df.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_spearman, cmap='coolwarm', annot=False, ax=ax)

    #ax.set_title('Correlation Heatmap')

    fig.tight_layout()
    fig.savefig('fig_correlation_spearman.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 9: Heatmaps (Grid Size vs Clouds)
    # -----------------------------

    binaries = ['flood_seq', 'flood_cuda', 'flood_cuda_soa']
    metrics = ['runtime_mean', 'speedup']

    for binary in binaries:
        for metric in metrics:

            # Skip invalid combination
            if binary == 'flood_seq' and metric == 'speedup':
                continue

            subset = avg_df[avg_df['binary'] == binary]

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index='rows',
                columns='clouds',
                values=metric,
                aggfunc='mean'
            )

            # Ensure sorted axes (important for readability)
            pivot = pivot.sort_index().sort_index(axis=1)

            # Skip empty / all-NaN
            if pivot.empty or not np.isfinite(pivot.values).any():
                continue

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.heatmap(
                pivot,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                cbar_kws={'label': metric.replace('_', ' ').capitalize()},
                ax=ax
            )

            # ax.set_title(
            #     f"{metric.replace('_', ' ').capitalize()} Heatmap\n"
            #     f"{binary} (Rows = Cols)"
            # )
            ax.set_xlabel('Number of Clouds')
            ax.set_ylabel('Grid Size')

            fig.tight_layout()

            fig.savefig(
                f'fig_heatmap_{metric}_{binary}.pdf',
                bbox_inches='tight'
            )

            plt.close(fig)


    # -----------------------------
    # FIGURE 10: SoA vs AoS Difference Heatmap
    # -----------------------------
    comp = avg_df.pivot_table(
        index=['rows', 'clouds'],
        columns='binary',
        values='runtime_mean'
    ).reset_index()

    if 'flood_cuda' in comp and 'flood_cuda_soa' in comp:
        comp['diff'] = comp['flood_cuda'] - comp['flood_cuda_soa']

        pivot = comp.pivot_table(
            index='rows',
            columns='clouds',
            values='diff'
        ).sort_index().sort_index(axis=1)

        if not pivot.empty and np.isfinite(pivot.values).any():

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.heatmap(
                pivot,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Runtime Difference (AoS - SoA)'},
                ax=ax
            )

            #ax.set_title('SoA vs AoS Runtime Difference\n(Positive = SoA Faster)')
            ax.set_xlabel('Clouds')
            ax.set_ylabel('Grid Size')

            fig.tight_layout()
            fig.savefig('fig_heatmap_soa_vs_aos_diff.pdf', bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    analyze_flood_results('_logs/experiment_results.csv')