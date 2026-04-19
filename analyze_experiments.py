import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np


# -----------------------------
# Filename parsing
# -----------------------------
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


# -----------------------------
# Main analysis
# -----------------------------
def analyze_flood_results(csv_path):

    # Load
    df = pd.read_csv(csv_path)

    # Robust cleaning (removes duplicate headers etc.)
    df = df[pd.to_numeric(df['runtime'], errors='coerce').notna()]

    # -----------------------------
    # Metadata FIRST (fix)
    # -----------------------------
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

    # -----------------------------
    # Normalize precision loss
    # -----------------------------
    df['rel_precision_loss'] = df['precision_loss'] / df['total_water']

    # -----------------------------
    # Aggregate (scenario removed!)
    # -----------------------------
    config_cols = ['rows', 'clouds', 'ex_factor']

    avg_df = (
        df.groupby(['binary'] + config_cols)
        .agg(
            runtime_mean=('runtime', 'mean'),
            runtime_std=('runtime', 'std')
        )
        .reset_index()
    )

    # -----------------------------
    # Speedup computation (fixed)
    # -----------------------------
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

    # -----------------------------
    # Efficiency
    # -----------------------------
    avg_df['cloud_efficiency'] = avg_df['speedup'] / avg_df['clouds']
    avg_df['grid_efficiency'] = avg_df['speedup'] / avg_df['rows']

    # -----------------------------
    # Plot style
    # -----------------------------
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update({
        'figure.figsize': (4.5, 3),
        'axes.titlesize': 11,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
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

    # Ideal scaling
    # clouds = sorted(scaling['clouds'].unique())
    # ax.plot(clouds, clouds, '--', color='black', label='Ideal')

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

    # Compute speedup vs seq for both CUDA and CUDA SoA
    comp['speedup_cuda'] = comp['flood_seq'] / comp['flood_cuda']
    comp['speedup_cuda_soa'] = comp['flood_seq'] / comp['flood_cuda_soa']

    # Compute the ratio of SoA speedup to AoS speedup
    comp['soa_vs_aos_speedup'] = comp['speedup_cuda_soa'] / comp['speedup_cuda']

    # Pivot for heatmap: rows vs clouds, value = soa_vs_aos_speedup
    pivot = comp.pivot_table(
        index='rows',
        columns='clouds',
        values='soa_vs_aos_speedup'
    ).sort_index().sort_index(axis=1)

    if not pivot.empty and np.isfinite(pivot.values).any():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=1.0,
            cbar_kws={'label': 'SoA/AoS Speedup Ratio'},
            ax=ax
        )
        ax.set_xlabel('Clouds')
        ax.set_ylabel('Grid Size (Rows)')
        fig.tight_layout()
        fig.savefig('fig_soa_vs_aos_speedup_heatmap.pdf', bbox_inches='tight')
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
        hue_order=['flood_seq', 'flood_cuda', 'flood_cuda_soa'],
        style='binary',
        style_order=['flood_seq', 'flood_cuda', 'flood_cuda_soa'],
        alpha=0.6,
        ax=ax
    )

    #ax.set_title('Relative Precision Loss vs Rain')
    ax.set_xlabel('Total Rain')
    ax.set_xscale('log')
    ax.set_ylabel('Relative Precision Loss')
    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_precision_loss.pdf')
    plt.close(fig)

    # --- Relative Precision Loss Difference Heatmap ---
    # Compute rel_precision_loss for each implementation (use df, not avg_df)
    prec_comp = df.pivot_table(
        index=['rows', 'clouds'],
        columns='binary',
        values='rel_precision_loss',
        aggfunc='mean'
    ).reset_index()

    # Compute relative difference from flood_seq (normalized by flood_seq)
    for col in ['flood_cuda', 'flood_cuda_soa']:
        prec_comp[f'rel_diff_{col}'] = (prec_comp[col] - prec_comp['flood_seq']) / prec_comp['flood_seq']

    # Heatmap for CUDA
    pivot_prec_cuda = prec_comp.pivot_table(
        index='rows',
        columns='clouds',
        values='rel_diff_flood_cuda'
    ).sort_index().sort_index(axis=1)
    if not pivot_prec_cuda.empty and np.isfinite(pivot_prec_cuda.values).any():
        fig, ax = plt.subplots(figsize=(6, 4))
        # Set vmin/vmax symmetric around zero for centered colorbar
        absmax = np.nanmax(np.abs(pivot_prec_cuda.values))
        sns.heatmap(
            pivot_prec_cuda,
            annot=True,
            annot_kws={'size': 8},
            fmt='.2e',
            cmap='coolwarm',
            center=0.0,
            vmin=-absmax,
            vmax=absmax,
            cbar_kws={'label': 'Rel. Precision Loss Diff (CUDA vs Seq)'},
            ax=ax
        )
        ax.set_xlabel('Clouds')
        ax.set_ylabel('Grid Size (Rows)')
        fig.tight_layout()
        fig.savefig('fig_rel_precision_loss_diff_cuda_heatmap.pdf', bbox_inches='tight')
        plt.close(fig)

    # Heatmap for CUDA SoA
    pivot_prec_soa = prec_comp.pivot_table(
        index='rows',
        columns='clouds',
        values='rel_diff_flood_cuda_soa'
    ).sort_index().sort_index(axis=1)
    if not pivot_prec_soa.empty and np.isfinite(pivot_prec_soa.values).any():
        fig, ax = plt.subplots(figsize=(6, 4))
        absmax = np.nanmax(np.abs(pivot_prec_soa.values))
        sns.heatmap(
            pivot_prec_soa,
            annot=True,
            annot_kws={'size': 8},
            fmt='.2e',
            cmap='coolwarm',
            center=0.0,
            vmin=-absmax,
            vmax=absmax,
            cbar_kws={'label': 'Rel. Precision Loss Diff (CUDA SoA vs Seq)'},
            ax=ax,
        )
        ax.set_xlabel('Clouds')
        ax.set_ylabel('Grid Size (Rows)')
        fig.tight_layout()
        fig.savefig('fig_rel_precision_loss_diff_cuda_soa_heatmap.pdf', bbox_inches='tight')
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
    # -----------------------------
    # FIGURE 11: Scenario Impact (Normalized Runtime)
    # -----------------------------
    # Normalize runtime per configuration to isolate scenario effect

    scenario_df = df.copy()

    # Define configuration (excluding scenario)
    config_cols = ['binary', 'rows', 'clouds', 'ex_factor']

    # Compute mean runtime per config
    scenario_df['config_mean_runtime'] = (
        scenario_df.groupby(config_cols)['runtime']
        .transform('mean')
    )

    # Normalize
    scenario_df['normalized_runtime'] = (
        scenario_df['runtime'] / scenario_df['config_mean_runtime']
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        data=scenario_df,
        x='scenario',
        y='normalized_runtime',
        hue='binary',
        ax=ax
    )

    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Normalized Runtime (relative to config mean)')
    # ax.set_title('Scenario Impact on Runtime (Normalized)')

    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_scenario_normalized_runtime.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 12: Scenario Impact on Speedup
    # -----------------------------
    # Compute speedup per individual run first

    speedup_df = df.copy()

    # Get sequential runtime per config
    seq_runtime = (
        speedup_df[speedup_df['binary'] == 'flood_seq']
        .groupby(['rows', 'clouds', 'ex_factor'])['runtime']
        .mean()
    )

    def compute_row_speedup(row):
        key = (row['rows'], row['clouds'], row['ex_factor'])
        if row['binary'] != 'flood_seq' and key in seq_runtime.index:
            return seq_runtime.loc[key] / row['runtime']
        return np.nan

    speedup_df['speedup'] = speedup_df.apply(compute_row_speedup, axis=1)

    fig, ax = plt.subplots()

    sns.boxplot(
        data=speedup_df[speedup_df['binary'] != 'flood_seq'],
        x='scenario',
        y='speedup',
        hue='binary',
        ax=ax
    )

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Speedup vs Seq')
    # ax.set_title('Scenario Impact on Speedup')

    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_scenario_speedup.pdf')
    plt.close(fig)

    # -----------------------------
    # FIGURE 13: Scenario Impact (Violin Plot)
    # -----------------------------
    fig, ax = plt.subplots()

    sns.violinplot(
        data=scenario_df,
        x='scenario',
        y='normalized_runtime',
        hue='binary',
        hue_order=['flood_seq', 'flood_cuda', 'flood_cuda_soa'],
        split=False,
        inner='quartile',
        cut=0,
        scale='width',
        ax=ax
    )

    # Reference line at perfect agreement
    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Normalized Runtime (relative to config mean)')
    # ax.set_title('Scenario Impact on Runtime (Distribution View)')

    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_scenario_violin_runtime.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()

    sns.violinplot(
        data=speedup_df[speedup_df['binary'] != 'flood_seq'],
        x='scenario',
        y='speedup',
        hue='binary',
        hue_order=['flood_cuda', 'flood_cuda_soa'],
        split=False,
        inner='quartile',
        cut=0,
        scale='width',
        ax=ax
    )

    # Reference line at perfect agreement
    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Speedup vs Seq')
    # ax.set_title('Scenario Impact on Speedup (Distribution View)')

    ax.legend(title='Implementation')

    fig.tight_layout()
    fig.savefig('fig_scenario_violin_speedup.pdf')
    plt.close(fig)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    analyze_flood_results('_logs/experiment_results.csv')