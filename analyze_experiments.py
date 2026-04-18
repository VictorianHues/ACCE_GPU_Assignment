import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import re
import numpy as np

def parse_filename(filename):
    """
    Parses the filename based on the generator's template:
    exp_{rows}x{cols}_{scen}_c{clouds}_ex{ex}_t{thresh}_m{mins}.in
    """
    pattern = r"exp_(?P<rows>\d+)x(?P<cols>\d+)_(?P<scenario>\w)_c(?P<clouds>\d+)_ex(?P<ex>\d+)_t(?P<thresh>[^ _]+)_m(?P<mins>\d+)"
    match = re.search(pattern, filename)
    if match:
        d = match.groupdict()
        # Convert numeric types
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

def analyze_flood_results(csv_path, output_pdf):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    # Remove any rows that are actually the header (e.g., if header line appears as a row)
    if 'run' in df.columns:
        df = df[df['run'] != 'run']

    # --- NEW: Average results for each (binary, input_file) group and save to CSV ---
    group_cols = ['binary', 'input_file']
    avg_cols = [col for col in df.columns if col not in ['run', 'binary', 'input_file']]
    averaged = df.groupby(group_cols, as_index=False)[avg_cols].mean()
    averaged.to_csv('_logs/experiment_results_averaged.csv', index=False)
    
    # 2. Extract metadata from the input_file column
    metadata = df['input_file'].apply(parse_filename)
    df = pd.concat([df, metadata], axis=1)

    # Ensure numeric columns are correct type
    numeric_cols = ['runtime', 'precision_loss', 'total_rain', 'total_water', 'total_water_loss', 'max_spillage_minute', 'max_spillage_scenario', 'max_water_scenario', 'rows', 'cols', 'total_cells', 'clouds', 'ex_factor', 'mins']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Calculate Speedup (always relative to flood_seq)
    config_cols = ['rows', 'scenario', 'clouds', 'ex_factor']
    avg_df = df.groupby(['binary'] + config_cols)['runtime'].mean().reset_index()

    # Create a baseline lookup from flood_seq
    seq_runs = avg_df[avg_df['binary'] == 'flood_seq'].set_index(config_cols)['runtime']

    def get_speedup(row):
        key = tuple(row[config_cols])
        # Only compute speedup for non-flood_seq binaries
        if row['binary'] != 'flood_seq' and key in seq_runs and row['runtime'] > 0:
            return seq_runs[key] / row['runtime']
        return np.nan

    avg_df['speedup'] = avg_df.apply(get_speedup, axis=1)

    # 4. Visualization

    sns.set_theme(style="whitegrid", palette="tab10")
    # Common style params for clarity
    plt_params = {
        'figure.figsize': (5.5, 4),  # ~two-column width
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7,
        'legend.title_fontsize': 7,
        'lines.markersize': 6,
        'lines.linewidth': 1.5,
        'axes.titlepad': 8
    }
    plt.rcParams.update(plt_params)

    # --- FIGURE 1: Strong Scaling (Increasing Clouds, Fixed Grid) ---
    scaling_data = avg_df[avg_df['binary'] != 'flood_seq']
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=scaling_data, x='clouds', y='speedup', hue='binary', style='rows', markers=True, dashes=False, ax=ax1)
    ax1.set_xscale('log', base=2)
    ax1.set_title('Speedup vs. Cloud Density')
    ax1.set_ylabel('Speedup ($T_{seq} / T_{cuda}$)')
    ax1.set_xlabel('Number of Clouds (log2)')
    ax1.legend(loc='best', frameon=True)
    fig1.tight_layout()
    fig1.savefig('figure_speedup_vs_clouds.pdf', bbox_inches='tight')
    plt.close(fig1)

    # --- FIGURE 2: Problem Size Scaling (Increasing Grid Size) ---
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=scaling_data, x='rows', y='speedup', hue='binary', style='clouds', markers=True, dashes=False, ax=ax2)
    ax2.set_xscale('log', base=2)
    ax2.set_title('Speedup vs. Grid Size (Rows=Cols)')
    ax2.set_ylabel('Speedup')
    ax2.set_xlabel('Grid Size (Rows = Cols, log2)')
    ax2.legend(loc='best', frameon=True)
    fig2.tight_layout()
    fig2.savefig('figure_speedup_vs_grid_size.pdf', bbox_inches='tight')
    plt.close(fig2)

    # --- FIGURE 3: Precision Loss Analysis ---
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df[df['binary'] != 'flood_seq'], x='total_rain', y='precision_loss', hue='binary', alpha=0.5, ax=ax3)
    ax3.set_title('Precision Loss vs. Total Rainfall')
    ax3.set_xlabel('Total Rainfall Volume')
    ax3.set_ylabel('Precision Loss')
    ax3.legend(loc='best', frameon=True)
    fig3.tight_layout()
    fig3.savefig('figure_precision_loss_vs_rain.pdf', bbox_inches='tight')
    plt.close(fig3)

    # --- FIGURE 4: Scenario Efficiency (Filtered for fixed grid size and cloud count) ---
    # Choose the most common or median grid size and cloud count for scenario comparison
    grid_mode = avg_df['rows'].mode()[0]
    clouds_mode = avg_df['clouds'].mode()[0]
    filtered = avg_df[(avg_df['rows'] == grid_mode) & (avg_df['clouds'] == clouds_mode) & (avg_df['binary'] != 'flood_seq')]
    fig4, ax4 = plt.subplots()
    sns.barplot(data=filtered, x='scenario', y='speedup', hue='binary', ax=ax4)
    ax4.set_title(f'Speedup by Scenario (Rows={grid_mode}, Clouds={clouds_mode})')
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Speedup')
    ax4.legend(loc='best', frameon=True)
    fig4.tight_layout()
    fig4.savefig('figure_speedup_by_scenario.pdf', bbox_inches='tight')
    plt.close(fig4)

    print("Analysis Complete! Figures saved as individual PDF files for publication.")

    # --- FIGURE 5: Runtime Distribution by Implementation ---
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df, x='binary', y='runtime', hue='scenario', ax=ax5)
    ax5.set_title('Runtime Distribution by Implementation and Scenario')
    ax5.set_xlabel('Implementation')
    ax5.set_ylabel('Runtime (s)')
    ax5.legend(loc='best', frameon=True, title='Scenario')
    fig5.tight_layout()
    fig5.savefig('figure_runtime_distribution_by_impl.pdf', bbox_inches='tight')
    plt.close(fig5)

    # --- FIGURE 6: Precision Loss vs. Runtime ---
    fig6, ax6 = plt.subplots()
    sns.scatterplot(data=df, x='runtime', y='precision_loss', hue='binary', style='scenario', alpha=0.7, ax=ax6)
    ax6.set_title('Precision Loss vs. Runtime')
    ax6.set_xlabel('Runtime (s)')
    ax6.set_ylabel('Precision Loss')
    ax6.legend(loc='best', frameon=True)
    fig6.tight_layout()
    fig6.savefig('figure_precision_loss_vs_runtime.pdf', bbox_inches='tight')
    plt.close(fig6)

    # --- FIGURE 7: Total Water Loss Analysis ---
    fig7, ax7 = plt.subplots()
    sns.boxplot(data=df, x='binary', y='total_water_loss', hue='scenario', ax=ax7)
    ax7.set_title('Total Water Loss by Implementation and Scenario')
    ax7.set_xlabel('Implementation')
    ax7.set_ylabel('Total Water Loss')
    ax7.legend(loc='best', frameon=True, title='Scenario')
    fig7.tight_layout()
    fig7.savefig('figure_total_water_loss_by_impl.pdf', bbox_inches='tight')
    plt.close(fig7)

    # --- FIGURE 8: Scenario Comparison (Average Runtime, Filtered) ---
    avg_runtime_filtered = df[(df['rows'] == grid_mode) & (df['clouds'] == clouds_mode)]
    avg_runtime = avg_runtime_filtered.groupby(['scenario', 'binary'])['runtime'].mean().reset_index()
    fig8, ax8 = plt.subplots()
    sns.barplot(data=avg_runtime, x='scenario', y='runtime', hue='binary', ax=ax8)
    ax8.set_title(f'Average Runtime by Scenario (Rows={grid_mode}, Clouds={clouds_mode})')
    ax8.set_xlabel('Scenario')
    ax8.set_ylabel('Average Runtime (s)')
    ax8.legend(loc='best', frameon=True)
    fig8.tight_layout()
    fig8.savefig('figure_avg_runtime_by_scenario.pdf', bbox_inches='tight')
    plt.close(fig8)

    # --- FIGURE 9: Correlation Heatmap ---
    corr = df[[col for col in df.columns if df[col].dtype in [np.float64, np.int64]]].corr()
    fig_width = max(7, 0.7 * len(corr.columns))
    fig_height = max(6, 0.7 * len(corr.columns))
    fig9, ax9 = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        ax=ax9,
        cbar_kws={'shrink': 0.7},
        annot_kws={"size": 7}
    )
    ax9.set_title('Correlation Heatmap (Numeric Columns)')
    fig9.tight_layout()
    fig9.savefig('figure_correlation_heatmap.pdf', bbox_inches='tight')
    plt.close(fig9)
    
    # --- FIGURE 10: Heatmap (Grid Size x Clouds, colored by Runtime, faceted by Scenario) ---
    # With rows == cols, heatmap is simpler
    for scenario in sorted(avg_df['scenario'].unique()):
        for metric in ['runtime', 'speedup']:
            pivot = avg_df[avg_df['scenario'] == scenario].pivot_table(
                index='rows', columns='clouds', values=metric, aggfunc='mean')
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=ax, cbar_kws={'label': metric})
            ax.set_title(f'{metric.capitalize()} Heatmap\nScenario: {scenario} (Rows=Cols)')
            ax.set_xlabel('Number of Clouds')
            ax.set_ylabel('Grid Size (Rows=Cols)')
            fig.tight_layout()
            fig.savefig(f'figure_heatmap_{metric}_scenario_{scenario}.pdf', bbox_inches='tight')
            plt.close(fig)

    # --- FIGURE 11: Bubble Plot (Grid Size vs Runtime/Speedup, bubble=clouds, color=scenario) ---
    for metric in ['runtime', 'speedup']:
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(
            avg_df['rows'],
            avg_df[metric],
            s=avg_df['clouds'] * 0.7,  # scale for visibility
            c=avg_df['scenario'].astype('category').cat.codes,
            cmap='tab10',
            alpha=0.5,
            edgecolor='k',
            label=None
        )
        # Colorbar for scenario
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(avg_df['scenario'].unique())))
        cbar.ax.set_yticklabels(sorted(avg_df['scenario'].unique()))
        ax.set_xscale('log')
        ax.set_xlabel('Grid Size (Rows=Cols, log2)')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs Grid Size (Bubble=Clouds, Color=Scenario)')
        fig.tight_layout()
        fig.savefig(f'figure_bubble_{metric}_gridsize.pdf', bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    analyze_flood_results('_logs/experiment_results.csv', 'flood_cuda_report.pdf')