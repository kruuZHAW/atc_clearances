import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

##### HELPER ####

def build_per_day_plot(per_day: pd.DataFrame, 
                       feature: str,
                       title: str,
                       legend_anchor: Tuple[int, int]):
    
    """"
    per_day: dataframe containing the data
    feature: feature to plot among ["singleton_rate", 'mean_rows_per_callsign', 'median_gap']
    """
    
    y = per_day[feature]
    mean_val = y.mean()
    q25, q75 = np.quantile(y, [0.25, 0.75])

    # --- Plot ---
    x = np.arange(len(per_day))
    labels = per_day.index.strftime('%Y-%m-%d')

    plt.style.use('seaborn-v0_8-whitegrid')  
    fig, ax = plt.subplots(figsize=(12, 6))

    # IQR band and reference lines
    ax.axhspan(q25, q75, alpha=0.18, label=f'IQR (Q25={q25:.2f}, Q75={q75:.2f})', zorder=1)
    ax.plot(x, y.values, marker='o', linewidth=2.0, alpha=0.9, zorder=3, label='Daily mean')
    ax.axhline(mean_val, linestyle='--', linewidth=1.5, zorder=4, label=f'Mean = {mean_val:.2f}')
    ax.axhline(q25, linestyle=':', linewidth=1.0, zorder=4, label=f'Q25 = {q25:.2f}')
    ax.axhline(q75, linestyle=':', linewidth=1.0, zorder=4, label=f'Q75 = {q75:.2f}')

    # Labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title(title, fontsize=17, pad=15)

    # Style
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles, lab = ax.get_legend_handles_labels()
    order = [lab.index(l) for l in [
        f'IQR (Q25={q25:.2f}, Q75={q75:.2f})', 'Daily mean',
        f'Mean = {mean_val:.2f}', f'Q25 = {q25:.2f}', f'Q75 = {q75:.2f}'
    ] if l in lab]
    ax.legend([handles[i] for i in order],
            [lab[i] for i in order],
            frameon=False, ncol=2, fontsize=13, loc='upper left', bbox_to_anchor=legend_anchor)

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    
def main():
    coms_per_cs = pd.read_parquet("coms_per_callsign.parquet")
    singleton_per_cs = pd.read_parquet("singleton_rate_df.parquet")
    response_gap = pd.read_parquet("response_gap.parquet")
    
    build_per_day_plot(coms_per_cs,
                        'mean_rows_per_callsign',
                        'Average # of communications per callsign',
                        (0, 0.3))
    
    build_per_day_plot(singleton_per_cs,
                        'singleton_rate',
                        'Per-day rate of detected conversations with only one call (in %)',
                        (0, 0.95))
    
    build_per_day_plot(response_gap,
                        'median_gap',
                        'Per-day median gap between communications within a detected conversation (s)',
                        (0, 0.95))
    
    return
    
if __name__ == "__main__":
    main()