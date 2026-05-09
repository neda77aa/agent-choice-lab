import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_pettibone_data(csv_paths: list[Path], out_dir: Path):
    df_list = []
    for csv_path in csv_paths:
        if csv_path.is_dir():
            csv_files = list(csv_path.glob("*.csv"))
            df_list.extend([pd.read_csv(f) for f in csv_files])
        else:
            df_list.append(pd.read_csv(csv_path))
            
    if not df_list:
        print("No CSV files found!")
        return
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Filter for Pettibone stimuli
    pettibone_df = df[df['stimulus_id'].str.startswith('pettibone_2000_')].copy()
    
    if pettibone_df.empty:
        print("No Pettibone & Wedell (2000) experiments found in the provided data.")
        return
        
    # Extract the decoy type from the stimulus ID (e.g., pettibone_2000_computers_inferior -> inferior)
    pettibone_df['decoy_type'] = pettibone_df['stimulus_id'].apply(lambda x: x.split('_')[-1].capitalize())
    
    # We focus on the 3-option condition where the decoy effect is observed
    df_3opt = pettibone_df[pettibone_df['condition'] == '3_opt'].copy()
    
    # Calculate choice percentages
    choice_counts = df_3opt.groupby(['decoy_type', 'choice_key']).size().reset_index(name='count')
    total_counts = df_3opt.groupby('decoy_type').size().reset_index(name='total')
    
    merged = pd.merge(choice_counts, total_counts, on='decoy_type')
    merged['percentage'] = (merged['count'] / merged['total']) * 100
    
    # Human benchmark data from the paper
    human_data = {
        'decoy_type': ['Inferior', 'Inferior', 'Inferior', 
                       'Compromise', 'Compromise', 'Compromise',
                       'Phantom', 'Phantom', 'Phantom'],
        'choice_key': ['Target', 'Rival', 'Decoy',
                       'Target', 'Rival', 'Decoy',
                       'Target', 'Rival', 'Decoy'],
        'percentage': [58, 29, 13,
                       46, 32, 22,
                       57, 43, 0] # Phantom decoy couldn't be chosen
    }
    human_df = pd.DataFrame(human_data)
    human_df['source'] = 'Human (Paper)'
    
    merged['source'] = 'LLM'
    
    # Combine LLM and Human data for plotting
    plot_df = pd.concat([merged[['decoy_type', 'choice_key', 'percentage', 'source']], human_df], ignore_index=True)
    
    # Ensure all choice keys are present even if 0
    categories = ['Inferior', 'Compromise', 'Phantom']
    keys = ['Target', 'Rival', 'Decoy']
    sources = ['LLM', 'Human (Paper)']
    
    # Reindex to fill missing values with 0
    idx = pd.MultiIndex.from_product([categories, keys, sources], names=['decoy_type', 'choice_key', 'source'])
    plot_df = plot_df.set_index(['decoy_type', 'choice_key', 'source']).reindex(idx, fill_value=0).reset_index()
    
    # Plotting
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Create a grid of subplots for each decoy type
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    for i, decoy in enumerate(categories):
        ax = axes[i]
        subset = plot_df[plot_df['decoy_type'] == decoy]
        
        sns.barplot(
            data=subset, 
            x='choice_key', 
            y='percentage', 
            hue='source', 
            ax=ax,
            palette={'LLM': 'skyblue', 'Human (Paper)': 'salmon'}
        )
        
        ax.set_title(f'{decoy} Decoy Effect')
        ax.set_xlabel('Chosen Alternative')
        ax.set_ylabel('Selection Percentage (%)' if i == 0 else '')
        ax.set_ylim(0, 100)
        
        if i == 2:
            ax.legend(title='Source')
        else:
            ax.get_legend().remove()
            
    plt.suptitle('Comparison of Decoy Effects: LLM vs Human (Pettibone & Wedell 2000)', fontsize=16)
    plt.tight_layout()
    
    plot_path = out_dir / 'pettibone_2000_decoy_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Save the aggregated data to CSV
    csv_path = out_dir / 'pettibone_2000_summary.csv'
    plot_df.to_csv(csv_path, index=False)
    
    print(f"Analysis complete. Plot saved to {plot_path}")
    print(f"Summary data saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Pettibone & Wedell 2000 experiments.")
    parser.add_argument('--csv-path', type=str, nargs='+', required=True, help='Path(s) to results CSV file or directories containing CSVs')
    parser.add_argument('--out-dir', type=str, default='analysis_output/pettibone', help='Output directory')
    args = parser.parse_args()
    
    paths = [Path(p) for p in args.csv_path]
    analyze_pettibone_data(paths, Path(args.out_dir))