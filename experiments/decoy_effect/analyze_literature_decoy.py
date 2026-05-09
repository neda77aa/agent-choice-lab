import argparse
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_ENABLED = True
except ImportError:
    PLOTS_ENABLED = False
    print("Warning: matplotlib/seaborn not found or incompatible. Skipping plots.")

from pathlib import Path

# ── Centralised model styling ──────────────────────────────────────────────
# Every model gets a fixed colour so figures are consistent regardless of
# which subset of models is included.  Keys are the *display* names (after
# stripping the "vertex-" prefix).
MODEL_COLORS = {
    "gemini-2-5-pro":       "#2ca02c",   # green
    "claude-opus-4-6":      "#ff7f0e",   # orange
    "gpt-5-2":              "#1f77b4",   # blue
    "deepseek-v3-2":        "#9467bd",   # purple
    "qwen-3-5":             "#17becf",   # teal
    "Human Benchmark":      "#d62728",   # red
}

def clean_agent_name(name: str) -> str:
    """Strip the 'vertex-' prefix so figures show the bare model name."""
    s = str(name)
    if s.startswith("vertex-"):
        return s[len("vertex-"):]
    return s

def get_model_color(name: str) -> str:
    """Return the fixed colour for a (cleaned) model name, gray if unknown."""
    return MODEL_COLORS.get(name, "gray")

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    STATSMODELS_ENABLED = True
except Exception as e:
    # statsmodels depends on scipy; on some machines scipy may be mis-installed
    # (e.g., wrong architecture wheel). We still want plots/aggregations to run.
    STATSMODELS_ENABLED = False
    sm = None
    smf = None
    print(f"Warning: statsmodels/scipy unavailable ({e}). Regressions will be skipped; plots will still be generated.")

def run_regression(df_subset: pd.DataFrame, subset_name: str, out_dir: Path):
    """Runs OLS regressions to analyze target share and decoy effects.

    Primary specification (per model):
      - Uses only 3-option trials
      - Controls for the corresponding 2-option target share (baseline attractiveness)
      - Uses absolute position difference between target and rival
      - Avoids collinearity between position and condition indicators
    Supplementary: pooled regression across all models with agent fixed effects.
    """
    if not STATSMODELS_ENABLED:
        print(f"Skipping regression analysis for {subset_name} (statsmodels/scipy not available).")
        return {}

    print(f"Running regression analysis for {subset_name}...")
    regression_results = {}

    df_reg = df_subset.copy()

    # Calculate positions based on option_order
    def get_pos(order, option):
        if pd.isna(order):
            return 0
        parts = order.split(',')
        if option in parts:
            return parts.index(option) + 1
        return 0

    df_reg['pos_target'] = df_reg['option_order'].apply(lambda x: get_pos(x, 'Target'))
    df_reg['pos_rival'] = df_reg['option_order'].apply(lambda x: get_pos(x, 'Rival'))
    df_reg['pos_decoy'] = df_reg['option_order'].apply(lambda x: get_pos(x, 'Decoy'))

    # Absolute position difference between target and rival (Rec 2)
    df_reg['abs_pos_diff_target_rival'] = (df_reg['pos_target'] - df_reg['pos_rival']).abs()

    file_suffix = f"_{subset_name}" if subset_name != 'regular' else ""

    # --- Compute baseline target share from 2-option condition (Rec 4) ---
    group_keys = ['agent', 'stimulus_id', 'temperature', 'prompting_mode']
    df_2opt = df_reg[df_reg['condition'] == '2_opt']
    baseline = df_2opt.groupby(group_keys)['is_target'].mean().reset_index()
    baseline.rename(columns={'is_target': 'baseline_target_share'}, inplace=True)

    df_3opt = df_reg[df_reg['condition'] == '3_opt'].copy()
    df_3opt = pd.merge(df_3opt, baseline, on=group_keys, how='left')

    # --- Per-model regressions on 3-opt with baseline control (Rec 1 + 3 + 4) ---
    agents = sorted(df_reg['agent'].unique())
    formula = ("is_target ~ baseline_target_share + C(pos_target) "
               "+ abs_pos_diff_target_rival + temperature + C(prompting_mode)")

    out_file = out_dir / f"regression_results{file_suffix}.txt"
    with open(out_file, "w") as f:
        f.write("Regression Results\n")
        f.write("Specification: 3-option trials only, 2-option baseline as control\n")
        f.write("DV: is_target (1 = target chosen, 0 = otherwise)\n")
        f.write("Model: OLS (Linear Probability Model)\n")
        f.write("=" * 70 + "\n")
        f.write("\nVariable descriptions:\n")
        f.write("- baseline_target_share: P(Target) from matching 2-option condition\n")
        f.write("- C(pos_target): position of target in choice list (1, 2, or 3)\n")
        f.write("- abs_pos_diff_target_rival: |pos_target - pos_rival|\n\n")

        # Per-model regressions
        for agent_name in agents:
            df_agent = df_3opt[df_3opt['agent'] == agent_name].dropna(subset=['baseline_target_share'])

            if df_agent.empty:
                print(f"  Skipping {agent_name}: insufficient data.")
                continue

            try:
                model = smf.ols(formula, data=df_agent).fit()
                regression_results[agent_name] = model
                f.write("=" * 70 + "\n")
                f.write(f"Per-Model Regression: {agent_name}\n")
                f.write("=" * 70 + "\n\n")
                f.write(model.summary().as_text())
                f.write("\n\n")
            except Exception as e:
                print(f"  Error running regression for {agent_name}: {e}")

        # Supplementary: Pooled regression with agent fixed effects
        df_3opt_clean = df_3opt.dropna(subset=['baseline_target_share'])
        if not df_3opt_clean.empty:
            try:
                formula_pooled = ("is_target ~ baseline_target_share + C(pos_target) "
                                  "+ abs_pos_diff_target_rival + temperature "
                                  "+ C(prompting_mode) + C(agent)")
                model_pooled = smf.ols(formula_pooled, data=df_3opt_clean).fit()
                regression_results['_pooled'] = model_pooled
                f.write("=" * 70 + "\n")
                f.write("Pooled Regression (All Models, with agent fixed effects)\n")
                f.write("=" * 70 + "\n\n")
                f.write(model_pooled.summary().as_text())
                f.write("\n")
            except Exception as e:
                print(f"  Error running pooled regression: {e}")

    return regression_results

def analyze_data(csv_paths: list[Path], out_dir: Path):
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
    
    # Exclude Moran 2006 experiments from the analysis
    if 'stimulus_id' in df.columns:
        df = df[~df['stimulus_id'].astype(str).str.contains('moran_2006', na=False)].copy()
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Drop unparseable / Invalid rows before computing target share. Coding them as
    # non-Target would deflate P(Target) — affects models with high Invalid rates
    # (e.g. DeepSeek for-me trials are 16% Invalid).
    n_before = len(df)
    df = df[df['choice_key'].isin(['Target', 'Rival', 'Decoy'])].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped:,} Invalid/unparseable rows ({100*n_dropped/n_before:.2f}% of total).")

    # Drop truncated reasoning rows. For deliberative/knowledge modes, the choice_key
    # was set by a regex fallback (last A/B/C mentioned in the unfinished reasoning)
    # whenever the model did not reach the canonical "Therefore, I choose [X]" sentence.
    # Those choices are unreliable and should not enter any table or figure.
    # Qwen wraps reasoning in <think>...</think>; for Qwen we instead require visible
    # text after the </think> closing tag (the "I choose X" can appear before or after).
    if 'raw_text' in df.columns and 'prompting_mode' in df.columns:
        rt = df['raw_text'].fillna('').astype(str)
        is_reason = df['prompting_mode'].astype(str).str.startswith(('deliberative', 'knowledge'))
        agent_str = df.get('agent', pd.Series([''] * len(df))).astype(str).str.lower()
        is_qwen = agent_str.str.contains('qwen', na=False)

        # Standard completion check: contains "therefore" (case-insensitive) OR "i choose"
        has_final = rt.str.lower().str.contains('therefore', na=False) | rt.str.lower().str.contains('i choose', na=False)

        # Qwen-specific: require non-empty text after </think>
        qwen_after = rt.str.split(r'</think>', regex=True, n=1).str[-1].str.strip()
        qwen_complete = rt.str.contains(r'</think>', regex=True, na=False) & (qwen_after != '')

        # A reasoning row is "complete" if it satisfies the model-appropriate check
        complete = (~is_reason) | (is_qwen & qwen_complete) | (~is_qwen & has_final)

        n_before2 = len(df)
        df = df[complete].copy()
        n_truncated = n_before2 - len(df)
        if n_truncated:
            print(f"Dropped {n_truncated:,} truncated reasoning rows lacking 'Therefore, I choose [X]' "
                  f"(or, for Qwen, lacking visible answer after </think>) "
                  f"({100*n_truncated/n_before2:.2f}% of remaining).")

    df['is_target'] = (df['choice_key'] == 'Target').astype(float)
    
    # Ensure agent column exists
    if 'agent' not in df.columns:
        df['agent'] = 'unknown_agent'

    # Clean agent names (strip "vertex-" prefix) for display
    df['agent'] = df['agent'].apply(clean_agent_name)
        
    # Split the DataFrame into regular and for_me runs
    is_for_me_mask = df['prompting_mode'].astype(str).str.endswith('_for_me')
    df_regular = df[~is_for_me_mask].copy()
    df_forme = df[is_for_me_mask].copy()
    
    # Process both subsets
    subsets = []
    if not df_regular.empty:
        subsets.append(('regular', df_regular))
    if not df_forme.empty:
        subsets.append(('forme', df_forme))
        
    for subset_name, subset_df in subsets:
        print(f"Processing subset: {subset_name}")
        
        # Run regression analysis
        reg_results = run_regression(subset_df, subset_name, out_dir)
        
        # Group by agent, stimulus, condition, temperature, prompting_mode
        group_cols = ['agent', 'stimulus_id', 'temperature', 'prompting_mode']
        grouped = subset_df.groupby(group_cols + ['condition'])['is_target'].mean().reset_index()
        
        # Group for continuous probability if available
        if 'prob_Target' in subset_df.columns:
            prob_grouped = subset_df.groupby(group_cols + ['condition'])['prob_Target'].mean().reset_index()
            prob_pivoted = prob_grouped.pivot(index=group_cols, 
                                              columns='condition', values='prob_Target').reset_index()
            prob_pivoted = prob_pivoted.rename(columns={'2_opt': 'prob_2_opt', '3_opt': 'prob_3_opt'})
            if 'prob_2_opt' in prob_pivoted.columns and 'prob_3_opt' in prob_pivoted.columns:
                prob_pivoted['Delta_LLM_Prob'] = prob_pivoted['prob_3_opt'] - prob_pivoted['prob_2_opt']
            else:
                prob_pivoted['Delta_LLM_Prob'] = np.nan
                if 'prob_2_opt' not in prob_pivoted.columns: prob_pivoted['prob_2_opt'] = np.nan
                if 'prob_3_opt' not in prob_pivoted.columns: prob_pivoted['prob_3_opt'] = np.nan
        else:
            prob_pivoted = None

        # Pivot condition to get P(target|2) and P(target|3)
        pivoted = grouped.pivot(index=group_cols, 
                                columns='condition', values='is_target').reset_index()
        
        # Calculate LLM Delta
        if '2_opt' in pivoted.columns and '3_opt' in pivoted.columns:
            pivoted['Delta_LLM'] = pivoted['3_opt'] - pivoted['2_opt']
        else:
            print("Warning: Missing 2_opt or 3_opt data. Creating empty Delta_LLM.")
            pivoted['Delta_LLM'] = np.nan
            if '2_opt' not in pivoted.columns: pivoted['2_opt'] = np.nan
            if '3_opt' not in pivoted.columns: pivoted['3_opt'] = np.nan
            
        if prob_pivoted is not None:
            pivoted = pd.merge(pivoted, prob_pivoted, on=group_cols, how='left')

        # Merge human data from the first row of each stimulus
        if 'human_target_2_opt' in subset_df.columns and 'human_target_3_opt' in subset_df.columns:
            human_data = subset_df[['stimulus_id', 'human_target_2_opt', 'human_target_3_opt']].drop_duplicates()
            human_data['Delta_Human'] = human_data['human_target_3_opt'] - human_data['human_target_2_opt']
            results = pd.merge(pivoted, human_data, on='stimulus_id', how='left')
        else:
            human_data = pd.DataFrame()
            results = pivoted
            results['Delta_Human'] = np.nan
            
        file_suffix = f"_{subset_name}" if subset_name != 'regular' else ""

        results.to_csv(out_dir / f"detailed_results{file_suffix}.csv", index=False)
        
        if PLOTS_ENABLED:
            try:
                # Plots
                sns.set_style("whitegrid")
                
                # 1. Delta vs Temperature — exclude models that ignore temperature
                # at the API level (GPT-5.2 drops temperature via additional_drop_params)
                # to avoid implying temperature variation that the model never received.
                TEMP_IGNORING_AGENTS = {'gpt-5-2'}
                temp_results = results[~results['agent'].isin(TEMP_IGNORING_AGENTS)].copy()
                plt.figure(figsize=(8, 5))

                # Check if we have multiple agents to split by hue
                has_multiple_agents = 'agent' in temp_results.columns and temp_results['agent'].nunique() > 1
                hue_arg = 'agent' if has_multiple_agents else None

                agent_palette = {a: get_model_color(a) for a in temp_results['agent'].unique()} if hue_arg else None
                sns.lineplot(data=temp_results, x='temperature', y='Delta_LLM', hue=hue_arg, marker='o', errorbar=('ci', 95), palette=agent_palette)

                if not temp_results['Delta_Human'].isna().all():
                    mean_human_delta = temp_results['Delta_Human'].mean()
                    plt.axhline(mean_human_delta, color='red', linestyle='--', label='Human Mean Δ')

                title_suffix = " (For Me)" if subset_name == "forme" else ""
                plt.title(f'Decoy Effect (Δ) across Temperatures{title_suffix}\n(GPT-5.2 excluded — API ignores temperature)')
                plt.ylabel('Δ (P(Target|3) - P(Target|2))')
                plt.xlabel('Temperature')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(out_dir / f"delta_vs_temperature{file_suffix}.png")
                plt.close()
                
                # 2. Delta vs Prompting Mode
                # Build a palette over ALL agents in `results` — the temperature plot's
                # agent_palette excludes GPT-5.2 and would be missing keys here.
                full_agent_palette = {a: get_model_color(a) for a in results['agent'].unique()} if hue_arg else None
                plt.figure(figsize=(8, 5))
                sns.barplot(data=results, x='prompting_mode', y='Delta_LLM', hue=hue_arg, errorbar=('ci', 95), palette=full_agent_palette)
                
                if not results['Delta_Human'].isna().all():
                    plt.axhline(mean_human_delta, color='red', linestyle='--', label='Human Mean Δ')
                    
                plt.title(f'Decoy Effect (Δ) by Prompting Mode{" (For Me)" if subset_name == "forme" else ""}')
                plt.ylabel('Δ (P(Target|3) - P(Target|2))')
                plt.xlabel('Prompting Mode')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(out_dir / f"delta_vs_prompting_mode{file_suffix}.png")
                plt.close()
                
                # 3. Forest Plot by Stimulus
                plt.figure(figsize=(10, 8))
                
                # Plot LLM estimates with Error Bars (95% CI)
                sns.pointplot(
                    data=results,
                    y='stimulus_id',
                    x='Delta_LLM',
                    hue=hue_arg,
                    errorbar=('ci', 95),
                    linestyles='', # Do not connect the dots across different stimuli
                    dodge=0.4 if hue_arg else False,  # Spread out the dots vertically
                    capsize=0.1,
                    orient='h',
                    markers='o',
                    palette=full_agent_palette
                )

                # Draw horizontal separator lines between stimuli for better readability
                unique_stimuli = results['stimulus_id'].unique()
                for i in range(len(unique_stimuli) - 1):
                    plt.axhline(i + 0.5, color='gray', linestyle=':', alpha=0.3)
                
                # Plot Human Benchmarks as distinct red X marks
                if not results['Delta_Human'].isna().all():
                    human_subset = results.drop_duplicates(subset=['stimulus_id']).dropna(subset=['Delta_Human'])
                    if not human_subset.empty:
                        try:
                            # Map categorical y-axis to numeric indices safely
                            y_categories = results['stimulus_id'].unique()
                            y_map = {val: i for i, val in enumerate(y_categories)}
                            
                            y_indices = []
                            x_vals = []
                            for _, row in human_subset.iterrows():
                                val = row['stimulus_id']
                                if val in y_map:
                                    y_indices.append(y_map[val])
                                    x_vals.append(row['Delta_Human'])
                                    
                            if x_vals:
                                plt.scatter(
                                    x=x_vals, 
                                    y=y_indices, 
                                    color='red', 
                                    marker='x', 
                                    s=100, 
                                    zorder=10, 
                                    label='Human Benchmark'
                                )
                        except Exception as e:
                            import traceback
                            print(f"Warning: Could not plot human benchmark scatter points: {e}")
                            traceback.print_exc()

                # Draw vertical line at 0 (no effect)
                plt.axvline(0, color='black', linestyle='--', alpha=0.5)
                
                # Formatting
                plt.title(f'Decoy Effect (Δ) by Stimulus{" (For Me)" if subset_name == "forme" else ""}')
                plt.xlabel('Estimated Effect (Δ = P(Target|3) - P(Target|2))')
                plt.ylabel('Stimulus / Scenario')
                
                # Fix Legend
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(out_dir / f"delta_vs_stimulus_forest_plot{file_suffix}.png")
                plt.close()

                # 3b. Split Forest Plots — one for Pettibone, one for the rest
                pettibone_mask = results['stimulus_id'].str.startswith('pettibone_2000_')
                split_groups = [
                    ('classic', ~pettibone_mask, 'Decoy Effect (Δ) by Stimulus — Classic Studies'),
                    ('pettibone', pettibone_mask, 'Decoy Effect (Δ) by Stimulus — Pettibone & Wedell (2000)'),
                ]
                for split_tag, mask, split_title in split_groups:
                    split_df = results[mask]
                    if split_df.empty:
                        continue

                    n_stimuli = split_df['stimulus_id'].nunique()
                    row_height = 0.45 if split_tag == 'classic' else 0.55
                    fig_h = max(5, row_height * n_stimuli)
                    fig_w = 14 if split_tag == 'classic' else 12
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                    sns.pointplot(
                        data=split_df,
                        y='stimulus_id',
                        x='Delta_LLM',
                        hue=hue_arg,
                        errorbar=('ci', 95),
                        linestyles='',
                        dodge=0.4 if hue_arg else False,
                        capsize=0.1,
                        orient='h',
                        markers='o',
                        palette=full_agent_palette,
                        ax=ax
                    )

                    unique_stim = split_df['stimulus_id'].unique()
                    for i in range(len(unique_stim) - 1):
                        ax.axhline(i + 0.5, color='gray', linestyle=':', alpha=0.3)

                    if not split_df['Delta_Human'].isna().all():
                        human_sub = split_df.drop_duplicates(subset=['stimulus_id']).dropna(subset=['Delta_Human'])
                        if not human_sub.empty:
                            try:
                                y_cats = split_df['stimulus_id'].unique()
                                y_map = {v: i for i, v in enumerate(y_cats)}
                                y_idx, x_v = [], []
                                for _, r in human_sub.iterrows():
                                    if r['stimulus_id'] in y_map:
                                        y_idx.append(y_map[r['stimulus_id']])
                                        x_v.append(r['Delta_Human'])
                                if x_v:
                                    ax.scatter(x=x_v, y=y_idx, color='red',
                                               marker='x', s=120, zorder=10,
                                               label='Human Benchmark')
                            except Exception:
                                pass

                    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                    fs = 15 if split_tag == 'classic' else 14
                    ax.set_title(f'{split_title}{" (For Me)" if subset_name == "forme" else ""}',
                                 fontsize=fs + 1, pad=12)
                    ax.set_xlabel('Estimated Effect (Δ = P(Target|3) - P(Target|2))', fontsize=fs)
                    ax.set_ylabel('Stimulus / Scenario', fontsize=fs)
                    ax.tick_params(axis='y', labelsize=fs - 1)
                    ax.tick_params(axis='x', labelsize=fs - 1)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1),
                              loc='upper left', fontsize=fs - 1)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"delta_vs_stimulus_forest_{split_tag}{file_suffix}.png",
                                bbox_inches='tight', dpi=150)
                    plt.close(fig)

                # 4. Delta vs Temperature by Prompting Mode across Models
                # Create a dictionary to map prompting modes to colors different from agent default colors
                base_mode_colors = {'fast': '#d62728', 'deliberative': '#9467bd', 'knowledge': '#8c564b'}

                # Exclude GPT-5.2 — its API ignores temperature (drop_params strips it).
                temp_mode_results = results[~results['agent'].isin(TEMP_IGNORING_AGENTS)].copy()

                current_modes = temp_mode_results['prompting_mode'].unique()
                palette = {}
                for mode in current_modes:
                    base_mode = mode.replace('_for_me', '')
                    palette[mode] = base_mode_colors.get(base_mode, 'black')

                g = sns.relplot(
                    data=temp_mode_results,
                    x='temperature',
                    y='Delta_LLM',
                    hue='prompting_mode',
                    col='agent',
                    col_wrap=3,
                    kind='line',
                    marker='o',
                    errorbar=('ci', 95),
                    height=4,
                    aspect=1.2,
                    palette=palette
                )
                g.fig.suptitle(f'Decoy Effect (Δ) vs Temperature by Prompting Mode{" (For Me)" if subset_name == "forme" else ""}\n(GPT-5.2 excluded — API ignores temperature)', y=1.05)
                g.set_axis_labels('Temperature', 'Δ (P(Target|3) - P(Target|2))')
                plt.tight_layout()
                plt.savefig(out_dir / f"delta_vs_temp_by_mode_agent{file_suffix}.png")
                plt.close()
                
                # 5. Target Share Forest Plot by Stimulus and Condition (2_opt vs 3_opt)
                llm_plot_df = subset_df[['agent', 'stimulus_id', 'condition', 'is_target']].copy()
                llm_plot_df.rename(columns={'agent': 'Entity', 'is_target': 'target_share'}, inplace=True)
                
                if 'human_target_2_opt' in human_data.columns and 'human_target_3_opt' in human_data.columns:
                    hum_2 = human_data[['stimulus_id', 'human_target_2_opt']].copy()
                    hum_2['condition'] = '2_opt'
                    hum_2['Entity'] = 'Human Benchmark'
                    hum_2.rename(columns={'human_target_2_opt': 'target_share'}, inplace=True)
                    
                    hum_3 = human_data[['stimulus_id', 'human_target_3_opt']].copy()
                    hum_3['condition'] = '3_opt'
                    hum_3['Entity'] = 'Human Benchmark'
                    hum_3.rename(columns={'human_target_3_opt': 'target_share'}, inplace=True)
                    
                    plot_df = pd.concat([llm_plot_df, hum_2, hum_3], ignore_index=True)
                else:
                    plot_df = llm_plot_df

                # Prepare styling – use centralised MODEL_COLORS
                entities = plot_df['Entity'].unique().tolist()

                if 'Human Benchmark' in entities:
                    entities.remove('Human Benchmark')
                    hue_order = entities + ['Human Benchmark']
                    markers = ['o'] * len(entities) + ['X']
                else:
                    hue_order = entities
                    markers = ['o'] * len(entities)

                palette_ents = {ent: get_model_color(ent) for ent in hue_order}
                    
                g = sns.catplot(
                    data=plot_df,
                    y='stimulus_id',
                    x='target_share',
                    hue='Entity',
                    hue_order=hue_order,
                    col='condition',
                    col_order=['2_opt', '3_opt'],
                    kind='point',
                    errorbar=('ci', 95),
                    linestyles='',
                    dodge=0.5,
                    capsize=0.1,
                    height=10,
                    aspect=0.8,
                    palette=palette_ents,
                    markers=markers
                )
                
                g.fig.suptitle(f'Target Share by Stimulus: Models vs Human Benchmark{" (For Me)" if subset_name == "forme" else ""}', y=1.02)
                g.set_axis_labels('Target Share (P(Target))', 'Stimulus / Scenario')
                
                # Draw horizontal separator lines
                for ax in g.axes.flat:
                    unique_stimuli = plot_df['stimulus_id'].unique()
                    for i in range(len(unique_stimuli) - 1):
                        ax.axhline(i + 0.5, color='gray', linestyle=':', alpha=0.3)
                        
                # Move legend outside the plot
                sns.move_legend(g, "center left", bbox_to_anchor=(1.02, 0.5))
                
                plt.savefig(out_dir / f"target_share_vs_stimulus_by_agent{file_suffix}.png", bbox_inches='tight')
                plt.close()

                # 6. Frederick et al. (2014): compare decoy-effect magnitudes for visual vs non-visual stimuli
                # We operationalize the decoy effect as Δ = P(Target|3_opt) - P(Target|2_opt).
                # The Frederick stimuli include both non-visual IDs and corresponding `_visual` IDs.
                # For a fair comparison, we restrict to base stimuli that appear in BOTH variants.
                fred_mask = results['stimulus_id'].astype(str).str.startswith('frederick_2014_')
                fred_res = results.loc[fred_mask, ['agent', 'stimulus_id', 'temperature', 'prompting_mode', 'Delta_LLM']].copy()
                if not fred_res.empty:
                    fred_res['is_visual'] = fred_res['stimulus_id'].astype(str).str.endswith('_visual')
                    fred_res['base_stimulus_id'] = fred_res['stimulus_id'].astype(str).str.replace(r'_visual$', '', regex=True)

                    # Determine matched bases (present in both visual and non-visual forms)
                    base_counts = (
                        fred_res.groupby(['base_stimulus_id', 'is_visual'])
                        .size()
                        .reset_index(name='n')
                        .pivot(index='base_stimulus_id', columns='is_visual', values='n')
                    )
                    matched_bases = base_counts.dropna().index.tolist() if base_counts is not None else []
                    fred_res = fred_res[fred_res['base_stimulus_id'].isin(matched_bases)].copy()

                    if not fred_res.empty:
                        fred_res['visual_label'] = np.where(fred_res['is_visual'], 'Visual', 'Non-visual')

                        # Use colors that do NOT overlap with the agent palette (blue/orange/green)
                        # Suggested: purple + gray
                        visual_palette = {
                            'Non-visual': '#7f7f7f',  # gray
                            'Visual': '#9467bd',      # purple
                        }

                        plt.figure(figsize=(9, 4.8))
                        sns.barplot(
                            data=fred_res,
                            x='agent',
                            y='Delta_LLM',
                            hue='visual_label',
                            errorbar=('ci', 95),
                            palette=visual_palette,
                        )
                        plt.axhline(0, color='black', linestyle='--', alpha=0.6)
                        plt.title(
                            f"Frederick et al. (2014): Decoy effect (Δ) in visual vs non-visual stimuli"
                            f"{' (For Me)' if subset_name == 'forme' else ''}"
                        )
                        plt.ylabel('Δ = P(Target|3\_opt) − P(Target|2\_opt)')
                        plt.xlabel('Agent')
                        plt.xticks(rotation=30, ha='right')
                        plt.legend(title='Stimulus format', bbox_to_anchor=(1.02, 1), loc='upper left')
                        plt.tight_layout()
                        plt.savefig(out_dir / f"frederick_decoy_effect_visual_vs_nonvisual_by_agent{file_suffix}.png")
                        plt.close()


            except Exception as e:
                import traceback
                print(f"Error generating plots for {subset_name}: {e}")
                traceback.print_exc()
        
        # Report
        with open(out_dir / f"report{file_suffix}.md", "w") as f:
            f.write(f"# Decoy Effect Replication with LLMs{' (For Me Perspective)' if subset_name == 'forme' else ''}\n\n")
            f.write("## Overview\n")
            f.write("This report tests whether LLMs violate regularity by exhibiting the decoy effect, and whether knowledge/deliberation changes this behavior.\n\n")
            
            f.write("## Overall Results\n")
            overall_mean = results['Delta_LLM'].mean()
            f.write(f"- **Mean LLM Decoy Effect (Δ) [Discrete Choices]:** {overall_mean:.3f}\n")
            if 'Delta_LLM_Prob' in results.columns:
                overall_prob_mean = results['Delta_LLM_Prob'].mean()
                f.write(f"- **Mean LLM Decoy Effect (Δ) [Continuous log-p]:** {overall_prob_mean:.3f}\n")
            if 'Delta_Human' in results.columns:
                f.write(f"- **Mean Human Decoy Effect (Δ):** {results['Delta_Human'].mean():.3f}\n\n")

            f.write("## Decoy Effect by Agent / Model\n")
            agent_agg_cols = ['Delta_LLM']
            if 'Delta_LLM_Prob' in results.columns:
                agent_agg_cols.append('Delta_LLM_Prob')
            agent_agg = results.groupby('agent')[agent_agg_cols].mean().reset_index()
            f.write(agent_agg.to_markdown(index=False) + "\n\n")

            # Model vs Human Benchmark difference table
            if 'human_target_2_opt' in results.columns and 'human_target_3_opt' in results.columns:
                results['diff_2opt'] = results['2_opt'] - results['human_target_2_opt']
                results['diff_3opt'] = results['3_opt'] - results['human_target_3_opt']

                human_mean_2 = results['human_target_2_opt'].dropna().mean()
                human_mean_3 = results['human_target_3_opt'].dropna().mean()

                diff_table = results.groupby('agent').agg(
                    Model_2opt=('2_opt', 'mean'),
                    Diff_2opt=('diff_2opt', 'mean'),
                    Model_3opt=('3_opt', 'mean'),
                    Diff_3opt=('diff_3opt', 'mean'),
                ).reset_index()

                diff_table['Avg_Diff'] = (diff_table['Diff_2opt'] + diff_table['Diff_3opt']) / 2
                diff_table.insert(2, 'Human_2opt', human_mean_2)
                diff_table.insert(5, 'Human_3opt', human_mean_3)

                # Format as percentages for readability
                pct_cols = ['Model_2opt', 'Human_2opt', 'Diff_2opt',
                            'Model_3opt', 'Human_3opt', 'Diff_3opt', 'Avg_Diff']
                diff_table_fmt = diff_table.copy()
                for c in pct_cols:
                    diff_table_fmt[c] = diff_table_fmt[c].apply(lambda v: f"{v * 100:.1f}%")

                diff_table_fmt.to_csv(out_dir / f"model_vs_human_diff{file_suffix}.csv", index=False)

                f.write("## Model vs Human Benchmark\n")
                f.write("Difference = Model − Human (positive means model over-selects target).\n\n")
                f.write(diff_table_fmt.to_markdown(index=False) + "\n\n")

            f.write("## Decoy Effect by Temperature\n")
            temp_agg = results.groupby('temperature')['Delta_LLM'].mean().reset_index()
            f.write(temp_agg.to_markdown(index=False) + "\n\n")
            
            f.write("## Decoy Effect by Prompting Mode\n")
            mode_agg = results.groupby('prompting_mode')['Delta_LLM'].mean().reset_index()
            f.write(mode_agg.to_markdown(index=False) + "\n\n")
            
            f.write("### Knowledge vs Implementation Gap\n")
            f.write("If explicit knowledge of the bias eliminates it, the `knowledge` mode should show Δ near 0. If it implements the effect anyway, it may be a structural property or implicit instruction following.\n\n")

            # Regression summary
            if reg_results:
                def _sig_stars(p):
                    if p < 0.001: return '***'
                    if p < 0.01: return '**'
                    if p < 0.05: return '*'
                    return ''

                def _fmt_coef(mdl, param):
                    try:
                        return f"{mdl.params[param]:.3f}{_sig_stars(mdl.pvalues[param])}"
                    except KeyError:
                        return "—"

                params_of_interest = [
                    ('baseline_target_share', 'baseline'),
                    ('C(pos_target)[T.2]', 'pos=2'),
                    ('C(pos_target)[T.3]', 'pos=3'),
                    ('abs_pos_diff_target_rival', 'abs\\_pos\\_diff'),
                    ('temperature', 'temp'),
                    ('C(prompting_mode)[T.fast]', 'fast'),
                    ('C(prompting_mode)[T.knowledge]', 'knowledge'),
                ]

                f.write("## Regression Analysis\n\n")
                f.write("Specification: 3-option trials only, with 2-option baseline target share as control.\n\n")
                f.write("Formula: `is_target ~ baseline_target_share + C(pos_target) + abs_pos_diff_target_rival + temperature + C(prompting_mode)`\n\n")

                header_cols = ['Model', 'N', 'R²'] + [label for _, label in params_of_interest]
                f.write('| ' + ' | '.join(header_cols) + ' |\n')
                f.write('| ' + ' | '.join(['---'] * len(header_cols)) + ' |\n')

                for agent_name, mdl in sorted(reg_results.items()):
                    if agent_name == '_pooled':
                        continue
                    row = [agent_name, str(int(mdl.nobs)), f"{mdl.rsquared:.3f}"]
                    row += [_fmt_coef(mdl, k) for k, _ in params_of_interest]
                    f.write('| ' + ' | '.join(row) + ' |\n')

                if '_pooled' in reg_results:
                    mdl = reg_results['_pooled']
                    row = ['**Pooled (agent FE)**', str(int(mdl.nobs)), f"{mdl.rsquared:.3f}"]
                    row += [_fmt_coef(mdl, k) for k, _ in params_of_interest]
                    f.write('| ' + ' | '.join(row) + ' |\n')

                f.write('\nSignificance: \\*p<0.05, \\*\\*p<0.01, \\*\\*\\*p<0.001\n\n')

            f.write("## Detailed Table\n")
            f.write(results.head(50).to_markdown(index=False) + "\n")
            
    print(f"Analysis complete. Results and plots saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, nargs='+', required=True, help='Path(s) to results CSV file or directories containing CSVs')
    parser.add_argument('--out-dir', type=str, default='analysis_output', help='Output directory')
    args = parser.parse_args()
    
    paths = [Path(p) for p in args.csv_path]
    analyze_data(paths, Path(args.out_dir))
