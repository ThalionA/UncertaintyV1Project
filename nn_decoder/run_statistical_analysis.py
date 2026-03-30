# -*- coding: utf-8 -*-
"""
Comprehensive Statistical Analysis of VR Decoder Sweep.
Runs separate multi-way ANOVAs for KL, JS, Wasserstein, and PCA Evaluation Metrics.
Explicitly tests the critical Architecture x Bin Size (Temporal Binning) interaction.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- Math Helpers ---
def calc_kl(p, q):
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

def calc_wasserstein(p, q):
    return np.sum(np.abs(np.cumsum(p, axis=1) - np.cumsum(q, axis=1)), axis=1)

def calc_js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * calc_kl(p, m) + 0.5 * calc_kl(q, m)

def calc_pca_dist(p, q, pcs=None, evar=None):
    if pcs is None or len(pcs) == 0: return np.full(p.shape[0], np.nan)
    if p.ndim == 3:
        proj_p = np.einsum('nct,kc->nkt', p, pcs)
        proj_q = np.einsum('nct,kc->nkt', q, pcs)
        evar_expand = evar[np.newaxis, :, np.newaxis]
        return np.sum(evar_expand * (proj_p - proj_q)**2, axis=1) * 100
    else:
        proj_p = np.dot(p, pcs.T)
        proj_q = np.dot(q, pcs.T)
        return np.sum(evar * (proj_p - proj_q)**2, axis=1) * 100

def load_and_melt_data():
    search_pattern = 'population_results_config_*.mat'
    file_list = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    rows = []
    print(f"Loading {len(file_list)} configurations and computing ALL evaluation metrics...")
    
    metrics = {'KL': calc_kl, 'JS': calc_js, 'Wasserstein': calc_wasserstein, 'PCA': calc_pca_dist}
    
    for f in file_list:
        try:
            mat = sio.loadmat(f, simplify_cells=True)
            cfg = mat['config']
            results = mat['results']
            
            # Extract Factors
            train_loss = cfg.get('custom_loss_func', 'KL')
            b_size = float(cfg.get('bin_size_ms', 50))
            window = cfg.get('time_window', 'full')
            lam = float(cfg.get('entropy_lambda', 0.1))
            
            for m_id, m_data in results.items():
                dist = m_data['Dist']
                pcs = dist.get('pcs', None)
                evar = dist.get('explained_var', None)
                
                for eval_name, m_func in metrics.items():
                    # Evaluate true models & Shuffles
                    if eval_name == 'PCA':
                        v_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded'], pcs, evar))
                        v_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded'], pcs, evar))
                        v_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded'], pcs, evar))
                        v_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded'], pcs, evar))
                    else:
                        v_spat = np.nanmean(m_func(dist['spat']['target'], dist['spat']['decoded']))
                        v_temp = np.nanmean(m_func(dist['temp']['target'], dist['temp']['decoded']))
                        v_spat_shf = np.nanmean(m_func(dist['spat_shf']['target'], dist['spat_shf']['decoded']))
                        v_temp_shf = np.nanmean(m_func(dist['temp_shf']['target'], dist['temp_shf']['decoded']))
                    
                    # Normalize to Shuffle
                    norm_spat = v_spat / v_spat_shf if v_spat_shf > 0 else np.nan
                    norm_temp = v_temp / v_temp_shf if v_temp_shf > 0 else np.nan
                    
                    # Append Spatial Row
                    rows.append({
                        'Mouse': m_id,
                        'Train_Loss': train_loss,
                        'Bin_Size': b_size,
                        'Window': window,
                        'Lambda': lam,
                        'Eval_Metric': eval_name,
                        'Architecture': 'Spatial',
                        'Norm_Divergence': norm_spat
                    })
                    
                    # Append Temporal Row
                    rows.append({
                        'Mouse': m_id,
                        'Train_Loss': train_loss,
                        'Bin_Size': b_size,
                        'Window': window,
                        'Lambda': lam,
                        'Eval_Metric': eval_name,
                        'Architecture': 'Temporal',
                        'Norm_Divergence': norm_temp
                    })
                
        except Exception as e:
            pass
            
    df = pd.DataFrame(rows).dropna()
    return df

def run_anovas(df):
    eval_metrics = df['Eval_Metric'].unique()
    
    for eval_metric in eval_metrics:
        print("\n" + "="*90)
        print(f"N-WAY ANOVA ON NORMALIZED {eval_metric.upper()} DIVERGENCE")
        print("="*90)
        
        subset = df[df['Eval_Metric'] == eval_metric]
        
        # Formula includes Main Effects + 2-way interactions of interest
        formula = 'Norm_Divergence ~ C(Architecture) + C(Bin_Size) + C(Train_Loss) + C(Window) + C(Lambda) + ' \
                  'C(Architecture):C(Bin_Size) + C(Architecture):C(Train_Loss) + C(Architecture):C(Window)'
                  
        try:
            model = ols(formula, data=subset).fit()
            anova_table = sm.stats.anova_lm(model, typ=2) 
            
            # Format the p-values for readability
            anova_table['Sig'] = anova_table['PR(>F)'].apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
            
            # Print the table, sorted by F-statistic (largest effect sizes at the top)
            print(anova_table.sort_values(by='F', ascending=False).to_string(float_format="%.4f"))
            print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
            
            # Highlight the critical interaction
            if 'C(Architecture):C(Bin_Size)' in anova_table.index:
                p_interaction = anova_table.loc['C(Architecture):C(Bin_Size)', 'PR(>F)']
                print("-" * 90)
                if p_interaction < 0.05:
                    print(f"[SUCCESS] The Architecture x Temporal Bin Size interaction IS significant (p = {p_interaction:.5f}).")
                    print(f"          -> SBC vs PPC differences depend heavily on the temporal integration window under {eval_metric}!")
                else:
                    print(f"[WARNING] The Architecture x Temporal Bin Size interaction is NOT significant (p = {p_interaction:.5f}).")
                    print(f"          -> The difference between Spatial/Temporal models is stable across temporal bins under {eval_metric}.")
                print("-" * 90)
        except Exception as e:
            print(f"Could not compute ANOVA for {eval_metric}: {e}")

if __name__ == "__main__":
    df = load_and_melt_data()
    if not df.empty:
        run_anovas(df)
        df.to_csv("Statistical_ANOVA_Data_Comprehensive.csv", index=False)
        print("\nExported raw data to Statistical_ANOVA_Data_Comprehensive.csv")
    else:
        print("No valid data found. Make sure the .mat files exist in this directory.")