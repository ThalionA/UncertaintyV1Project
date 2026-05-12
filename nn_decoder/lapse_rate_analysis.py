import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import io_coherence

def main():
    # Load IO results to get lapse rates
    animals = io_coherence.load_io_results('../ideal_observer/IOResults.mat')
    mouse_lapses = {}
    for mid, a in enumerate(animals):
        c = a['fit']['choice']
        total_lapse = float(c['gamma_r']) + float(c['delta_r'])
        mouse_lapses[mid] = total_lapse

    # Load comparison results
    df = pd.read_csv('choice_method_comparison.csv')
    
    # We want to correlate (decoder NLL - stim_mean NLL) against total_lapse
    # per architecture.
    # We will focus on the main decoders: PPC_PCA, SBC_PCA and target Q.
    # The gap is NLL(decoder) - NLL(stim_mean_Q).
    # NLL is lower is better, so a positive gap means decoder is worse than stim_mean.
    
    print("--- Lapse-rate interaction analysis ---")
    splits = df['split'].unique()
    for split in splits:
        print(f"\nSplit: {split}")
        df_split = df[df['split'] == split]
        
        # Get stim_mean_Q NLLs
        sm_nlls = df_split[df_split['method'] == 'stim_mean_Q'].set_index('mouse')['nll']
        
        methods_to_check = ['PPC_PCA', 'SBC_PCA', 'PPC_W', 'SBC_W', 'PPC_KL', 'SBC_KL', 'true_choice_SBC']
        for method in methods_to_check:
            method_df = df_split[df_split['method'] == method]
            if len(method_df) == 0:
                continue
            
            # compute gap
            method_df = method_df.set_index('mouse')
            gap = method_df['nll'] - sm_nlls
            
            lapses = [mouse_lapses[m] for m in gap.index]
            
            # Correlation
            r, p = pearsonr(lapses, gap)
            print(f"  {method:<16} gap vs lapse: r = {r:6.3f}, p = {p:6.3f}")

if __name__ == '__main__':
    main()
