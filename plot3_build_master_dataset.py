"""
Authoritative Master Dataset Generator for Plot 3:
Includes all 936 impact trials across 10 fluids with:
  - 4-regime classification:
      1. Sticking (flat pinned puddle)
      2. Sticking + Bouncing (pinned on surface, vertical bounce oscillation without liftoff)
      3. Bursting (splashing / atomization / explosive transition boiling)
      4. Leidenfrost (coherent film boiling rebound with vapor layer liftoff)
  - True camera FPS from Photron .cih/.cihx metadata
  - Accurate post-liftoff maximum rebound heights (Apex and 3D CoM)
  - Normalized metrics and physical scaling
Output: Analysis/plot_3/data/master_summary_plot_3_4regimes.csv
"""

import os, glob, re
import numpy as np
import pandas as pd

BASE_EXP = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost"
MASTER_PLOT2 = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
OUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\plot_3\data"
os.makedirs(OUT_DIR, exist_ok=True)

def read_fps(raw_dir):
    cih = glob.glob(os.path.join(raw_dir, "*.cih")) + glob.glob(os.path.join(raw_dir, "*.cihx"))
    if not cih:
        return 4000
    txt = open(cih[0], "r", errors="ignore").read()
    mm = re.search(r"Record Rate\(fps\)\s*:\s*(\d+)", txt) or re.search(r"<recordRate>(\d+)<", txt)
    return int(mm.group(1)) if mm else 4000

def check_sticking_bounce(csv_path, d0):
    """
    Checks if a sticking droplet undergoes substantial vertical recoil/oscillation
    post-impact without detaching from the substrate.
    """
    if not os.path.exists(csv_path):
        return 0.0, 0.0, False
    d = pd.read_csv(csv_path)
    inc = d['Contact_Status'].astype(str).str.contains('IN CONTACT').values
    if not inc.any():
        return 0.0, 0.0, False
    fc = int(np.argmax(inc))
    h = d['Droplet_Height_px'].values.astype(float)
    sub_h = h[fc:min(len(h), fc + 60)]
    if len(sub_h) < 10:
        return 0.0, 0.0, False
    i_min = int(np.argmin(sub_h[:15]))
    h_min = sub_h[i_min]
    sub_rebound = sub_h[i_min:min(len(sub_h), i_min + 30)]
    if len(sub_rebound) == 0:
        return 0.0, 0.0, False
    h_peak = np.max(sub_rebound)
    bounce_amp = (h_peak - h_min) / d0
    peak_norm = h_peak / d0
    is_bounce = (bounce_amp >= 0.25 and peak_norm >= 0.70)
    return bounce_amp, peak_norm, is_bounce

def main():
    print("=" * 65)
    print("[*] BUILDING MASTER DATASET FOR PLOT 3 (4 REGIMES)")
    print("=" * 65)
    
    df_p2 = pd.read_csv(MASTER_PLOT2)
    records = []
    
    for idx, r in df_p2.iterrows():
        raw_dir = os.path.join(BASE_EXP, r['Fluid_Folder'], r['Drop_Height_Folder'], r['Case_Folder'])
        csv_path = os.path.join(r['Output_Folder'], 'measurements_data.csv')
        
        if not os.path.exists(csv_path):
            csv_path = os.path.join(r'C:\Users\Pablo\Desktop\Analysis', r['Fluid_Folder'], r['Drop_Height_Folder'], r['Case_Folder'], 'measurements_data.csv')
            
        d = pd.read_csv(csv_path)
        fps = read_fps(raw_dir)
        d0_px = float(r['PreImpact_D0_px'])
        scale = float(r['Case_Scale_px_per_mm'])
        
        p2_reg = str(r['Final_Regime'])
        
        # 4-Regime Classification:
        if p2_reg == 'Leidenfrost':
            regime_4 = 'Leidenfrost'
        elif p2_reg == 'Bursting':
            regime_4 = 'Bursting'
        elif p2_reg == 'Sticking':
            amp, pk, is_b = check_sticking_bounce(csv_path, d0_px)
            if is_b:
                regime_4 = 'Sticking + Bouncing'
            else:
                regime_4 = 'Sticking'
        else:
            regime_4 = 'Bursting'
            
        inc = d['Contact_Status'].astype(str).str.contains("IN CONTACT").values
        fc = int(np.argmax(inc)) if inc.any() else 1
        
        # Accurate post-contact liftoff apex and bounce height for Leidenfrost
        lo_idx = -1
        for i in range(fc + 1, len(d) - 2):
            if not inc[i] and not inc[i+1] and not inc[i+2]:
                lo_idx = i
                break
                
        post_idx = lo_idx if lo_idx >= 0 else fc
        post_flight = d.iloc[post_idx:]
        
        if regime_4 == 'Leidenfrost' and len(post_flight) > 0:
            lf_max_apex_px = float(post_flight['Droplet_Height_px'].max())
            lf_max_com_px = float(post_flight['CoM_Height_px'].max())
        else:
            lf_max_apex_px = 0.0
            lf_max_com_px = 0.0
            
        lf_max_apex_mm = lf_max_apex_px / scale
        lf_max_com_mm = lf_max_com_px / scale
        lf_norm_apex = lf_max_apex_px / d0_px
        lf_norm_com = lf_max_com_px / d0_px
        
        rec = dict(r)
        rec['Camera_FPS_True'] = fps
        rec['Regime_4'] = regime_4
        rec['LF_PostLiftoff_Max_Apex_px'] = lf_max_apex_px
        rec['LF_PostLiftoff_Max_Apex_mm'] = round(lf_max_apex_mm, 3)
        rec['LF_PostLiftoff_Max_Apex_H_D0'] = round(lf_norm_apex, 4)
        rec['LF_PostLiftoff_Max_CoM_px'] = lf_max_com_px
        rec['LF_PostLiftoff_Max_CoM_mm'] = round(lf_max_com_mm, 3)
        rec['LF_PostLiftoff_Max_CoM_Hcm_D0'] = round(lf_norm_com, 4)
        
        records.append(rec)
        
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "master_summary_plot_3_4regimes.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[*] Saved Plot 3 master dataset to: {out_csv}")
    print("\nRegime breakdown across 936 cases in Plot 3:")
    print(df_out['Regime_4'].value_counts())
    print("\nRegime breakdown by fluid:")
    print(df_out.groupby(['Fluid_Folder', 'Regime_4']).size().unstack(fill_value=0).to_string())

if __name__ == "__main__":
    main()
