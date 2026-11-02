"""
Plot 3 - Category 2: Normalized Contact Diameter vs. Normalized Time
Strictly for Non-Bursting Regimes:
  - Case I:   Sticking (Wetting Film Deposition)
  - Case II:  Sticking + Bouncing (Vertical Oscillation on Substrate without Liftoff)
  - Case III: Leidenfrost (Vapor Film Liftoff & Rebound)
(Bursting is strictly excluded as contact diameter during fragmentation is unphysical)
Generates:
  - Individual multi-panel dynamic trajectory plots for each fluid
  - Master summary comparison
Output: Analysis/plot_3/02_contact_dia_vs_time/
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_3\data\master_summary_plot_3_4regimes.csv"
OUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\plot_3\02_contact_dia_vs_time"
os.makedirs(OUT_DIR, exist_ok=True)

HEIGHT_COLORS = {
    1.3: '#1f77b4',   # Deep Blue
    2.9: '#2ca02c',   # Emerald Green
    3.0: '#2ca02c',   # Emerald Green
    5.1: '#d62728',   # Crimson Red
    8.0: '#9467bd',   # Purple
    11.5: '#ff7f0e',  # Amber Orange
    15.6: '#8c564b',  # Saddle Brown
    20.5: '#e377c2',  # Rose Pink
    25.8: '#7f7f7f',  # Slate Grey
    32.0: '#17becf'   # Teal
}

# Fluid display labels and representative temperatures per regime
FLUID_CONFIG = {
    'pure water': {
        'label': 'Pure Water (0%)',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Puddle Deposition, T=130°C)'},
            {'regime': 'Sticking + Bouncing', 'temp': 130, 'title': 'Case II: Sticking + Bouncing (Surface Oscillation, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case III: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '20 perc acetone': {
        'label': '20% Acetone',
        'panels': [
            {'regime': 'Sticking',            'temp': 100, 'title': 'Case I: Sticking (Wetting Film, T=100°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '40 perc acetone': {
        'label': '40% Acetone',
        'panels': [
            {'regime': 'Sticking',            'temp': 100, 'title': 'Case I: Sticking (Wetting Film, T=100°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '60 perc acetone': {
        'label': '60% Acetone',
        'panels': [
            {'regime': 'Sticking',            'temp': 100, 'title': 'Case I: Sticking (Wetting Film, T=100°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '20 perc ethanol': {
        'label': '20% Ethanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Wetting Film, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '40 percent ethanol': {
        'label': '40% Ethanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Wetting Film, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '60 perc ethanol': {
        'label': '60% Ethanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Wetting Film, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '25 perc methanol': {
        'label': '25% Methanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 100, 'title': 'Case I: Sticking (Wetting Film, T=100°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '40 perc methanol': {
        'label': '40% Methanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Wetting Film, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
    '60 perc methanol': {
        'label': '60% Methanol',
        'panels': [
            {'regime': 'Sticking',            'temp': 130, 'title': 'Case I: Sticking (Wetting Film, T=130°C)'},
            {'regime': 'Leidenfrost',         'temp': 340, 'title': 'Case II: Leidenfrost Film Boiling (T=340°C)'}
        ]
    },
}

def clean_recoil_spikes(beta):
    beta = beta.copy()
    if len(beta) < 4:
        return beta
    peak_idx = int(np.argmax(beta))
    for i in range(peak_idx + 1, len(beta)):
        # If in retraction phase, beta cannot spike significantly above previous value due to shadow merging
        if beta[i] > beta[i-1] + 0.35:
            valid_next = None
            for j in range(i + 1, len(beta)):
                if beta[j] <= beta[i-1] + 0.2:
                    valid_next = beta[j]
                    break
            if valid_next is not None:
                beta[i] = (beta[i-1] + valid_next) / 2.0
            else:
                beta[i] = beta[i-1]
    return beta

def extract_curve(r):
    csv_p = os.path.join(r['Output_Folder'], 'measurements_data.csv')
    if not os.path.exists(csv_p):
        return None
    d = pd.read_csv(csv_p)
    inc = d['Contact_Status'].astype(str).str.contains('IN CONTACT').values
    fc = int(d[inc].index[0]) if inc.any() else 0
    
    fps = float(r.get('Camera_FPS_True', 4000.0))
    d0 = float(r['PreImpact_D0_px'])
    tau = float(r['Tau_inertial_ms'])
    
    start_idx = max(0, fc - 1)
    # Extract up to 25 frames post impact
    sub_d = d.iloc[start_idx:min(len(d), fc + 25)].copy()
    if len(sub_d) == 0:
        return None
        
    fn_rel = sub_d['Frame_Number'].values - sub_d['Frame_Number'].iloc[0]
    t_ms = fn_rel / fps * 1000.0
    t_star = t_ms / max(0.1, tau)
    
    beta_c = sub_d['Contact_Diameter_px'].values.copy() / d0
    if start_idx < fc:
        beta_c[0] = 0.0
    else:
        # If contact was already at frame 0, insert (0, 0) at t*=0
        dt_star = (1.0 / fps * 1000.0) / max(0.1, tau)
        t_star = np.insert(t_star + dt_star, 0, 0.0)
        beta_c = np.insert(beta_c, 0, 0.0)
        
    # For Leidenfrost cases: filter shadow bridge artifacts and retain up to liftoff
    reg = str(r['Regime_4'])
    if reg == 'Leidenfrost':
        beta_c = clean_recoil_spikes(beta_c)
        zero_indices = np.where((beta_c == 0) & (t_star > 0))[0]
        if len(zero_indices) > 0:
            first_zero = zero_indices[0]
            t_star = t_star[:first_zero + 1]
            beta_c = beta_c[:first_zero + 1]
            
    return t_star, beta_c

def plot_fluid_contact_dia(df, fl_key):
    cfg = FLUID_CONFIG[fl_key]
    panels = cfg['panels']
    num_panels = len(panels)
    
    fig, axes = plt.subplots(1, num_panels, figsize=(6.5 * num_panels, 5.8), sharey=True)
    if num_panels == 1:
        axes = [axes]
        
    title_str = (
        f"Substrate Contact Dynamics: {cfg['label']} (Strictly Non-Bursting Regimes)\n"
        r"Normalized Contact Diameter $\beta_c = D_c(t)/D_0$ vs. Dimensionless Time $t^* = (t - t_{\mathrm{contact}})/\tau$"
    )
    fig.suptitle(title_str, fontsize=13.5, fontweight='bold', y=0.98, color='#0f172a')
                 
    for idx, p_cfg in enumerate(panels):
        ax = axes[idx]
        reg = p_cfg['regime']
        temp = p_cfg['temp']
        title = p_cfg['title']
        
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10, color='#1e3a8a')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, 5.5)
        
        sub = df[(df['Fluid_Folder'] == fl_key) & (df['Temperature_C'] == temp)].sort_values('Drop_Height_cm')
        
        matching_cases = sub[sub['Regime_4'] == reg]
        height_groups = matching_cases.groupby('Drop_Height_cm', sort=True)
        
        has_lines = False
        for h_cm, group in height_groups:
            curves = []
            we_list = []
            for _, r in group.iterrows():
                res = extract_curve(r)
                if res is not None:
                    curves.append(res)
                    we_list.append(float(r['Weber_Number']))
                    
            if not curves:
                continue
                
            if len(curves) == 1:
                t_star, beta_c = curves[0]
            else:
                # Multiple runs for this height: average them on a unified time grid
                max_t = max(c[0].max() for c in curves)
                t_common = np.linspace(0.0, max_t, 35)
                interp_betas = []
                for t_s, b_c in curves:
                    fill_r = 0.0 if reg == 'Leidenfrost' else b_c[-1]
                    b_interp = np.interp(t_common, t_s, b_c, left=0.0, right=fill_r)
                    interp_betas.append(b_interp)
                t_star = t_common
                beta_c = np.mean(interp_betas, axis=0)
                
            we = np.mean(we_list)
            c = HEIGHT_COLORS.get(h_cm, '#334155')
            short_reg = "Stick" if "Sticking" in reg and "Bouncing" not in reg else ("Bounce" if "Bouncing" in reg else "LF")
            lbl = f"h = {h_cm:.1f} cm (We≈{we:.0f}) [{short_reg}]"
            
            ax.plot(t_star, beta_c, color=c, marker='o', markersize=4.2, lw=2.0, alpha=0.9, label=lbl)
            has_lines = True
                    
        if has_lines:
            ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92, edgecolor='#cbd5e1')
        else:
            ax.text(0.5, 0.5, "No Cases for this Regime at T", transform=ax.transAxes, ha='center', va='center', color='#94a3b8')
            
        ax.set_xlabel(r"Normalized Time $t^* = (t - t_{\mathrm{contact}})/\tau$", fontsize=10.5, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r"Contact Diameter Ratio $\beta_c = D_c(t)/D_0$", fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    safe_name = fl_key.replace(' ', '_').lower()
    out_p = os.path.join(OUT_DIR, f"contact_dia_dynamics_{safe_name}.png")
    for attempt in range(5):
        try:
            plt.savefig(out_p, dpi=250, bbox_inches='tight')
            break
        except Exception:
            time.sleep(0.6)
    plt.close()
    print(f"  [+] Saved contact dia dynamics: {out_p}")

def main():
    print("=" * 65)
    print("[*] GENERATING PLOT 3 CONTACT DIAMETER VS TIME (NON-BURSTING)")
    print("=" * 65)
    
    df = pd.read_csv(MASTER_CSV)
    
    for fl_key in FLUID_CONFIG.keys():
        plot_fluid_contact_dia(df, fl_key)
        
    print("\n[SUCCESS] All contact diameter dynamic plots generated successfully!")

if __name__ == "__main__":
    main()
