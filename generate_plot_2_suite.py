# -*- coding: utf-8 -*-
"""
Generate Complete Plot_2 Suite for Leidenfrost Droplet Impact Dynamics.
Incorporates:
  1. Manual ground-truth classifications for 20% Acetone and 20% Ethanol.
  2. Calibrated physical decision boundary for all remaining fluids:
     - Sticking (T <= 160C, flat puddle, Hcm/D0 < 0.85, no liftoff).
     - Leidenfrost Bouncing (Liftoff, Hcm/D0 >= 0.85, minor satellite shedding allowed, no massive bursting).
     - Bursting / Splashing (Massive fragmentation N >= 4, radial burst, or dynamic shattering at high We).
  3. Multi-panel spreading diameter vs. dimensionless time (strictly non-bursting).
  4. Max bounce height vs. Weber number (Leidenfrost bouncing only).
  5. Max bounce height vs. Temperature (Leidenfrost bouncing only).
  6. 3D Tri-axial rebound plots (Weber, Temperature, Max Height).
  7. Impact Regime Maps (Weber vs. Temperature) classifying Sticking, Bursting, Leidenfrost.
  8. Three novel research plots (Clanet universal collapse, Contact time invariance, Delta T_LD vs. Volatility).
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.svm import SVC

# Base directories
ANALYSIS_DIR = r"C:\Users\Pablo\Desktop\Analysis"
PLOT2_DIR = r"C:\Users\Pablo\Desktop\Analysis\plot_2"
MASTER_CSV_PATH = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"
MANUAL_CSV_PATH = r"c:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost\manual_classifications.csv"

# Subdirectories in plot_2
SUBDIRS = {
    'spreading': os.path.join(PLOT2_DIR, "01_spreading_vs_time_non_bursting"),
    'height_weber': os.path.join(PLOT2_DIR, "02_max_bounce_height_vs_weber"),
    'height_temp': os.path.join(PLOT2_DIR, "03_max_bounce_height_vs_temperature"),
    '3d_rebound': os.path.join(PLOT2_DIR, "04_3D_rebound_height_temp_weber"),
    'regime_maps': os.path.join(PLOT2_DIR, "05_impact_regime_maps_weber_vs_temp"),
    'novel': os.path.join(PLOT2_DIR, "06_novel_research_plots"),
    'four_regimes': os.path.join(PLOT2_DIR, "07_contact_dia_4_regime_array"),
}

for sdir in SUBDIRS.values():
    os.makedirs(sdir, exist_ok=True)

# Fluid physics properties
# Water density fixed at 1000.0 kg/m3 per user instruction
FLUID_PROPS = {
    'pure water': {
        'solvent': 'Water',
        'conc': 0,
        'label': 'Pure Water (0%)',
        'rho': 1000.0,
        'sigma': 0.0728,
        'mu': 0.001002,
        'T_boil_solvent': 100.0,
        'color': '#1f77b4'
    },
    '20 perc acetone': {
        'solvent': 'Acetone',
        'conc': 20,
        'label': 'Water + 20% Acetone',
        'rho': 0.2 * 784.0 + 0.8 * 1000.0,  # 956.8
        'sigma': 0.0415,
        'mu': 0.00122,
        'T_boil_solvent': 56.1,
        'color': '#ff7f0e'
    },
    '40 perc acetone': {
        'solvent': 'Acetone',
        'conc': 40,
        'label': 'Water + 40% Acetone',
        'rho': 0.4 * 784.0 + 0.6 * 1000.0,  # 913.6
        'sigma': 0.0318,
        'mu': 0.00115,
        'T_boil_solvent': 56.1,
        'color': '#d62728'
    },
    '60 perc acetone': {
        'solvent': 'Acetone',
        'conc': 60,
        'label': 'Water + 60% Acetone',
        'rho': 0.6 * 784.0 + 0.4 * 1000.0,  # 870.4
        'sigma': 0.0268,
        'mu': 0.00085,
        'T_boil_solvent': 56.1,
        'color': '#8c564b'
    },
    '20 perc ethanol': {
        'solvent': 'Ethanol',
        'conc': 20,
        'label': 'Water + 20% Ethanol',
        'rho': 0.2 * 789.0 + 0.8 * 1000.0,  # 957.8
        'sigma': 0.0405,
        'mu': 0.00178,
        'T_boil_solvent': 78.4,
        'color': '#2ca02c'
    },
    '40 percent ethanol': {
        'solvent': 'Ethanol',
        'conc': 40,
        'label': 'Water + 40% Ethanol',
        'rho': 0.4 * 789.0 + 0.6 * 1000.0,  # 915.6
        'sigma': 0.0312,
        'mu': 0.00235,
        'T_boil_solvent': 78.4,
        'color': '#17becf'
    },
    '60 perc ethanol': {
        'solvent': 'Ethanol',
        'conc': 60,
        'label': 'Water + 60% Ethanol',
        'rho': 0.6 * 789.0 + 0.4 * 1000.0,  # 873.4
        'sigma': 0.0262,
        'mu': 0.00215,
        'T_boil_solvent': 78.4,
        'color': '#bcbd22'
    },
    '25 perc methanol': {
        'solvent': 'Methanol',
        'conc': 25,
        'label': 'Water + 25% Methanol',
        'rho': 0.25 * 792.0 + 0.75 * 1000.0, # 948.0
        'sigma': 0.0462,
        'mu': 0.00142,
        'T_boil_solvent': 64.7,
        'color': '#9467bd'
    },
    '40 perc methanol': {
        'solvent': 'Methanol',
        'conc': 40,
        'label': 'Water + 40% Methanol',
        'rho': 0.4 * 792.0 + 0.6 * 1000.0,  # 916.8
        'sigma': 0.0375,
        'mu': 0.00155,
        'T_boil_solvent': 64.7,
        'color': '#e377c2'
    },
    '60 perc methanol': {
        'solvent': 'Methanol',
        'conc': 60,
        'label': 'Water + 60% Methanol',
        'rho': 0.6 * 792.0 + 0.4 * 1000.0,  # 875.2
        'sigma': 0.0295,
        'mu': 0.00135,
        'T_boil_solvent': 64.7,
        'color': '#7f7f7f'
    }
}

# Style configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['figure.autolayout'] = False

def load_and_calibrate_dataset():
    """Load master dataset and calibrate with manual classifications."""
    print("Loading dataset and merging manual ground truth...")
    df = pd.read_csv(MASTER_CSV_PATH)
    
    # Ensure numeric
    df['Drop_Height_cm'] = pd.to_numeric(df['Drop_Height_cm'], errors='coerce')
    df['Temperature_C'] = pd.to_numeric(df['Temperature_C'], errors='coerce')
    df['D0_m'] = pd.to_numeric(df['D0_m'], errors='coerce')
    
    # Fix any missing Temperature_C by parsing Case_Folder
    for idx, row in df.iterrows():
        if pd.isna(row['Temperature_C']):
            cf = str(row['Case_Folder'])
            m = re.search(r'(\d{3})', cf)
            if m:
                df.at[idx, 'Temperature_C'] = float(m.group(1))
    
    # Load manual annotations
    manual_lookup = {}
    if os.path.isfile(MANUAL_CSV_PATH):
        man_df = pd.read_csv(MANUAL_CSV_PATH)
        fluid_map = {'20% Acetone': '20 perc acetone', '20% Ethanol': '20 perc ethanol'}
        man_df['Fluid_Folder'] = man_df['fluid'].map(fluid_map)
        man_df['Drop_Height_cm'] = man_df['height_cm'].astype(float)
        man_df['Temperature_C'] = man_df['temp_c'].astype(float)
        for _, r in man_df.iterrows():
            key = (r['Fluid_Folder'], round(r['Drop_Height_cm'], 1), round(r['Temperature_C'], 0))
            reg = str(r['regime']).lower()
            if 'leidenfrost' in reg and 'stick' not in reg and 'burst' not in reg:
                manual_lookup[key] = 'Leidenfrost'
            elif 'burst' in reg:
                manual_lookup[key] = 'Bursting'
            elif 'stick' in reg:
                manual_lookup[key] = 'Sticking'
            elif 'leidenfrost' in reg:
                manual_lookup[key] = 'Leidenfrost'
            else:
                manual_lookup[key] = 'Bursting'
        print(f"Loaded {len(manual_lookup)} manual ground-truth classifications.")
        
    # Re-calculate physical properties with rho_water = 1000
    for idx, row in df.iterrows():
        fl = row['Fluid_Folder']
        p = FLUID_PROPS.get(fl, FLUID_PROPS['pure water'])
        h_m = row['Drop_Height_cm'] * 0.01
        v = np.sqrt(2.0 * 9.81 * max(0.001, h_m))
        d0 = row['D0_m']
        
        we = (p['rho'] * (v**2) * d0) / p['sigma']
        re_num = (p['rho'] * v * d0) / p['mu']
        tau_inertial = (d0 / v) * 1000.0 # ms
        tau_rayleigh = np.sqrt((p['rho'] * (d0**3)) / (8.0 * p['sigma'])) * 1000.0 # ms
        
        df.at[idx, 'Density_kg_m3'] = p['rho']
        df.at[idx, 'Surface_Tension_N_m'] = p['sigma']
        df.at[idx, 'Viscosity_Pa_s'] = p['mu']
        df.at[idx, 'Velocity_m_s'] = v
        df.at[idx, 'Weber_Number'] = we
        df.at[idx, 'Reynolds_Number'] = re_num
        df.at[idx, 'Tau_inertial_ms'] = tau_inertial
        df.at[idx, 'Tau_Rayleigh_ms'] = tau_rayleigh
        
    # Apply calibrated classification
    regimes = []
    for _, row in df.iterrows():
        key = (row['Fluid_Folder'], round(row['Drop_Height_cm'], 1), round(row['Temperature_C'], 0))
        if key in manual_lookup:
            regimes.append(manual_lookup[key])
            continue
            
        fl = row['Fluid_Folder']
        temp = row['Temperature_C']
        h_cm = row['Drop_Height_cm']
        h_rebound = row['Max_Normalized_3D_CoM_Hcm_D0']
        frags = row['Max_Fragment_Count']
        beta_sp = row['Max_Beta_Spread_Dsp_D0']
        
        # 1. Sticking: Low temp puddle, no liftoff
        if temp <= 130 and h_rebound < 0.85:
            regimes.append('Sticking')
            continue
        if temp == 160 and h_rebound < 0.80 and h_cm >= 8.0:
            regimes.append('Sticking')
            continue
            
        # 2. Leidenfrost Bouncing
        # Liftoff occurred, coherent rebound, small fragment count (1-3 minor shedding allowed)
        if temp >= 160 and h_rebound >= 0.85:
            if fl == 'pure water':
                if temp >= 220 and frags <= 4:
                    regimes.append('Leidenfrost')
                    continue
            else:
                if h_cm <= 5.1 and frags <= 4 and beta_sp < 4.2:
                    regimes.append('Leidenfrost')
                    continue
                if h_cm <= 2.9 and frags <= 5:
                    regimes.append('Leidenfrost')
                    continue
                    
        # 3. Bursting
        regimes.append('Bursting')
        
    df['Final_Regime'] = regimes
    
    # Save calibrated dataset
    out_csv = os.path.join(PLOT2_DIR, "master_summary_plot_2_rectified.csv")
    df.to_csv(out_csv, index=False)
    print(f"Calibrated dataset saved to: {out_csv}")
    print("\nRegime breakdown across 936 cases:")
    print(df['Final_Regime'].value_counts())
    
    return df

print("Setup complete.")

# Color palette for drop heights (Category 1)
HEIGHT_COLORS = {
    1.3: '#1f77b4',   # Deep Blue
    2.9: '#2ca02c',   # Emerald Green
    3.0: '#2ca02c',   # Emerald Green
    5.1: '#d62728',   # Crimson Red
    8.0: '#9467bd',   # Purple
    10.3: '#ff7f0e',  # Amber Orange
    11.5: '#ff7f0e',  # Amber Orange
    15.6: '#8c564b',  # Saddle Brown
    20.5: '#e377c2',  # Rose Pink
    25.8: '#7f7f7f',  # Slate Grey
    32.0: '#17becf'   # Teal
}

def get_height_color(h):
    # Match closest key
    for k in sorted(HEIGHT_COLORS.keys()):
        if abs(h - k) < 0.2:
            return HEIGHT_COLORS[k]
    return '#334155'


def extract_clean_contact_dia(df_case, tau, regime, we, h_cm):
    """
    Extract physical Contact Diameter ratio: beta_c(t*) = D_c(t) / D0
    - Strictly 20 frames after initial substrate contact.
    - Pre-impact touchdown: t* = 0, beta_c = 0.00
    - Next 20 frames: captures the primary impact expansion and recoil.
    - Liftoff (Leidenfrost): drops to 0.00 upon detachment.
    """
    df_c = df_case.sort_values('Time_s').reset_index(drop=True)
    c_mask = (df_c['Beta_Contact_Dc_D0'] > 0.05) | (df_c['Contact_Status'] == 'IN CONTACT')
    if not c_mask.any():
        c_mask = (df_c['Beta_Spread_Dsp_D0'] > 1.10) | (df_c['Contact_Status'] != 'PRE-IMPACT (AIRBORNE)')
    if not c_mask.any():
        return None, None
    first_c = int(c_mask.idxmax())
    i0 = max(0, first_c - 1)
    t0 = df_c.loc[i0, 'Time_s']
    
    # Strictly 20 frames after initial contact:
    # row i0 is pre-impact (airborne, beta_c = 0.0)
    # rows from first_c to first_c + 19 are exactly 20 frames of contact
    sub = df_c.loc[i0 : first_c + 19].copy().reset_index(drop=True)
    if len(sub) < 3:
        return None, None
        
    sub['t_star'] = (sub['Time_s'] - t0) / tau
    beta_c = sub['Beta_Contact_Dc_D0'].values.astype(float).copy()
    
    # Pre-impact touchdown point is exactly 0.0
    if i0 < first_c:
        beta_c[0] = 0.0
        
    # Clamp extreme specular glare / optical reflection spikes
    max_phys = max(2.5, 1.8 * (max(1.0, we) ** 0.25) + 0.8)
    beta_c = np.where(beta_c > max_phys, np.nan, beta_c)
    
    if regime == 'Leidenfrost':
        # Find primary contact liftoff:
        # After initial contact duration >= 1.5 ms, when Contact_Status is AIRBORNE or CoM rises > 1.35 D0
        lo_idx = None
        for idx in range(1, len(sub)):
            t_ms = sub.loc[idx, 'Time_ms'] - sub.loc[0, 'Time_ms']
            com = sub.loc[idx, 'Normalized_CoM_Height_Hcm_D0']
            st = sub.loc[idx, 'Contact_Status']
            if t_ms >= 1.5 and (st == 'AIRBORNE' or com >= 1.35):
                lo_idx = idx
                break
                
        if lo_idx is not None:
            contact_slice = beta_c[:lo_idx]
            contact_slice = np.where(contact_slice <= 0.05, np.nan, contact_slice)
            contact_slice = pd.Series(contact_slice).interpolate(method='linear').bfill().ffill().values
            beta_c[:lo_idx] = contact_slice
            # After liftoff, contact diameter on the substrate is strictly 0.00
            beta_c[lo_idx:] = 0.0
        else:
            beta_c = np.where(beta_c <= 0.05, np.nan, beta_c)
            beta_c = pd.Series(beta_c).interpolate(method='linear').bfill().ffill().values
    else:
        # Sticking: exclude extreme splash/shatter (We > 120 or h > 8.0 cm)
        if we > 120 or h_cm > 8.0:
            return None, None
        # Interpolate any isolated optical dropout in the positive contact segment
        if len(beta_c) > 1:
            c_pos = beta_c[1:]
            c_pos = np.where(c_pos <= 0.05, np.nan, c_pos)
            c_pos = pd.Series(c_pos).interpolate(method='linear').bfill().ffill().values
            beta_c[1:] = c_pos
            
    t_star = sub['t_star'].values
    return t_star, np.maximum(0.0, beta_c)


def generate_category_1_contact_dia(df):
    """
    Plot Category 1: Contact Diameter vs. Dimensionless time (strictly non-bursting, 20 frames after contact).
    Multi-panel subplots: Each subplot represents a distinct temperature.
    Curves show beta_c(t*) = D_contact / D0 for the first 20 frames after contact.
    """
    print("\n--- Generating Category 1: Contact Diameter vs. Dimensionless Time (First 20 Frames Post-Contact) ---")
    
    df_non_burst = df[df['Final_Regime'] != 'Bursting']
    print(f"Total non-bursting cases available: {len(df_non_burst)}")
    
    contact_dir = os.path.join(PLOT2_DIR, "01_contact_dia_vs_time_non_bursting")
    os.makedirs(contact_dir, exist_ok=True)
    
    for fl in FLUID_PROPS.keys():
        df_fl = df_non_burst[df_non_burst['Fluid_Folder'] == fl]
        fl_dir = os.path.join(ANALYSIS_DIR, fl)
        safe_name = fl.replace(' ', '_')
        label = FLUID_PROPS[fl]['label']
        
        if df_fl.empty:
            print(f"Skipping {fl} (no non-bursting cases).")
            continue
            
        temps = sorted(df_fl['Temperature_C'].unique())
        n_temps = len(temps)
        
        if n_temps == 0:
            continue
            
        # Pre-extract all valid curves for this fluid to determine optimal auto-scaled xlim
        curve_data_by_temp = {}
        all_t_stars = []
        all_betas = []
        
        for temp in temps:
            cases_temp = df_fl[df_fl['Temperature_C'] == temp].sort_values('Drop_Height_cm')
            temp_curves = []
            for _, r_case in cases_temp.iterrows():
                h_cm = r_case['Drop_Height_cm']
                h_fld = r_case['Drop_Height_Folder']
                c_fld = r_case['Case_Folder']
                we = r_case['Weber_Number']
                reg = r_case['Final_Regime']
                
                csv_path = os.path.join(fl_dir, h_fld, c_fld, 'measurements_data.csv')
                if not os.path.isfile(csv_path):
                    continue
                    
                df_case = pd.read_csv(csv_path)
                v = r_case['Velocity_m_s']
                d0 = r_case['D0_m']
                tau = d0 / max(0.01, v)
                
                t_star, beta_c = extract_clean_contact_dia(df_case, tau, reg, we, h_cm)
                if t_star is None or len(t_star) == 0:
                    continue
                    
                temp_curves.append((h_cm, we, t_star, beta_c))
                all_t_stars.extend(t_star)
                all_betas.extend(beta_c)
            curve_data_by_temp[temp] = temp_curves
            
        # Optimal auto-scaling for the 20-frame window
        max_t_limit = max(1.0, float(np.ceil(max(all_t_stars) * 2) / 2)) if all_t_stars else 2.5
        max_beta_limit = max(3.0, float(np.ceil(min(5.0, max(all_betas)) * 2) / 2)) if all_betas else 3.5
        
        n_cols = min(4, n_temps)
        n_rows = int(np.ceil(n_temps / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()
        
        fig.suptitle(f"Substrate Contact Dynamics (First 20 Frames Post-Impact)\n{label}",
                     fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                     
        for ax_i, temp in enumerate(temps):
            ax = axes[ax_i]
            temp_curves = curve_data_by_temp.get(temp, [])
            
            ax.set_title(f"Substrate Temp: T = {int(temp)}°C", fontsize=11, fontweight='bold', color='#1e3a8a', pad=8)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0, max_t_limit)
            ax.set_ylim(0, max_beta_limit)
            
            for h_cm, we, t_star, beta_c in temp_curves:
                color = get_height_color(h_cm)
                lbl = f"h = {h_cm:.1f} cm (We ≈ {we:.0f})"
                ax.plot(t_star, beta_c, color=color, lw=2.0, alpha=0.9, label=lbl)
                
            handles, labels_leg = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7.5, loc='upper right', framealpha=0.88)
            
            if ax_i % n_cols == 0:
                ax.set_ylabel(r"Contact Ratio $\beta_c = D_c(t)/D_0$", fontsize=10, fontweight='bold')
            if ax_i >= (n_rows - 1) * n_cols or ax_i == n_temps - 1:
                ax.set_xlabel(r"Dimensionless Time $t^* = (t - t_{\mathrm{contact}})/\tau$", fontsize=10, fontweight='bold')
                
        for ax_empty in axes[n_temps:]:
            ax_empty.set_visible(False)
            
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
        out_png_spread = os.path.join(SUBDIRS['spreading'], f"spreading_non_bursting_{safe_name}.png")
        out_png_contact = os.path.join(contact_dir, f"contact_dia_non_bursting_{safe_name}.png")
        plt.savefig(out_png_spread, dpi=250)
        plt.savefig(out_png_contact, dpi=250)
        plt.close()
        print(f"  [+] Saved contact diameter plot: {out_png_spread}")
        print(f"  [+] Saved contact diameter plot: {out_png_contact}")


# Alias for backward compatibility
generate_category_1_spreading = generate_category_1_contact_dia


def generate_category_2_height_vs_weber(df):
    """
    Plot Category 2: Maximum bounce height vs. Weber number (strictly Leidenfrost bouncing).
    """
    print("\n--- Generating Category 2: Max Bounce Height vs. Weber Number (Leidenfrost Only) ---")
    
    # Filter strictly to Leidenfrost bouncing cases
    df_lf = df[df['Final_Regime'] == 'Leidenfrost']
    print(f"Total Leidenfrost bouncing cases: {len(df_lf)}")
    
    # Figure 1: 10-Panel Grid across all fluids
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Temperature colormap
    temp_norm = Normalize(vmin=160, vmax=400)
    cmap = cm.plasma
    
    for i, (fl, p) in enumerate(FLUID_PROPS.items()):
        ax = axes[i]
        df_sub = df_lf[df_lf['Fluid_Folder'] == fl]
        
        ax.set_title(p['label'], fontsize=11, fontweight='bold', color='#1e3a8a', pad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 4.5)
        
        if not df_sub.empty:
            scatter = ax.scatter(
                df_sub['Weber_Number'],
                df_sub['Max_Normalized_3D_CoM_Hcm_D0'],
                c=df_sub['Temperature_C'],
                cmap=cmap,
                norm=temp_norm,
                s=55,
                edgecolors='#0f172a',
                linewidths=0.7,
                alpha=0.9,
                zorder=4
            )
            
            # Trendline if sufficient points
            if len(df_sub) >= 4:
                try:
                    z = np.polyfit(df_sub['Weber_Number'], df_sub['Max_Normalized_3D_CoM_Hcm_D0'], deg=2)
                    p_fit = np.poly1d(z)
                    we_line = np.linspace(df_sub['Weber_Number'].min(), df_sub['Weber_Number'].max(), 50)
                    ax.plot(we_line, p_fit(we_line), color=p['color'], lw=1.8, ls='--', alpha=0.8, zorder=3)
                except Exception:
                    pass
        else:
            ax.text(0.5, 0.5, "No Clean Rebound\n(Bursting/Deposition Only)",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='#94a3b8', fontsize=10, fontstyle='italic')
                    
        if i % 4 == 0:
            ax.set_ylabel(r"Bounce Height $\bar{H}_{\mathrm{cm}} = H_{\mathrm{cm}}/D_0$", fontsize=10, fontweight='bold')
        if i >= 6:
            ax.set_xlabel(r"Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=10, fontweight='bold')
            
    # Remove unused 2 axes
    axes[10].set_visible(False)
    axes[11].set_visible(False)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(cm.ScalarMappable(norm=temp_norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("Substrate Temperature T (°C)", fontsize=11, fontweight='bold')
    
    fig.suptitle("Maximum Leidenfrost Bounce Height vs. Impact Weber Number\n(Strictly Filtered to Coherent Film-Boiling Rebound Cases)",
                 fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                 
    plt.tight_layout(rect=[0.02, 0.03, 0.90, 0.94])
    out_grid = os.path.join(SUBDIRS['height_weber'], "max_bounce_height_vs_weber_10_fluids_grid.png")
    plt.savefig(out_grid, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_grid}")
    
    # Figure 2: Unified Cross-Fluid Comparison Plot
    fig_u, ax_u = plt.subplots(figsize=(11, 7))
    ax_u.set_title("Unified Rebound Trajectory vs. Weber Number Across Fluid Systems\n(Film-Boiling Leidenfrost Rebound)",
                   fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax_u.grid(True, linestyle='--', alpha=0.6)
    ax_u.set_xlim(0, 95)
    ax_u.set_ylim(0, 4.2)
    ax_u.set_xlabel(r"Impact Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=12, fontweight='bold')
    ax_u.set_ylabel(r"Maximum Bounce Height $\bar{H}_{\mathrm{cm}} = H_{\mathrm{cm}}/D_0$", fontsize=12, fontweight='bold')
    
    for fl, p in FLUID_PROPS.items():
        df_sub = df_lf[df_lf['Fluid_Folder'] == fl]
        if len(df_sub) >= 3:
            ax_u.scatter(df_sub['Weber_Number'], df_sub['Max_Normalized_3D_CoM_Hcm_D0'],
                         color=p['color'], label=p['label'], s=45, alpha=0.85, edgecolors='none')
            # Fit polynomial trendline
            try:
                z = np.polyfit(df_sub['Weber_Number'], df_sub['Max_Normalized_3D_CoM_Hcm_D0'], deg=2)
                p_fit = np.poly1d(z)
                we_line = np.linspace(df_sub['Weber_Number'].min(), df_sub['Weber_Number'].max(), 50)
                ax_u.plot(we_line, p_fit(we_line), color=p['color'], lw=2.0, alpha=0.9)
            except Exception:
                pass
                
    ax_u.legend(fontsize=9, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    out_unified = os.path.join(SUBDIRS['height_weber'], "max_bounce_height_vs_weber_unified_comparison.png")
    plt.savefig(out_unified, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_unified}")


def generate_category_3_height_vs_temp(df):
    """
    Plot Category 3: Maximum bounce height vs. Temperature.
    Clearly shows transition temperature T_LD (rebound starts, H_rebound > 0).
    Non-rebounding cases plotted as 0.
    """
    print("\n--- Generating Category 3: Max Bounce Height vs. Temperature ---")
    
    # Figure 1: 10-Panel Grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (fl, p) in enumerate(FLUID_PROPS.items()):
        ax = axes[i]
        df_fl = df[df['Fluid_Folder'] == fl]
        
        ax.set_title(p['label'], fontsize=11, fontweight='bold', color='#1e3a8a', pad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(80, 420)
        ax.set_ylim(-0.2, 4.5)
        
        # Sort drop heights
        heights = sorted(df_fl['Drop_Height_cm'].unique())
        
        for h_cm in heights:
            df_h = df_fl[df_fl['Drop_Height_cm'] == h_cm].sort_values('Temperature_C')
            
            # Map height: If Leidenfrost, take Hcm/D0; else 0
            temps = df_h['Temperature_C'].values
            h_rebound = []
            for _, r in df_h.iterrows():
                if r['Final_Regime'] == 'Leidenfrost':
                    h_rebound.append(r['Max_Normalized_3D_CoM_Hcm_D0'])
                else:
                    h_rebound.append(0.0) # No rebound
                    
            color = get_height_color(h_cm)
            we_mean = df_h['Weber_Number'].mean()
            
            # Only draw line if at least some rebound exists, or plot clean points
            ax.plot(temps, h_rebound, color=color, lw=1.5, marker='o', markersize=4,
                    alpha=0.85, label=f"h = {h_cm:.1f} cm (We≈{we_mean:.0f})")
                    
        if i % 4 == 0:
            ax.set_ylabel(r"Bounce Height $\bar{H}_{\mathrm{cm}} = H_{\mathrm{cm}}/D_0$", fontsize=10, fontweight='bold')
        if i >= 6:
            ax.set_xlabel("Substrate Temperature T (°C)", fontsize=10, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
            
    axes[10].set_visible(False)
    axes[11].set_visible(False)
    
    fig.suptitle("Maximum Bounce Height vs. Substrate Temperature Across 10 Fluid Systems\n(Non-Rebound Deposition & Shattering Regimes Mapped to 0 to Highlight T_LD Onset)",
                 fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                 
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_grid = os.path.join(SUBDIRS['height_temp'], "max_bounce_height_vs_temp_10_fluids_grid.png")
    plt.savefig(out_grid, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_grid}")
    
    # Figure 2: Unified Comparison showing T_LD transition
    fig_u, ax_u = plt.subplots(figsize=(11, 7))
    ax_u.set_title("Thermal Onset of Leidenfrost Rebound Across Binary Solvent Formulations\n(Release Height h = 1.3 cm / Low Weber Regime)",
                   fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax_u.grid(True, linestyle='--', alpha=0.6)
    ax_u.set_xlim(80, 420)
    ax_u.set_ylim(-0.2, 3.8)
    ax_u.set_xlabel("Substrate Temperature T (°C)", fontsize=12, fontweight='bold')
    ax_u.set_ylabel(r"Normalized Bounce Height $\bar{H}_{\mathrm{cm}} = H_{\mathrm{cm}}/D_0$", fontsize=12, fontweight='bold')
    
    # Compare h = 1.3 cm (cleanest low We rebound)
    for fl, p in FLUID_PROPS.items():
        df_h1 = df[(df['Fluid_Folder'] == fl) & (np.isclose(df['Drop_Height_cm'], 1.3, atol=0.2))].sort_values('Temperature_C')
        if not df_h1.empty:
            temps = df_h1['Temperature_C'].values
            h_vals = [r['Max_Normalized_3D_CoM_Hcm_D0'] if r['Final_Regime'] == 'Leidenfrost' else 0.0 for _, r in df_h1.iterrows()]
            ax_u.plot(temps, h_vals, color=p['color'], lw=2.0, marker='s', markersize=5, label=p['label'], alpha=0.9)
            
    ax_u.legend(fontsize=9, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    out_unified = os.path.join(SUBDIRS['height_temp'], "max_bounce_height_vs_temp_unified_comparison.png")
    plt.savefig(out_unified, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_unified}")


def generate_category_4_3d_rebound(df):
    """
    Plot Category 4: 3D Tri-Axial Rebound Plots (Weber Number, Temperature, Max Height).
    Includes 3D surface mesh and 2D contour maps.
    """
    print("\n--- Generating Category 4: 3D Tri-Axial Rebound Plots & 2D Contour Maps ---")
    
    # Master 4-Panel 3D Figure: Water, Acetone, Ethanol, Methanol
    fig = plt.figure(figsize=(20, 16))
    
    systems = [
        ('Pure Water', ['pure water']),
        ('Acetone Mixtures (20%, 40%, 60%)', ['20 perc acetone', '40 perc acetone', '60 perc acetone']),
        ('Ethanol Mixtures (20%, 40%, 60%)', ['20 perc ethanol', '40 percent ethanol', '60 perc ethanol']),
        ('Methanol Mixtures (25%, 40%, 60%)', ['25 perc methanol', '40 perc methanol', '60 perc methanol'])
    ]
    
    for idx, (title, fl_list) in enumerate(systems):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        df_sys = df[df['Fluid_Folder'].isin(fl_list)].dropna(subset=['Weber_Number', 'Temperature_C'])
        
        # Prepare points
        we_vals = df_sys['Weber_Number'].values
        t_vals = df_sys['Temperature_C'].values
        # Rebound height: 0 for non-rebound
        h_vals = np.array([r['Max_Normalized_3D_CoM_Hcm_D0'] if r['Final_Regime'] == 'Leidenfrost' else 0.0 for _, r in df_sys.iterrows()])
        
        # Scatter points
        sc = ax.scatter(we_vals, t_vals, h_vals, c=h_vals, cmap=cm.viridis, s=40, edgecolors='k', linewidth=0.5, alpha=0.85)
        
        # Generate smooth surface grid
        try:
            grid_we, grid_t = np.mgrid[min(we_vals):max(we_vals):30j, 100:400:30j]
            grid_h = griddata((we_vals, t_vals), h_vals, (grid_we, grid_t), method='linear')
            grid_h = np.nan_to_num(grid_h, nan=0.0)
            ax.plot_surface(grid_we, grid_t, grid_h, cmap=cm.viridis, alpha=0.35, edgecolor='none')
        except Exception:
            pass
            
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#1e3a8a')
        ax.set_xlabel('Weber Number (We)', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_zlabel(r'$\bar{H}_{\mathrm{cm}}$', fontsize=10, fontweight='bold', labelpad=8)
        ax.view_init(elev=28, azim=-60)
        
    fig.suptitle("3D Response Surface: Normalized Bounce Height vs. Impact Weber Number and Substrate Temperature",
                 fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                 
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    out_3d = os.path.join(SUBDIRS['3d_rebound'], "3D_rebound_surface_master.png")
    plt.savefig(out_3d, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_3d}")
    
    # 2D Contour Map Matrix across systems
    fig_c, axes_c = plt.subplots(2, 2, figsize=(16, 13))
    axes_c = axes_c.flatten()
    
    for idx, (title, fl_list) in enumerate(systems):
        ax = axes_c[idx]
        df_sys = df[df['Fluid_Folder'].isin(fl_list)].dropna(subset=['Weber_Number', 'Temperature_C'])
        
        we_vals = df_sys['Weber_Number'].values
        t_vals = df_sys['Temperature_C'].values
        h_vals = np.array([r['Max_Normalized_3D_CoM_Hcm_D0'] if r['Final_Regime'] == 'Leidenfrost' else 0.0 for _, r in df_sys.iterrows()])
        
        # Grid interpolation
        xi = np.linspace(5, 120, 100)
        yi = np.linspace(100, 400, 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        try:
            zi = griddata((we_vals, t_vals), h_vals, (xi_grid, yi_grid), method='cubic')
        except Exception:
            zi = griddata((we_vals, t_vals), h_vals, (xi_grid, yi_grid), method='linear')
        zi = np.clip(np.nan_to_num(zi, nan=0.0), 0, 4.5)
        
        contour = ax.contourf(xi_grid, yi_grid, zi, levels=14, cmap=cm.Blues, alpha=0.9)
        lines = ax.contour(xi_grid, yi_grid, zi, levels=6, colors='#0f172a', linewidths=0.8, alpha=0.7)
        ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Overlay actual data points
        ax.scatter(we_vals, t_vals, c='white', edgecolors='black', s=25, alpha=0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1e3a8a')
        ax.set_xlabel(r"Weber Number ($\mathrm{We}$)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Temperature T (°C)", fontsize=11, fontweight='bold')
        ax.set_xlim(5, 120)
        ax.set_ylim(100, 400)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    cbar_c = fig_c.colorbar(contour, ax=axes_c.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar_c.set_label(r"Normalized Bounce Height $\bar{H}_{\mathrm{cm}} = H_{\mathrm{cm}}/D_0$", fontsize=11, fontweight='bold')
    
    fig_c.suptitle("Iso-Height Contour Maps: Leidenfrost Rebound Islands as a Function of (We, T)",
                   fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                   
    out_contour = os.path.join(SUBDIRS['3d_rebound'], "2D_rebound_contour_maps.png")
    plt.savefig(out_contour, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_contour}")


def generate_category_5_regime_maps(df):
    """
    Plot Category 5: Impact Regime Maps (Weber Number vs. Temperature).
    Constructs continuous territorial phase boundaries (Sticking, Bursting, Leidenfrost)
    with smooth SVM decision boundaries, overlaid with experimental data points.
    """
    print("\n--- Generating Category 5: Impact Regime Maps (Weber vs. Temperature) with Territorial Areas ---")
    
    indiv_dir = os.path.join(SUBDIRS['regime_maps'], "individual_fluids")
    os.makedirs(indiv_dir, exist_ok=True)
    
    reg_map = {'Sticking': 0, 'Bursting': 1, 'Leidenfrost': 2}
    cmap_custom = ListedColormap(['#e0f2fe', '#fee2e2', '#dcfce7'])
    
    reg_styles = {
        'Leidenfrost': {'marker': 'o', 'color': '#059669', 'label': 'Leidenfrost Bouncing', 'size': 35},
        'Bursting':    {'marker': 'X', 'color': '#dc2626', 'label': 'Explosive Bursting / Splashing', 'size': 32},
        'Sticking':    {'marker': 'D', 'color': '#2563eb', 'label': 'Contact Sticking / Wetting', 'size': 28}
    }
    
    # Grid coordinates for interpolation
    xi = np.linspace(0, 180, 250)
    yi = np.linspace(80, 420, 250)
    Xi, Yi = np.meshgrid(xi, yi)
    
    we_mean, we_std = 90.0, 50.0
    t_mean, t_std = 250.0, 100.0
    grid_points = np.column_stack([(Xi.ravel() - we_mean)/we_std, (Yi.ravel() - t_mean)/t_std])
    
    # 1. 10-Panel Master Grid Figure
    fig, axes = plt.subplots(3, 4, figsize=(22, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (fl, p) in enumerate(FLUID_PROPS.items()):
        ax = axes[i]
        df_fl = df[df['Fluid_Folder'] == fl].dropna(subset=['Weber_Number', 'Temperature_C'])
        df_fl['reg_code'] = df_fl['Final_Regime'].map(reg_map)
        
        X_data = df_fl[['Weber_Number', 'Temperature_C']].values
        y_data = df_fl['reg_code'].values
        
        stick_df = df_fl[df_fl['Final_Regime'] == 'Sticking']
        lf_df = df_fl[df_fl['Final_Regime'] == 'Leidenfrost']
        
        t_stick_max = stick_df['Temperature_C'].max() if not stick_df.empty else 130.0
        t_lf_min = lf_df['Temperature_C'].min() if not lf_df.empty else 190.0
        we_lf_max = lf_df['Weber_Number'].max() if not lf_df.empty else 60.0
        
        anchors_x = []
        anchors_y = []
        
        # Bottom edge (T=80)
        for w in np.linspace(0, 180, 15):
            anchors_x.append([w, 80.0])
            anchors_y.append(0)
            
        # Left edge (We=0)
        for t in np.linspace(80, 420, 15):
            anchors_x.append([0.0, t])
            anchors_y.append(0 if t < t_lf_min else 2)
            
        # Right edge (We=180)
        for t in np.linspace(80, 420, 15):
            anchors_x.append([180.0, t])
            anchors_y.append(0 if t < t_stick_max else 1)
            
        # Top edge (T=420)
        for w in np.linspace(0, 180, 15):
            anchors_x.append([w, 420.0])
            anchors_y.append(2 if w <= we_lf_max else 1)
            
        X = np.vstack([X_data, np.array(anchors_x)])
        y = np.concatenate([y_data, np.array(anchors_y)])
        
        X_norm = np.column_stack([(X[:, 0] - we_mean)/we_std, (X[:, 1] - t_mean)/t_std])
        
        clf = SVC(kernel='rbf', C=3.0, gamma=0.8)
        clf.fit(X_norm, y)
        
        Zi = clf.predict(grid_points).reshape(Xi.shape)
        
        # Territorial Areas
        ax.contourf(Xi, Yi, Zi, levels=[-0.5, 0.6, 1.45, 2.5], cmap=cmap_custom, alpha=0.75, zorder=1)
        
        # Boundary curves
        ax.contour(Xi, Yi, Zi, levels=[0.6], colors=['#2563eb'], linewidths=1.6, linestyles='--', zorder=2)
        ax.contour(Xi, Yi, Zi, levels=[1.45], colors=['#059669'], linewidths=1.6, linestyles='--', zorder=2)
        
        # Experimental Points
        for reg_name, style in reg_styles.items():
            sub = df_fl[df_fl['Final_Regime'] == reg_name]
            if not sub.empty:
                ax.scatter(sub['Weber_Number'], sub['Temperature_C'],
                           marker=style['marker'], color=style['color'],
                           s=style['size'], label=style['label'] if i == 0 else '',
                           alpha=0.9, edgecolors='k', linewidths=0.5, zorder=4)
                           
        ax.set_title(p['label'], fontsize=12, fontweight='bold', color='#1e3a8a', pad=8)
        ax.grid(True, linestyle=':', alpha=0.4, color='#64748b')
        ax.set_xlim(0, 180)
        ax.set_ylim(80, 420)
        
        if i % 4 == 0:
            ax.set_ylabel('Substrate Temperature T (°C)', fontsize=10, fontweight='bold')
        if i >= 6:
            ax.set_xlabel(r'Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)', fontsize=10, fontweight='bold')
            
        # Also Generate High-Res Individual Fluid Regime Map
        fig_ind, ax_ind = plt.subplots(figsize=(8.5, 6.5))
        ax_ind.contourf(Xi, Yi, Zi, levels=[-0.5, 0.6, 1.45, 2.5], cmap=cmap_custom, alpha=0.75, zorder=1)
        ax_ind.contour(Xi, Yi, Zi, levels=[0.6], colors=['#2563eb'], linewidths=1.8, linestyles='--', zorder=2)
        ax_ind.contour(Xi, Yi, Zi, levels=[1.45], colors=['#059669'], linewidths=1.8, linestyles='--', zorder=2)
        
        for reg_name, style in reg_styles.items():
            sub = df_fl[df_fl['Final_Regime'] == reg_name]
            if not sub.empty:
                ax_ind.scatter(sub['Weber_Number'], sub['Temperature_C'],
                               marker=style['marker'], color=style['color'],
                               s=style['size']*1.4, label=style['label'],
                               alpha=0.9, edgecolors='k', linewidths=0.6, zorder=4)
                               
        ax_ind.set_title(f"Impact Regime Phase Diagram: {p['label']}\nTerritorial Regimes & Experimental Impact Trials",
                         fontsize=13, fontweight='bold', color='#0f172a', pad=10)
        ax_ind.set_xlabel(r"Impact Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=11, fontweight='bold')
        ax_ind.set_ylabel("Substrate Temperature T (°C)", fontsize=11, fontweight='bold')
        ax_ind.set_xlim(0, 180)
        ax_ind.set_ylim(80, 420)
        ax_ind.grid(True, linestyle=':', alpha=0.5, color='#64748b')
        ax_ind.legend(loc='upper right', fontsize=9.5, framealpha=0.95)
        
        # Subtle watermark annotations
        ax_ind.text(25, 340, "LEIDENFROST\nBOUNCING", fontsize=11, fontweight='bold', color='#047857', alpha=0.45, ha='center', va='center')
        ax_ind.text(125, 270, "EXPLOSIVE BURSTING\n/ SHATTERING", fontsize=11, fontweight='bold', color='#b91c1c', alpha=0.45, ha='center', va='center')
        ax_ind.text(80, 110, "STICKING / CONTACT WETTING", fontsize=10, fontweight='bold', color='#1d4ed8', alpha=0.45, ha='center', va='center')
        
        plt.tight_layout()
        safe_fl_name = re.sub(r'[^a-zA-Z0-9_]', '_', fl)
        indiv_out = os.path.join(indiv_dir, f"regime_map_{safe_fl_name}.png")
        fig_ind.savefig(indiv_out, dpi=250)
        plt.close(fig_ind)

    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.95, edgecolor='#cbd5e1')
    axes[10].set_visible(True)
    axes[10].axis('off')
    
    legend_elements = [
        Patch(facecolor='#e0f2fe', edgecolor='#2563eb', linestyle='--', label='Contact Sticking / Wetting Zone'),
        Patch(facecolor='#fee2e2', edgecolor='#dc2626', linestyle='--', label='Explosive Bursting / Transition Shattering'),
        Patch(facecolor='#dcfce7', edgecolor='#059669', linestyle='--', label='Leidenfrost Film Boiling / Clean Rebound')
    ]
    axes[10].legend(handles=legend_elements, loc='center', fontsize=10.5, title='Regime Phase Territories', title_fontsize=11.5, framealpha=0.95)
    
    # Subplot 11: Mechanism Guide Box
    axes[11].set_visible(True)
    axes[11].axis('off')
    guide_text = (
        "Phase Boundary Physics:\n"
        "• Sticking (Blue): T < T_boil (~130-160°C)\n"
        "  Direct liquid-solid contact & substrate wetting.\n"
        "• Leidenfrost (Green): T > T_LD & Low-to-mid We\n"
        "  Stable continuous vapor film cushion prevents contact.\n"
        "• Bursting (Red): High We OR Intermediate Boiling\n"
        "  Dynamic impact pressure disrupts vapor film,\n"
        "  causing nucleate boiling violently & droplet shattering."
    )
    axes[11].text(0.05, 0.5, guide_text, fontsize=9.5, va='center', ha='left',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.2),
                  linespacing=1.4, color='#1e293b')
                  
    fig.suptitle("Empirical Impact Regime Phase Maps with Continuous Territorial Domains\nSubstrate Temperature vs. Weber Number Across 10 Fluid Systems (936 Impact Trials)",
                 fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
                 
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_regime = os.path.join(SUBDIRS['regime_maps'], "regime_map_master_10_fluids_grid.png")
    plt.savefig(out_regime, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_regime}")
    
    # ----------------------------------------------------
    # 2. Cross-Solvent Boundary Comparison Map
    # ----------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(11, 7.5))
    ax_b.set_title("Comparative Dynamic Leidenfrost Boundary Curves Across All 10 Fluids\nSmooth Decision Boundary T_LD(We) Delineating Coherent Rebound vs. Violent Bursting",
                   fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax_b.grid(True, linestyle=':', alpha=0.5, color='#64748b')
    ax_b.set_xlim(0, 140)
    ax_b.set_ylim(80, 420)
    ax_b.set_xlabel(r"Impact Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=12, fontweight='bold')
    ax_b.set_ylabel("Substrate Temperature T (°C)", fontsize=12, fontweight='bold')
    
    # Shaded general wetting zone in the background
    ax_b.axhspan(80, 140, color='#e0f2fe', alpha=0.45, label='Universal Wetting / Sticking Domain (T < 140°C)')
    ax_b.text(70, 110, 'CONTACT DEPOSITION & REGULAR WETTING DOMAIN', fontsize=10, fontweight='bold', color='#1d4ed8', ha='center', alpha=0.6)
    
    for fl, p in FLUID_PROPS.items():
        df_fl = df[df['Fluid_Folder'] == fl].dropna(subset=['Weber_Number', 'Temperature_C'])
        df_fl['reg_code'] = df_fl['Final_Regime'].map(reg_map)
        
        X_data = df_fl[['Weber_Number', 'Temperature_C']].values
        y_data = df_fl['reg_code'].values
        
        stick_df = df_fl[df_fl['Final_Regime'] == 'Sticking']
        lf_df = df_fl[df_fl['Final_Regime'] == 'Leidenfrost']
        t_stick_max = stick_df['Temperature_C'].max() if not stick_df.empty else 130.0
        t_lf_min = lf_df['Temperature_C'].min() if not lf_df.empty else 190.0
        we_lf_max = lf_df['Weber_Number'].max() if not lf_df.empty else 60.0
        
        anchors_x, anchors_y = [], []
        for w in np.linspace(0, 180, 15):
            anchors_x.append([w, 80.0]); anchors_y.append(0)
            anchors_x.append([w, 420.0]); anchors_y.append(2 if w <= we_lf_max else 1)
        for t in np.linspace(80, 420, 15):
            anchors_x.append([0.0, t]); anchors_y.append(0 if t < t_lf_min else 2)
            anchors_x.append([180.0, t]); anchors_y.append(0 if t < t_stick_max else 1)
            
        X = np.vstack([X_data, np.array(anchors_x)])
        y = np.concatenate([y_data, np.array(anchors_y)])
        X_norm = np.column_stack([(X[:, 0] - we_mean)/we_std, (X[:, 1] - t_mean)/t_std])
        
        clf = SVC(kernel='rbf', C=3.0, gamma=0.8)
        clf.fit(X_norm, y)
        Zi = clf.predict(grid_points).reshape(Xi.shape)
        
        # Leidenfrost boundary contour (level 1.45)
        ax_b.contour(Xi, Yi, Zi, levels=[1.45], colors=[p['color']], linewidths=1.8, linestyles='--')
        ax_b.plot([], [], color=p['color'], lw=1.8, ls='--', label=p['label'])
        
    ax_b.text(25, 360, 'COHERENT LEIDENFROST\nBOUNCING ENVELOPE', fontsize=12, fontweight='bold', color='#047857', alpha=0.55, ha='center')
    ax_b.text(105, 320, 'DYNAMIC SHATTERING\n& BURSTING DOMAIN', fontsize=12, fontweight='bold', color='#b91c1c', alpha=0.55, ha='center')
    ax_b.legend(fontsize=9, loc='upper right', framealpha=0.95, edgecolor='#cbd5e1', ncol=2)
    plt.tight_layout()
    out_b = os.path.join(SUBDIRS['regime_maps'], "regime_map_cross_solvent_comparison.png")
    plt.savefig(out_b, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_b}")


def generate_category_6_novel_plots(df):
    """
    Plot Category 6: The 3 Novel Research Plots:
      1. Clanet Universal Master Collapse: beta(t*) / We^(1/4) vs. t*
      2. Contact Time Invariance Testing: tau_c / tau_Rayleigh vs. We
      3. Dynamic Leidenfrost Temperature Depression: Delta T_LD vs. Volatility
    """
    print("\n--- Generating Category 6: Novel High-Impact Research Plots ---")
    
    # ----------------------------------------------------
    # Novel Plot 1: Universal Leidenfrost Contact Diameter Scaling: beta_c(t*) / We^(1/4)
    # ----------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 7))
    ax1.set_title(r"Universal Leidenfrost Contact Diameter Scaling: $\beta_c(t^*) / \mathrm{We}^{1/4}$ vs. Dimensionless Time $t^*$" + "\n(All 10 Fluid Formulations in Coherent Film-Boiling Regime)",
                  fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 4.0)
    ax1.set_ylim(0, 1.8)
    ax1.set_xlabel(r"Dimensionless Time $t^* = (t - t_{\mathrm{contact}}) / \tau$", fontsize=12, fontweight='bold')
    ax1.set_ylabel(r"Rescaled Contact Ratio $\beta_c(t^*) / \mathrm{We}^{1/4}$", fontsize=12, fontweight='bold')
    
    # Theoretical Clanet peak line
    ax1.axhline(1.34, color='#dc2626', ls='--', lw=2.0, label=r"Universal Peak Theory: $\beta_{\max}/\mathrm{We}^{1/4} \approx 1.34$ (Clanet et al.)")
    
    df_lf = df[df['Final_Regime'] == 'Leidenfrost']
    
    # Sample curves from each fluid
    for fl, p in FLUID_PROPS.items():
        cases_fl = df_lf[df_lf['Fluid_Folder'] == fl]
        if cases_fl.empty:
            continue
            
        fl_dir = os.path.join(ANALYSIS_DIR, fl)
        
        # Take up to 3 representative cases across different heights
        sample_cases = cases_fl.drop_duplicates('Drop_Height_cm').head(3)
        
        for _, r_c in sample_cases.iterrows():
            csv_path = os.path.join(fl_dir, r_c['Drop_Height_Folder'], r_c['Case_Folder'], 'measurements_data.csv')
            if not os.path.isfile(csv_path):
                continue
                
            df_c = pd.read_csv(csv_path)
            v = r_c['Velocity_m_s']
            d0 = r_c['D0_m']
            tau = d0 / max(0.01, v)
            we = r_c['Weber_Number']
            
            t_star, beta_c = extract_clean_contact_dia(df_c, tau, 'Leidenfrost', we, r_c['Drop_Height_cm'])
            if t_star is None or len(t_star) == 0:
                continue
            beta_rescaled = beta_c / (we ** 0.25)
            
            ax1.plot(t_star, beta_rescaled, color=p['color'], lw=1.4, alpha=0.5)
            
        # Draw one colored proxy in legend
        ax1.plot([], [], color=p['color'], lw=2.0, label=p['label'])
        
    ax1.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    out_nov1 = os.path.join(SUBDIRS['novel'], "novel_01_clanet_universal_master_collapse.png")
    plt.savefig(out_nov1, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_nov1}")
    
    # ----------------------------------------------------
    # Novel Plot 2: Contact Time Invariance Testing
    # ----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    ax2.set_title(r"Contact Time Invariance Testing: Dimensionless Contact Duration $\tau_c / \tau_{\mathrm{Rayleigh}}$ vs. $\mathrm{We}$" + "\n(Evaluating Influence of Volatile Solvent Boiling on Rayleigh Oscillation Period)",
                  fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(5, 95)
    ax2.set_ylim(0.5, 4.0)
    ax2.set_xlabel(r"Impact Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=12, fontweight='bold')
    ax2.set_ylabel(r"Dimensionless Contact Time $\tau_c / \tau_{\mathrm{Rayleigh}}$", fontsize=12, fontweight='bold')
    
    # Rayleigh / Biance theoretical line: tau_c / tau_Rayleigh ≈ pi / sqrt(2) ≈ 2.22
    ax2.axhline(2.22, color='#dc2626', ls='--', lw=2.0, label=r"Rayleigh-Biance Invariance: $\tau_c \approx 2.22 \, \tau_{\mathrm{Rayleigh}}$")
    
    # Extract contact duration for all Leidenfrost cases
    for fl, p in FLUID_PROPS.items():
        cases_fl = df_lf[df_lf['Fluid_Folder'] == fl]
        if cases_fl.empty:
            continue
            
        fl_dir = os.path.join(ANALYSIS_DIR, fl)
        we_list = []
        tc_ratio_list = []
        
        for _, r_c in cases_fl.iterrows():
            csv_path = os.path.join(fl_dir, r_c['Drop_Height_Folder'], r_c['Case_Folder'], 'measurements_data.csv')
            if not os.path.isfile(csv_path):
                continue
                
            df_c = pd.read_csv(csv_path)
            # Find contact duration: frames from frame 1 to first airborne
            # where Contact_Status becomes AIRBORNE or Contact_Diameter drops to 0
            contact_mask = (df_c['Contact_Diameter_px'] > 0)
            if not contact_mask.any():
                continue
                
            # First airborne frame after contact start
            in_contact_indices = df_c[contact_mask].index
            first_contact = in_contact_indices[0]
            
            # Find subsequent liftoff
            subsequent_airborne = df_c.loc[first_contact:][df_c.loc[first_contact:]['Contact_Diameter_px'] == 0]
            if subsequent_airborne.empty:
                continue
            first_liftoff = subsequent_airborne.index[0]
            
            tc_ms = (first_liftoff - first_contact) * 0.25 # ms
            tau_rayleigh_ms = r_c['Tau_Rayleigh_ms']
            
            if 1.0 < tc_ms < 25.0 and tau_rayleigh_ms > 0:
                ratio = tc_ms / tau_rayleigh_ms
                we_list.append(r_c['Weber_Number'])
                tc_ratio_list.append(ratio)
                
        if len(we_list) >= 2:
            ax2.scatter(we_list, tc_ratio_list, color=p['color'], label=p['label'], s=45, alpha=0.85, edgecolors='k', linewidth=0.5)
            # Fit line
            if len(set(we_list)) >= 2:
                try:
                    z = np.polyfit(we_list, tc_ratio_list, 1)
                    p_fit = np.poly1d(z)
                    we_grid = np.linspace(min(we_list), max(we_list), 20)
                    ax2.plot(we_grid, p_fit(we_grid), color=p['color'], lw=1.5, ls=':')
                except Exception:
                    pass
            
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    out_nov2 = os.path.join(SUBDIRS['novel'], "novel_02_contact_time_invariance_vs_weber.png")
    plt.savefig(out_nov2, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_nov2}")
    
    # ----------------------------------------------------
    # Novel Plot 3: Leidenfrost Temperature Shift vs. Volatility
    # ----------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    fig3.suptitle("Dynamic Leidenfrost Temperature Depression: Influence of Volatile Solvent Additives",
                  fontsize=14, fontweight='bold', y=0.98, color='#0f172a')
                  
    # Left: T_LD vs Concentration for each solvent family
    ax3a.set_title(r"Dynamic Leidenfrost Temperature $T_{\mathrm{LD}}$ vs. Volumetric Concentration", fontsize=11, fontweight='bold', color='#1e3a8a')
    ax3a.grid(True, linestyle='--', alpha=0.6)
    ax3a.set_xlim(-5, 65)
    ax3a.set_ylim(140, 330)
    ax3a.set_xlabel("Solvent Concentration (% v/v)", fontsize=11, fontweight='bold')
    ax3a.set_ylabel(r"Dynamic Leidenfrost Temperature $T_{\mathrm{LD}}$ (°C)", fontsize=11, fontweight='bold')
    
    # Reference Water T_LD
    df_water_lf = df_lf[df_lf['Fluid_Folder'] == 'pure water']
    t_ld_water = df_water_lf['Temperature_C'].min() if not df_water_lf.empty else 280.0
    ax3a.axhline(t_ld_water, color='#1f77b4', ls='--', lw=1.5, label=f"Pure Water Baseline ({int(t_ld_water)}°C)")
    
    families = {
        'Acetone': {'fluids': [('20 perc acetone', 20), ('40 perc acetone', 40), ('60 perc acetone', 60)], 'color': '#d62728', 't_boil': 56.1},
        'Ethanol': {'fluids': [('20 perc ethanol', 20), ('40 percent ethanol', 40), ('60 perc ethanol', 60)], 'color': '#2ca02c', 't_boil': 78.4},
        'Methanol': {'fluids': [('25 perc methanol', 25), ('40 perc methanol', 40), ('60 perc methanol', 60)], 'color': '#9467bd', 't_boil': 64.7}
    }
    
    depressions = []
    
    for fam_name, fam_data in families.items():
        concs = [0]
        t_lds = [t_ld_water]
        
        for fl_name, c_val in fam_data['fluids']:
            df_fam_lf = df_lf[df_lf['Fluid_Folder'] == fl_name]
            if not df_fam_lf.empty:
                t_min = df_fam_lf['Temperature_C'].min()
                concs.append(c_val)
                t_lds.append(t_min)
                depressions.append({
                    'family': fam_name,
                    'conc': c_val,
                    't_boil': fam_data['t_boil'],
                    'delta_t_ld': t_ld_water - t_min
                })
                
        ax3a.plot(concs, t_lds, marker='o', lw=2.2, color=fam_data['color'], label=f"{fam_name} Series (Boiling Pt: {fam_data['t_boil']}°C)")
        
    ax3a.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    # Right: Depression Delta T_LD vs Solvent Boiling Point
    ax3b.set_title(r"Leidenfrost Temperature Depression $\Delta T_{\mathrm{LD}}$ vs. Solvent Boiling Point", fontsize=11, fontweight='bold', color='#1e3a8a')
    ax3b.grid(True, linestyle='--', alpha=0.6)
    ax3b.set_xlim(50, 85)
    ax3b.set_ylim(0, 140)
    ax3b.set_xlabel("Pure Solvent Boiling Point (°C)", fontsize=11, fontweight='bold')
    ax3b.set_ylabel(r"Temperature Depression $\Delta T_{\mathrm{LD}} = T_{\mathrm{LD,water}} - T_{\mathrm{LD,mix}}$ (°C)", fontsize=11, fontweight='bold')
    
    df_dep = pd.DataFrame(depressions)
    if not df_dep.empty:
        for conc_val in [20, 40, 60]:
            sub_c = df_dep[df_dep['conc'].isin([conc_val, 25])]
            if not sub_c.empty:
                ax3b.plot(sub_c['t_boil'], sub_c['delta_t_ld'], marker='s', lw=1.8, markersize=7,
                          label=f"~{conc_val}% Concentration")
                          
    ax3b.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_nov3 = os.path.join(SUBDIRS['novel'], "novel_03_leidenfrost_temperature_depression_vs_volatility.png")
    plt.savefig(out_nov3, dpi=250)
    plt.close()
    print(f"  [+] Saved: {out_nov3}")


def extract_raw_20frame_contact_dia(df_case, tau=None):
    """
    Extract raw, unsmoothed Contact Diameter ratio beta_c(t) = D_c(t) / D0
    for the first 20 frames after initial substrate contact.
    - Frame i0 (1 frame prior to contact): t_ms = 0.0 ms, beta_c = 0.0
    - Frames first_c .. first_c + 19: exact raw OpenCV measurements from CSV.
    Returns:
      t_ms: post-impact physical elapsed time in ms = (sub['Time_s'] - t0) * 1000.0
      beta_c: raw contact diameter ratio
    """
    df_c = df_case.sort_values('Time_s').reset_index(drop=True)
    c_mask = (df_c['Beta_Contact_Dc_D0'] > 0.05) | (df_c['Contact_Status'] == 'IN CONTACT')
    if not c_mask.any():
        c_mask = (df_c['Beta_Spread_Dsp_D0'] > 1.10) | (df_c['Contact_Status'] != 'PRE-IMPACT (AIRBORNE)')
    if not c_mask.any():
        return None, None
    first_c = int(c_mask.idxmax())
    i0 = max(0, first_c - 1)
    t0 = df_c.loc[i0, 'Time_s']

    sub = df_c.loc[i0 : first_c + 19].copy().reset_index(drop=True)
    if len(sub) < 2:
        return None, None

    t_ms = ((sub['Time_s'] - t0) * 1000.0).values.astype(float)
    beta_c = sub['Beta_Contact_Dc_D0'].values.astype(float).copy()
    if i0 < first_c:
        beta_c[0] = 0.0

    return t_ms, np.maximum(0.0, beta_c)


def generate_category_7_four_regime_array(df):
    """
    Plot Category 7: 2x2 Array of 4 Representative Regime Temperatures per Fluid.
    Each panel corresponds to one distinct regime temperature:
      1. Sticking / Wetting Deposition (130°C, or 100°C for 60% Acetone)
      2. Transition / Bounce without Leidenfrost (160°C, or 130°C for 60% Acetone)
      3. Vigorous Thermal Bursting (190°C)
      4. Leidenfrost Film Boiling Rebound (340°C)
    Within each panel, all drop release heights (h = 1.3 cm .. 32.0 cm) are plotted
    over the first 20 frames (uniform 5.0 ms physical duration) after contact using
    raw, unflattened OpenCV measurements.
    """
    print("\n--- Generating Category 7: 4-Regime Temperature Array (2x2 Grid, All Heights, First 20 Frames / 5.0 ms) ---")

    out_dir = SUBDIRS['four_regimes']
    os.makedirs(out_dir, exist_ok=True)

    checklist_rows = []

    for fl, p in FLUID_PROPS.items():
        df_fl = df[df['Fluid_Folder'] == fl].copy()
        fl_dir = os.path.join(ANALYSIS_DIR, fl)
        safe_name = fl.replace(' ', '_')
        label = p['label']

        if fl == '60 perc acetone':
            regime_panels = [
                (100, "Case I: Sticking / Wetting Deposition", "#1e40af"),
                (130, "Case II: Transition / Partial Rebound", "#0f766e"),
                (190, "Case III: Vigorous Thermal Bursting", "#b91c1c"),
                (340, "Case IV: Leidenfrost Film Boiling", "#15803d"),
            ]
        elif fl == 'pure water':
            regime_panels = [
                (130, "Case I: Sticking / Wetting Deposition", "#1e40af"),
                (160, "Case II: Bounce without Leidenfrost (Nucleate Transition)", "#0f766e"),
                (190, "Case III: Vigorous Thermal Bursting", "#b91c1c"),
                (340, "Case IV: Leidenfrost Film Boiling", "#15803d"),
            ]
        else:
            regime_panels = [
                (130, "Case I: Sticking / Wetting Deposition", "#1e40af"),
                (160, "Case II: Transition / Nucleate Onset", "#0f766e"),
                (190, "Case III: Vigorous Thermal Bursting", "#b91c1c"),
                (340, "Case IV: Leidenfrost Film Boiling", "#15803d"),
            ]

        panel_curves = {}
        all_times = []
        all_betas = []

        for temp_val, reg_title, _ in regime_panels:
            sub_t = df_fl[df_fl['Temperature_C'] == temp_val].sort_values(['Drop_Height_cm', 'Case_Folder'])
            # Deduplicate same Drop_Height_cm keeping first (S0001)
            sub_t = sub_t.drop_duplicates(subset=['Drop_Height_cm'], keep='first')
            curves = []

            for _, r_case in sub_t.iterrows():
                h_cm = float(r_case['Drop_Height_cm'])
                h_fld = str(r_case['Drop_Height_Folder'])
                c_fld = str(r_case['Case_Folder'])
                we = float(r_case['Weber_Number'])
                reg = str(r_case['Final_Regime'])

                csv_path = os.path.join(fl_dir, h_fld, c_fld, 'measurements_data.csv')
                if not os.path.isfile(csv_path):
                    continue

                df_case = pd.read_csv(csv_path)
                v = float(r_case['Velocity_m_s'])
                d0 = float(r_case['D0_m'])
                tau = d0 / max(0.01, v)

                t_ms, beta_c = extract_raw_20frame_contact_dia(df_case, tau)
                if t_ms is None or len(t_ms) == 0:
                    continue

                curves.append((h_cm, we, reg, t_ms, beta_c, h_fld, c_fld))
                all_times.extend(t_ms)
                # Filter out extreme full-frame glare (> 8.0) when computing axis y-limit
                valid_b = [b for b in beta_c if b <= 8.0]
                all_betas.extend(valid_b)

                checklist_rows.append({
                    'Fluid_Folder': fl,
                    'Fluid_Label': label,
                    'Panel_Temperature_C': int(temp_val),
                    'Panel_Regime_Case': reg_title,
                    'Drop_Height_cm': h_cm,
                    'Weber_Number': round(we, 1),
                    'Classified_Regime': reg,
                    'Drop_Height_Folder': h_fld,
                    'Case_Folder': c_fld,
                    'Max_Beta_Contact_20frames': round(float(np.max(beta_c)), 3),
                    'Zero_Dropouts_in_20frames': int(np.sum(beta_c[1:] == 0.0))
                })

            panel_curves[temp_val] = curves

        max_t_limit = 5.0
        p98_beta = float(np.percentile(all_betas, 98)) if all_betas else 4.0
        max_beta_limit = max(3.5, min(6.5, float(np.ceil(p98_beta * 2) / 2) + 0.5))

        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
        axes = axes.flatten()

        fig.suptitle(
            f"4-Regime Substrate Contact Dynamics Array (First 20 Frames / 5.0 ms Post-Impact)\n"
            f"{label} — All Drop Heights Included (Raw Unflattened Measurements)",
            fontsize=14.5, fontweight='bold', y=0.98, color='#0f172a'
        )

        for idx, (temp_val, reg_title, title_color) in enumerate(regime_panels):
            ax = axes[idx]
            curves = panel_curves.get(temp_val, [])

            ax.set_title(f"T = {int(temp_val)}°C — {reg_title}", fontsize=11.5, fontweight='bold', color=title_color, pad=8)
            ax.grid(True, linestyle='--', alpha=0.65)
            ax.set_xlim(0, max_t_limit)
            ax.set_ylim(0, max_beta_limit)

            for h_cm, we, reg, t_ms, beta_c, _, _ in curves:
                color = get_height_color(h_cm)
                short_reg = "Stick" if reg == "Sticking" else ("LF" if reg == "Leidenfrost" else "Burst")
                lbl = f"h = {h_cm:.1f} cm (We≈{we:.0f}) [{short_reg}]"
                ax.plot(
                    t_ms, beta_c,
                    color=color, lw=2.0, marker='o', markersize=3.4,
                    alpha=0.9, label=lbl
                )

            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8.0, loc='upper right', ncol=2, framealpha=0.90, edgecolor='#cbd5e1')

            if idx % 2 == 0:
                ax.set_ylabel(r"Contact Diameter Ratio $\beta_c = D_c(t)/D_0$", fontsize=11, fontweight='bold')
            if idx >= 2:
                ax.set_xlabel(r"Post-Impact Time $t - t_{\mathrm{contact}}$ (ms)", fontsize=11, fontweight='bold')

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        out_png = os.path.join(out_dir, f"contact_dia_4_regimes_{safe_name}.png")
        saved = False
        for attempt in range(3):
            try:
                if os.path.exists(out_png):
                    try: os.remove(out_png)
                    except: pass
                plt.savefig(out_png, dpi=250)
                saved = True
                break
            except Exception as e:
                import time
                time.sleep(0.5)
                if attempt == 2:
                    print(f"  [-] Warning: could not save {out_png}: {e}")
        plt.close()
        if saved:
            print(f"  [+] Saved 4-regime array plot: {out_png}")

    df_check = pd.DataFrame(checklist_rows)
    check_csv = os.path.join(out_dir, "four_regime_selected_cases.csv")
    df_check.to_csv(check_csv, index=False)
    print(f"  [+] Saved 4-regime case checklist CSV: {check_csv}")


if __name__ == "__main__":
    df_master = load_and_calibrate_dataset()
    cat = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    
    if cat in ['1', 'cat1', 'spreading', 'all']:
        generate_category_1_spreading(df_master)
    if cat in ['2', 'cat2', 'all']:
        generate_category_2_height_vs_weber(df_master)
    if cat in ['3', 'cat3', 'all']:
        generate_category_3_height_vs_temp(df_master)
    if cat in ['4', 'cat4', 'all']:
        generate_category_4_3d_rebound(df_master)
    if cat in ['5', 'cat5', 'all']:
        generate_category_5_regime_maps(df_master)
    if cat in ['6', 'cat6', 'novel', 'all']:
        generate_category_6_novel_plots(df_master)
    if cat in ['7', 'cat7', 'four_regimes', '4regimes', 'all']:
        generate_category_7_four_regime_array(df_master)
    
    print("\n=======================================================")
    print("ALL PLOT_2 SUITE GENERATIONS COMPLETED SUCCESSFULLY!")
    print(f"Saved into: {PLOT2_DIR}")
    print("=======================================================")
