"""
Publication-Quality Plotting Suite for Leidenfrost Maximum Rebound Height
Visualizes:
  - Weber Number (We)
  - Maximum Bounce Height (Normalized H_max/D0 and physical mm)
  - Substrate Temperature (T in °C)
Across all 10 fluid systems for Leidenfrost film-boiling cases only.
Generates:
  1. 10-panel grid: Weber Number (X) vs. Max Height (Y) with Temperature lines
  2. 10-panel grid: Temperature (X) vs. Max Height (Y) with Weber/Height lines
  3. Physical millimeter versions of both grids
  4. Individual high-resolution publication figures for all 10 fluids
  5. Master unified cross-fluid overview plot
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

MASTER_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
OUTPUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\plots\leidenfrost_max_height"
INDIVIDUAL_DIR = os.path.join(OUTPUT_DIR, "individual_fluids")
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

# Fluid metadata for clear, clean presentation
FLUID_INFO = {
    'pure water': {'label': 'Pure Water (0%)', 'color': '#1d4ed8'},
    '20 perc acetone': {'label': '20% Acetone', 'color': '#0284c7'},
    '40 perc acetone': {'label': '40% Acetone', 'color': '#0d9488'},
    '60 perc acetone': {'label': '60% Acetone', 'color': '#059669'},
    '20 perc ethanol': {'label': '20% Ethanol', 'color': '#d97706'},
    '40 percent ethanol': {'label': '40% Ethanol', 'color': '#ea580c'},
    '60 perc ethanol': {'label': '60% Ethanol', 'color': '#dc2626'},
    '25 perc methanol': {'label': '25% Methanol', 'color': '#7c3aed'},
    '40 perc methanol': {'label': '40% Methanol', 'color': '#9333ea'},
    '60 perc methanol': {'label': '60% Methanol', 'color': '#c026d3'},
}

# Distinct styling for temperatures
TEMPS_LIST = [160, 190, 220, 250, 280, 310, 340, 370, 400]
TEMP_PALETTE = {
    160: '#64748b',
    190: '#0284c7',
    220: '#059669',
    250: '#84cc16',
    280: '#eab308',
    310: '#f97316',
    340: '#ef4444',
    370: '#b91c1c',
    400: '#7f1d1d'
}
TEMP_MARKERS = {
    160: 'o',
    190: 's',
    220: '^',
    250: 'v',
    280: 'D',
    310: 'P',
    340: '*',
    370: 'X',
    400: 'p'
}

# Height colors for Option B
HEIGHT_PALETTE = {
    1.3: '#2563eb',
    2.9: '#059669',
    3.0: '#059669',
    5.1: '#d97706',
    8.0: '#dc2626',
    11.5: '#9333ea',
    15.6: '#475569'
}

def load_data():
    df = pd.read_csv(MASTER_CSV)
    lf = df[df['Final_Regime'] == 'Leidenfrost'].copy()
    lf['Temperature_C'] = lf['Temperature_C'].astype(int)
    return lf

def plot_grid_weber_vs_height(lf, height_col='Max_Normalized_Height_H_D0', is_norm=True):
    """
    10-panel grid:
    X = Weber Number (We)
    Y = Max Height
    Lines = Different Temperatures
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5), sharey=True)
    axes = axes.flatten()

    y_label = r"Max Normalized Height $H_{\mathrm{max}}/D_0$" if is_norm else r"Max Bounce Height $H_{\mathrm{max}}$ (mm)"
    y_max = 7.0 if is_norm else 22.0

    fluids_ordered = list(FLUID_INFO.keys())

    for idx, fl in enumerate(fluids_ordered):
        ax = axes[idx]
        sub = lf[lf['Fluid_Folder'] == fl]
        fl_meta = FLUID_INFO.get(fl, {'label': fl, 'color': '#000000'})

        ax.set_title(fl_meta['label'], fontsize=12, fontweight='bold', color='#0f172a', pad=8)
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.set_ylim(0, y_max)
        ax.set_xlim(0, max(120, float(sub['Weber_Number'].max() * 1.1)) if not sub.empty else 100)

        if sub.empty:
            ax.text(0.5, 0.5, "No Leidenfrost Cases", transform=ax.transAxes,
                    ha='center', va='center', color='#94a3b8', fontstyle='italic')
            continue

        available_temps = sorted(sub['Temperature_C'].unique())
        for t in available_temps:
            sub_t = sub[sub['Temperature_C'] == t].sort_values('Weber_Number')
            if not sub_t.empty:
                c = TEMP_PALETTE.get(t, '#64748b')
                m = TEMP_MARKERS.get(t, 'o')
                ax.plot(
                    sub_t['Weber_Number'], sub_t[height_col],
                    color=c, marker=m, markersize=5.5, lw=1.6, alpha=0.9,
                    label=f"{t}°C"
                )

        ax.set_xlabel(r"Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=10, fontweight='bold')
        if idx % 5 == 0:
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')

    # Shared Legend at top
    # Build unique legend handles from all temperatures
    handles = []
    labels = []
    for t in TEMPS_LIST:
        if t in lf['Temperature_C'].values:
            c = TEMP_PALETTE.get(t, '#64748b')
            m = TEMP_MARKERS.get(t, 'o')
            line, = plt.plot([], [], color=c, marker=m, markersize=6, lw=1.6, label=f"{t}°C")
            handles.append(line)
            labels.append(f"{t}°C")

    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.98), fontsize=10.5, framealpha=0.95,
               edgecolor='#cbd5e1', title='Substrate Temperature (°C)', title_fontproperties={'weight': 'bold'})

    unit_title = "Normalized $H_{\mathrm{max}}/D_0$" if is_norm else "Physical $H_{\mathrm{max}}$ (mm)"
    fig.suptitle(f"Leidenfrost Film Boiling: Rebound Height ({unit_title}) vs. Impact Weber Number\n(Different Temperatures as Distinct Lines Across All 10 Fluids)",
                 fontsize=14.5, fontweight='bold', y=1.03, color='#0f172a')

    plt.tight_layout()
    suffix = "norm" if is_norm else "mm"
    out_file = os.path.join(OUTPUT_DIR, f"max_bounce_height_vs_weber_by_temperature_grid_{suffix}.png")
    plt.savefig(out_file, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved grid: {out_file}")
    return out_file

def plot_grid_temp_vs_height(lf, height_col='Max_Normalized_Height_H_D0', is_norm=True):
    """
    10-panel grid (The continuous-temperature perspective):
    X = Temperature (°C)
    Y = Max Height
    Lines = Different Drop Heights / Weber Numbers
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5), sharey=True)
    axes = axes.flatten()

    y_label = r"Max Normalized Height $H_{\mathrm{max}}/D_0$" if is_norm else r"Max Bounce Height $H_{\mathrm{max}}$ (mm)"
    y_max = 7.0 if is_norm else 22.0

    fluids_ordered = list(FLUID_INFO.keys())

    for idx, fl in enumerate(fluids_ordered):
        ax = axes[idx]
        sub = lf[lf['Fluid_Folder'] == fl]
        fl_meta = FLUID_INFO.get(fl, {'label': fl, 'color': '#000000'})

        ax.set_title(fl_meta['label'], fontsize=12, fontweight='bold', color='#0f172a', pad=8)
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.set_ylim(0, y_max)
        ax.set_xlim(150, 415)

        if sub.empty:
            ax.text(0.5, 0.5, "No Leidenfrost Cases", transform=ax.transAxes,
                    ha='center', va='center', color='#94a3b8', fontstyle='italic')
            continue

        available_heights = sorted(sub['Drop_Height_cm'].unique())
        for h in available_heights:
            sub_h = sub[sub['Drop_Height_cm'] == h].sort_values('Temperature_C')
            we_mean = sub_h['Weber_Number'].mean()
            c = HEIGHT_PALETTE.get(h, '#334155')
            ax.plot(
                sub_h['Temperature_C'], sub_h[height_col],
                color=c, marker='o', markersize=5.0, lw=1.8, alpha=0.9,
                label=f"h={h}cm (We≈{we_mean:.0f})"
            )

        ax.set_xlabel("Substrate Temperature (°C)", fontsize=10, fontweight='bold')
        if idx % 5 == 0:
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.88, edgecolor='#cbd5e1')

    unit_title = "Normalized $H_{\mathrm{max}}/D_0$" if is_norm else "Physical $H_{\mathrm{max}}$ (mm)"
    fig.suptitle(f"Leidenfrost Film Boiling: Rebound Height ({unit_title}) vs. Substrate Temperature\n(Different Weber Numbers / Drop Heights as Continuous Lines Across All 10 Fluids)",
                 fontsize=14.5, fontweight='bold', y=0.98, color='#0f172a')

    plt.tight_layout()
    suffix = "norm" if is_norm else "mm"
    out_file = os.path.join(OUTPUT_DIR, f"max_bounce_height_vs_temperature_by_weber_grid_{suffix}.png")
    plt.savefig(out_file, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved grid: {out_file}")
    return out_file

def plot_individual_fluid_plots(lf):
    """
    Generates clean, simple, standalone figures for every fluid:
    Side-by-side or individual high-resolution plots.
    """
    for fl, meta in FLUID_INFO.items():
        sub = lf[lf['Fluid_Folder'] == fl]
        if sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left: Weber on X, Max Height on Y, Temperature lines
        ax1.set_title(f"{meta['label']} — Bounce Height vs. Weber Number", fontsize=12, fontweight='bold', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(0, max(5.0, float(sub['Max_Normalized_Height_H_D0'].max() * 1.25)))

        temps = sorted(sub['Temperature_C'].unique())
        for t in temps:
            sub_t = sub[sub['Temperature_C'] == t].sort_values('Weber_Number')
            c = TEMP_PALETTE.get(t, '#64748b')
            m = TEMP_MARKERS.get(t, 'o')
            ax1.plot(
                sub_t['Weber_Number'], sub_t['Max_Normalized_Height_H_D0'],
                color=c, marker=m, markersize=6.5, lw=1.8, label=f"T = {t}°C"
            )

        ax1.set_xlabel(r"Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=11, fontweight='bold')
        ax1.set_ylabel(r"Maximum Normalized Height $H_{\mathrm{max}}/D_0$", fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.9, edgecolor='#cbd5e1')

        # Right: Temperature on X, Max Height on Y, Height lines
        ax2.set_title(f"{meta['label']} — Bounce Height vs. Temperature", fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(0, max(5.0, float(sub['Max_Normalized_Height_H_D0'].max() * 1.25)))

        heights = sorted(sub['Drop_Height_cm'].unique())
        for h in heights:
            sub_h = sub[sub['Drop_Height_cm'] == h].sort_values('Temperature_C')
            we_mean = sub_h['Weber_Number'].mean()
            c = HEIGHT_PALETTE.get(h, '#334155')
            ax2.plot(
                sub_h['Temperature_C'], sub_h['Max_Normalized_Height_H_D0'],
                color=c, marker='o', markersize=6.5, lw=2.0, label=f"h = {h:.1f} cm (We≈{we_mean:.0f})"
            )

        ax2.set_xlabel("Substrate Temperature (°C)", fontsize=11, fontweight='bold')
        ax2.set_ylabel(r"Maximum Normalized Height $H_{\mathrm{max}}/D_0$", fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9.5, loc='upper left', framealpha=0.9, edgecolor='#cbd5e1')

        fig.suptitle(f"Leidenfrost Rebound Dynamics: {meta['label']} (N = {len(sub)} Trials)",
                     fontsize=14, fontweight='bold', y=0.98, color='#0f172a')

        plt.tight_layout()
        safe_name = fl.replace(' ', '_').lower()
        out_f = os.path.join(INDIVIDUAL_DIR, f"leidenfrost_max_height_{safe_name}.png")
        plt.savefig(out_f, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved individual plot: {out_f}")

def plot_unified_clean_master(lf):
    """
    Creates a single, highly readable master plot showing all 10 fluids.
    To keep it clean and simple as requested:
    - Depicts Weber Number on X
    - Maximum Height H_max/D0 on Y
    - Temperature as a smooth, intuitive visual color gradient with clean markers
    - Subtle fluid grouping or trendlines
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    temp_norm = Normalize(vmin=190, vmax=400)
    cmap = cm.plasma

    # Scatter of all points
    sc = ax.scatter(
        lf['Weber_Number'],
        lf['Max_Normalized_Height_H_D0'],
        c=lf['Temperature_C'],
        cmap=cmap,
        norm=temp_norm,
        s=65,
        edgecolors='#0f172a',
        linewidths=0.7,
        alpha=0.88,
        zorder=4
    )

    # Clean trendline across all fluids for Leidenfrost rebound
    z = np.polyfit(lf['Weber_Number'], lf['Max_Normalized_Height_H_D0'], deg=1)
    p_fit = np.poly1d(z)
    we_grid = np.linspace(lf['Weber_Number'].min(), lf['Weber_Number'].max(), 100)
    ax.plot(we_grid, p_fit(we_grid), color='#1e293b', lw=2.2, ls='--',
            label=f"Cross-Fluid Mean Trend: $H/D_0 = {z[0]:.3f}\\mathrm{{We}} + {z[1]:.2f}$", zorder=3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Substrate Temperature T (°C)", fontsize=11, fontweight='bold')

    ax.set_title("Unified Leidenfrost Rebound Scaling Across All 10 Fluids (182 Trials)\nMaximum Normalized Height vs. Weber Number and Temperature",
                 fontsize=13.5, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel(r"Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"Maximum Normalized Height $H_{\mathrm{max}}/D_0$", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 7.0)
    ax.legend(fontsize=10.5, loc='upper left', framealpha=0.92, edgecolor='#cbd5e1')

    plt.tight_layout()
    out_master = os.path.join(OUTPUT_DIR, "unified_leidenfrost_height_vs_weber_temp_master.png")
    plt.savefig(out_master, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved unified master: {out_master}")

def main():
    print("=" * 65)
    print("[*] GENERATING LEIDENFROST MAXIMUM HEIGHT PLOTS FOR ALL FLUIDS")
    print("=" * 65)

    lf = load_data()
    print(f"[*] Loaded {len(lf)} Leidenfrost rebound cases across {lf['Fluid_Folder'].nunique()} fluids.")

    # 1. 10-panel Grid: Weber vs. Height with Temperature Lines (Normalized H/D0)
    plot_grid_weber_vs_height(lf, height_col='Max_Normalized_Height_H_D0', is_norm=True)

    # 2. 10-panel Grid: Weber vs. Height with Temperature Lines (Physical mm)
    plot_grid_weber_vs_height(lf, height_col='Max_Droplet_Height_mm', is_norm=False)

    # 3. 10-panel Grid: Temperature vs. Height with Weber Lines (Normalized H/D0)
    plot_grid_temp_vs_height(lf, height_col='Max_Normalized_Height_H_D0', is_norm=True)

    # 4. 10-panel Grid: Temperature vs. Height with Weber Lines (Physical mm)
    plot_grid_temp_vs_height(lf, height_col='Max_Droplet_Height_mm', is_norm=False)

    # 5. Individual standalone plots for each of the 10 fluids
    plot_individual_fluid_plots(lf)

    # 6. Unified cross-fluid master plot
    plot_unified_clean_master(lf)

    print("\n[SUCCESS] All Leidenfrost maximum height plots generated successfully!")

if __name__ == "__main__":
    main()
