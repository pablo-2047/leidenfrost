"""
Plot 3 - Category 1: 4-Regime Phase Maps (Weber Number vs. Temperature)
Axes:
  - X-Axis: Substrate Temperature T (°C)
  - Y-Axis: Impact Weber Number We (rho * V^2 * D0 / sigma)
4 Regimes marked with smooth, continuous colour-filled territorial areas:
  1. Sticking (Soft Blue)
  2. Sticking + Bouncing (Soft Violet / Purple)
  3. Bursting (Soft Coral / Red)
  4. Leidenfrost (Soft Mint Green)
Overlaid with crisp experimental data points.
Generates:
  - 10 Individual fluid regime maps
  - 1 Combined 10-panel master grid
Output: Analysis/plot_3/01_regime_maps_weber_vs_temp/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

MASTER_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_3\data\master_summary_plot_3_4regimes.csv"
OUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\plot_3\01_regime_maps_weber_vs_temp"
INDIV_DIR = os.path.join(OUT_DIR, "individual_fluids")
os.makedirs(INDIV_DIR, exist_ok=True)

REGIME_MAP = {
    'Sticking': 0,
    'Sticking + Bouncing': 1,
    'Bursting': 2,
    'Leidenfrost': 3
}

REGIME_COLORS_FILL = [
    '#bfdbfe',  # Sticking (Light blue)
    '#ddd6fe',  # Sticking + Bouncing (Soft violet)
    '#fecaca',  # Bursting (Light coral/red)
    '#bbf7d0'   # Leidenfrost (Light mint green)
]

REGIME_STYLES = {
    'Sticking':            {'marker': 's', 'color': '#1d4ed8', 'edge': '#0f172a', 'label': 'Sticking (Pinned Puddle)', 'size': 50},
    'Sticking + Bouncing': {'marker': 'D', 'color': '#7c3aed', 'edge': '#0f172a', 'label': 'Sticking + Bouncing (No Liftoff)', 'size': 55},
    'Bursting':            {'marker': 'X', 'color': '#dc2626', 'edge': '#0f172a', 'label': 'Bursting / Splashing / Boiling', 'size': 50},
    'Leidenfrost':         {'marker': 'o', 'color': '#059669', 'edge': '#0f172a', 'label': 'Leidenfrost (Vapor Film Liftoff)', 'size': 60}
}

FLUID_INFO = {
    'pure water':         {'label': 'Pure Water (0%)',    'Tb': 100.0},
    '20 perc acetone':    {'label': '20% Acetone',       'Tb': 75.0},
    '40 perc acetone':    {'label': '40% Acetone',       'Tb': 66.0},
    '60 perc acetone':    {'label': '60% Acetone',       'Tb': 61.0},
    '20 perc ethanol':    {'label': '20% Ethanol',       'Tb': 87.0},
    '40 percent ethanol': {'label': '40% Ethanol',       'Tb': 83.0},
    '60 perc ethanol':    {'label': '60% Ethanol',       'Tb': 81.0},
    '25 perc methanol':   {'label': '25% Methanol',      'Tb': 84.0},
    '40 perc methanol':   {'label': '40% Methanol',      'Tb': 78.0},
    '60 perc methanol':   {'label': '60% Methanol',      'Tb': 72.0},
}

def train_territorial_model(df_fl, max_we):
    """
    Fits a smooth, physics-consistent decision boundary model
    mapping (Temperature_C, Weber_Number) -> Regime code (0, 1, 2, 3).
    """
    X_raw = df_fl[['Temperature_C', 'Weber_Number']].values
    y_raw = df_fl['Regime_4'].map(REGIME_MAP).values
    
    # Feature scaling for isotropic RBF
    t_min, t_max = 80.0, 420.0
    w_min, w_max = 0.0, max(250.0, max_we * 1.05)
    
    def scale_features(X):
        T_scaled = (X[:, 0] - 250.0) / 100.0
        W_scaled = (X[:, 1] - 80.0) / 60.0
        return np.column_stack([T_scaled, W_scaled])
        
    X_train = list(X_raw)
    y_train = list(y_raw)
    
    # Physics-informed boundary anchors:
    # 1. Very high Weber (top of domain) is Bursting
    for t_a in np.linspace(80, 420, 18):
        X_train.append([t_a, w_max])
        y_train.append(2)  # Bursting
        
    # 2. Right edge at high Weber is Bursting
    for w_a in np.linspace(150, w_max, 10):
        X_train.append([420.0, w_a])
        y_train.append(2)  # Bursting
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_scaled = scale_features(X_train)
    
    # Use KNN with distance weighting for smooth territorial partitions
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X_scaled, y_train)
    
    return clf, scale_features, (t_min, t_max, w_min, w_max)

def plot_single_fluid_regime_map(df_fl, fl_key):
    meta = FLUID_INFO.get(fl_key, {'label': fl_key, 'Tb': 100.0})
    max_we = float(df_fl['Weber_Number'].max())
    
    clf, scale_func, (t_min, t_max, w_min, w_max) = train_territorial_model(df_fl, max_we)
    
    # Dense meshgrid
    ti = np.linspace(t_min, t_max, 300)
    wi = np.linspace(w_min, w_max, 300)
    Ti, Wi = np.meshgrid(ti, wi)
    grid_pts = np.column_stack([Ti.ravel(), Wi.ravel()])
    grid_scaled = scale_func(grid_pts)
    
    Zi = clf.predict(grid_scaled).reshape(Ti.shape)
    
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    
    # Filled territorial areas
    cmap_custom = ListedColormap(REGIME_COLORS_FILL)
    cf = ax.contourf(Ti, Wi, Zi, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                     colors=REGIME_COLORS_FILL, alpha=0.72)
    
    # Contour boundary lines
    ax.contour(Ti, Wi, Zi, levels=[0.5, 1.5, 2.5], colors=['#3b82f6', '#8b5cf6', '#ef4444'],
               linewidths=1.8, linestyles='--')
               
    # Boiling point reference line
    tb = meta['Tb']
    ax.axvline(tb, color='#0284c7', ls=':', lw=1.6, alpha=0.75, label=f"Boiling Point $T_b = {tb:.0f}^\circ\\mathrm{{C}}$")
    
    # Overlay experimental trials
    for reg_name, style in REGIME_STYLES.items():
        sub_reg = df_fl[df_fl['Regime_4'] == reg_name]
        if not sub_reg.empty:
            ax.scatter(
                sub_reg['Temperature_C'], sub_reg['Weber_Number'],
                marker=style['marker'], s=style['size'], c=style['color'],
                edgecolors=style['edge'], linewidths=0.9, alpha=0.95,
                label=f"{style['label']} (N = {len(sub_reg)})", zorder=5
            )
            
    ax.set_title(f"Dynamic Impact Regime Map: {meta['label']}\nTerritorial Regimes: Sticking, Sticking + Bouncing, Bursting, Leidenfrost",
                 fontsize=13.5, fontweight='bold', pad=12, color='#0f172a')
    ax.set_xlabel(r"Substrate Temperature $T$ ($^\circ\mathrm{C}$)", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"Impact Weber Number ($\mathrm{We} = \rho V^2 D_0 / \sigma$)", fontsize=12, fontweight='bold')
    
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, w_max)
    ax.grid(True, linestyle=':', alpha=0.45, color='#475569')
    
    # Two-column legend: Top right
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.92, edgecolor='#cbd5e1')
    
    plt.tight_layout()
    safe_name = fl_key.replace(' ', '_').lower()
    out_path = os.path.join(INDIV_DIR, f"regime_map_{safe_name}.png")
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {out_path}")

def plot_10panel_master_grid(df):
    """
    Creates a unified 10-panel master grid showing all 10 fluid regime maps.
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 10.5), sharex=True, sharey=False)
    axes = axes.flatten()
    
    fluids_ordered = list(FLUID_INFO.keys())
    
    for idx, fl_key in enumerate(fluids_ordered):
        ax = axes[idx]
        df_fl = df[df['Fluid_Folder'] == fl_key]
        meta = FLUID_INFO.get(fl_key, {'label': fl_key, 'Tb': 100.0})
        max_we = float(df_fl['Weber_Number'].max())
        
        clf, scale_func, (t_min, t_max, w_min, w_max) = train_territorial_model(df_fl, max_we)
        
        ti = np.linspace(t_min, t_max, 200)
        wi = np.linspace(w_min, w_max, 200)
        Ti, Wi = np.meshgrid(ti, wi)
        grid_pts = np.column_stack([Ti.ravel(), Wi.ravel()])
        grid_scaled = scale_func(grid_pts)
        Zi = clf.predict(grid_scaled).reshape(Ti.shape)
        
        # Color fill
        ax.contourf(Ti, Wi, Zi, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                    colors=REGIME_COLORS_FILL, alpha=0.70)
        ax.contour(Ti, Wi, Zi, levels=[0.5, 1.5, 2.5], colors=['#3b82f6', '#8b5cf6', '#ef4444'],
                   linewidths=1.2, linestyles='--')
                   
        # Tb line
        ax.axvline(meta['Tb'], color='#0284c7', ls=':', lw=1.2, alpha=0.7)
        
        # Points
        for reg_name, style in REGIME_STYLES.items():
            sub_reg = df_fl[df_fl['Regime_4'] == reg_name]
            if not sub_reg.empty:
                ax.scatter(
                    sub_reg['Temperature_C'], sub_reg['Weber_Number'],
                    marker=style['marker'], s=32, c=style['color'],
                    edgecolors=style['edge'], linewidths=0.6, alpha=0.92, zorder=5
                )
                
        ax.set_title(meta['label'], fontsize=12, fontweight='bold', color='#0f172a', pad=6)
        ax.set_xlim(80, 420)
        ax.set_ylim(0, w_max)
        ax.grid(True, linestyle=':', alpha=0.45)
        
        ax.set_xlabel(r"Temperature $T$ ($^\circ\mathrm{C}$)", fontsize=10, fontweight='bold')
        if idx % 5 == 0:
            ax.set_ylabel(r"Weber Number ($\mathrm{We}$)", fontsize=11, fontweight='bold')

    # Master legend at top
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1d4ed8', markeredgecolor='#0f172a', markersize=9, label='Sticking (Wetting Film)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#7c3aed', markeredgecolor='#0f172a', markersize=9, label='Sticking + Bouncing (Oscillating)'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#dc2626', markeredgecolor='#0f172a', markersize=9, label='Bursting (Atomization / Splashing)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#059669', markeredgecolor='#0f172a', markersize=10, label='Leidenfrost (Vapor Film Liftoff)'),
        plt.Line2D([0], [0], color='#0284c7', ls=':', lw=1.8, label='Boiling Point $T_b$')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.98),
               fontsize=11, framealpha=0.95, edgecolor='#cbd5e1', title='Dynamic Impact Regimes (Filled Territorial Domains)',
               title_fontproperties={'weight': 'bold', 'size': 12})
               
    fig.suptitle("Impact Regime Maps Across 10 Fluid Formulations (936 Impact Trials)\nAxes: Weber Number (Y) vs. Substrate Temperature (X) with Colour-Filled Territorial Areas",
                 fontsize=15, fontweight='bold', y=1.03, color='#0f172a')
                 
    plt.tight_layout()
    out_master = os.path.join(OUT_DIR, "regime_map_master_10_fluids_grid.png")
    plt.savefig(out_master, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved master grid: {out_master}")

def main():
    print("=" * 65)
    print("[*] GENERATING PLOT 3 REGIME MAPS (WEBER ON Y, TEMP ON X)")
    print("=" * 65)
    
    df = pd.read_csv(MASTER_CSV)
    
    for fl_key in FLUID_INFO.keys():
        df_fl = df[df['Fluid_Folder'] == fl_key]
        plot_single_fluid_regime_map(df_fl, fl_key)
        
    plot_10panel_master_grid(df)
    print("\n[SUCCESS] All regime maps generated successfully!")

if __name__ == "__main__":
    main()
