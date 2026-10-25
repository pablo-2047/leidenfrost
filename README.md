# Leidenfrost Droplet Impact Dynamics: Experimental Research Compendium

[![GitHub Pages](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-1e3a8a?style=for-the-badge&logo=github)](https://pablo-2047.github.io/leidenfrost/)
[![Dataset](https://img.shields.io/badge/Dataset-CSV_Compiled-059669?style=for-the-badge&logo=microsoftexcel)](./master_summary_with_physics.csv)
[![High Speed Imaging](https://img.shields.io/badge/Camera-4000_FPS-0284c7?style=for-the-badge)](https://pablo-2047.github.io/leidenfrost/)
[![Experimental Cases](https://img.shields.io/badge/Runs_Analyzed-936_Cases-475569?style=for-the-badge)](https://pablo-2047.github.io/leidenfrost/)

---

## 📌 Executive Overview

This repository hosts the experimental data, physics-based normalization models, and comprehensive publication-grade visualization dashboard for a large-scale high-speed investigation (**936 discrete experimental runs** at **4,000 frames per second**) examining binary droplet impact dynamics onto superheated polished aluminum substrates across the nucleate boiling, transition boiling, and film-boiling dynamic Leidenfrost regimes ($100^\circ\text{C}$ to $400^\circ\text{C}$).

The study quantifies the influence of volatile solvent additives (**Acetone, Ethanol, Methanol** across concentrations $0\%$ to $60\%$ by volume) on:
1. **Hydrodynamic Spreading Dynamics & Capillary Recoil:** $\beta(t) = D_{\text{spread}}(t)/D_0$ versus dimensionless time $t^*$.
2. **Apex vs. 3D Volume Center-of-Mass Trajectories:** $\bar{H}(t^*) = H_{\text{apex}}/D_0$ and $\bar{H}_{\text{cm}}(t^*) = H_{\text{cm}}/D_0$ (eliminating jet tip artifacts).
3. **Weber Number Scaling:** Impact energy conversion into coherent vertical rebound $H_{\max}/D_0$ across $\text{We} \in [4, 150]$.
4. **Solutal Marangoni & Explosive Nucleation:** Phase transitions, explosive droplet fragmentation counts ($N_{\text{fragments}}$), and dynamic Leidenfrost suppression.
5. **Universal Scaling Laws:** Verification of the Clanet / Biance capillary-inertial scaling law ($\beta_{\max} \approx 1.34 \cdot \text{We}^{1/4}$) across all 10 fluid systems.

---

## 🔬 Experimental Matrix & Fluids

All experiments were conducted with droplet equivalent spherical diameters $D_0 \approx 2.0\text{–}3.2\text{ mm}$ impacting from release heights $h \in \{1.3\text{ cm}, 5.1\text{ cm}, 10.3\text{ cm}, 15.6\text{ cm}\}$ over 10 substrate temperatures ($100^\circ\text{C}$ to $400^\circ\text{C}$).

| Fluid System | Volumetric Concentration | Impact Trials | Temperature Range | Release Heights |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Deionized Water** | $0\%$ Additive | 100 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Acetone** | $20\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Acetone** | $40\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Acetone** | $60\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Ethanol** | $20\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Ethanol** | $40\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Ethanol** | $60\%$ (v/v) | 88 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Methanol** | $25\%$ (v/v) | 111 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Methanol** | $40\%$ (v/v) | 99 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Water + Methanol** | $60\%$ (v/v) | 98 runs | $100^\circ\text{C} – 400^\circ\text{C}$ | $1.3, 5.1, 10.3, 15.6\text{ cm}$ |
| **Total** | — | **936 runs** | **209,425 High-Speed Frames** | — |

---

## 📐 Mathematical Formulations & Normalization

All raw temporal and spatial metrics are normalized using the following continuum mechanics relations:

1. **Free-Fall Velocity & Capillary-Inertial Time Scale:**
   $$V = \sqrt{2gh}, \qquad \tau = \frac{D_0}{V}$$

2. **Dimensionless Time Coordinate:**
   $$t^* = \frac{t - t_{\text{contact}}}{\tau}$$
   where $t_{\text{contact}}$ is the exact frame immediately preceding liquid-solid impact at 4,000 fps.

3. **Dimensionless Spreading & Dynamic Contact Factors:**
   $$\beta(t) = \frac{D_{\text{spread}}(t)}{D_0}, \qquad \beta_c(t) = \frac{D_{\text{contact}}(t)}{D_0}$$

4. **Normalized 3D Volume Center of Mass:**
   $$\bar{H}_{\text{cm}}(t) = \frac{H_{\text{cm}}(t)}{D_0}, \qquad z_{\text{cm}} = \frac{\pi \sum y \cdot r^2(y) \Delta y}{\pi \sum r^2(y) \Delta y}$$
   Discretized as axisymmetric disks across the binarized droplet mask to eliminate bias from high-velocity vertical jet tips and micro-splatter.

5. **Dimensionless Hydrodynamic Groups:**
   $$\text{We} = \frac{\rho V^2 D_0}{\sigma}, \qquad \text{Re} = \frac{\rho V D_0}{\mu}, \qquad \text{Oh} = \frac{\mu}{\sqrt{\rho \sigma D_0}}$$

6. **Restitution Coefficient (Droplet Elasticity):**
   $$e = \sqrt{\frac{H_{\text{rebound}}}{h_{\text{drop}}}} = \frac{V_{\text{rebound}}}{V_{\text{impact}}}$$

7. **Universal Clanet / Biance Leidenfrost Scaling Law:**
   $$\beta_{\max} \approx 1.34 \cdot \text{We}^{1/4}$$

---

## 🗂 Repository Structure

```
leidenfrost/
├── index.html                                        <- Journal-grade interactive research portal
├── master_summary_with_physics.csv                   <- Enriched dataset across all 936 cases
├── README.md                                         <- Scientific documentation & metadata
│
├── 01_height_vs_normalized_time/                     <- Category 1: Dimensionless height dynamics
│   ├── height_vs_time_pure_water.png
│   ├── height_vs_time_20_perc_acetone.png
│   ├── height_vs_time_40_perc_acetone.png
│   ├── height_vs_time_60_perc_acetone.png
│   ├── height_vs_time_20_perc_ethanol.png
│   ├── height_vs_time_40_percent_ethanol.png
│   ├── height_vs_time_60_perc_ethanol.png
│   ├── height_vs_time_25_perc_methanol.png
│   ├── height_vs_time_40_perc_methanol.png
│   └── height_vs_time_60_perc_methanol.png
│
├── 02_spreading_vs_normalized_time/                  <- Category 2: Hydrodynamic spreading & recoil
│   ├── spreading_vs_time_pure_water.png
│   ├── spreading_vs_time_20_perc_acetone.png
│   ├── spreading_vs_time_40_perc_acetone.png
│   ├── spreading_vs_time_60_perc_acetone.png
│   ├── spreading_vs_time_20_perc_ethanol.png
│   ├── spreading_vs_time_40_percent_ethanol.png
│   ├── spreading_vs_time_60_perc_ethanol.png
│   ├── spreading_vs_time_25_perc_methanol.png
│   ├── spreading_vs_time_40_perc_methanol.png
│   └── spreading_vs_time_60_perc_methanol.png
│
├── 03_max_height_vs_weber_number/                    <- Category 3: Restitution & Weber scaling
│   ├── max_height_vs_weber_all_10_fluids.png         (10-panel matrix across 100°C–400°C)
│   └── max_height_vs_weber_leidenfrost_comparison.png (Direct comparison at 340°C & 370°C)
│
├── 04_concentration_effects/                         <- Category 4: Additive concentration series
│   ├── concentration_effect_acetone.png              (4-panel: beta, H_cm, beta_c, N_frag)
│   ├── concentration_effect_ethanol.png              (4-panel: beta, H_cm, beta_c, N_frag)
│   ├── concentration_effect_methanol.png             (4-panel: beta, H_cm, beta_c, N_frag)
│   └── concentration_cross_solvent_comparison.png    (Direct cross-solvent comparison)
│
└── 05_scaling_and_regime_maps/                       <- Category 5: Theoretical verification & phase maps
    ├── scaling_law_beta_max_vs_weber.png             (Clanet We^(1/4) validation)
    ├── regime_maps_weber_vs_temperature.png          (Phase diagrams: Deposition vs Bursting vs Leidenfrost)
    └── restitution_coefficient_vs_temperature.png   (Elasticity e vs T transition)
```

---

## 🚀 How to View & Use

### Option 1: Live Interactive Dashboard (Recommended)
Visit the live GitHub Pages site:
👉 **[https://pablo-2047.github.io/leidenfrost/](https://pablo-2047.github.io/leidenfrost/)**

*(To enable GitHub Pages in your repository: Go to **Settings** → **Pages** → under **Build and deployment**, select **Deploy from a branch**, choose **main** branch and **/ (root)** folder, then click **Save**).*

### Option 2: Offline Local Inspection
1. Clone this repository:
   ```bash
   git clone https://github.com/pablo-2047/leidenfrost.git
   ```
2. Double-click `index.html` to open the complete interactive dashboard in any modern web browser. All image assets and stylesheets load locally via relative paths without requiring any internet connection or web server.

---

## 📊 Summary of Scientific Findings

1. **Vapor Cushion Frictionlessness (Clanet Scaling):**
   In the fully developed film boiling regime ($T \ge 310^\circ\text{C}$), the maximum spreading ratio $\beta_{\max}$ collapses cleanly onto the universal power law $\beta_{\max} \approx 1.34 \cdot \text{We}^{1/4}$, confirming that surface shear stress over the insulating vapor cushion is effectively zero.
2. **Dynamic Leidenfrost Depression by Solvents:**
   Addition of volatile solvents (acetone, ethanol, methanol) depresses the dynamic Leidenfrost temperature compared to pure water ($T_{\text{LD}} \approx 310^\circ\text{C}$) down to $T_{\text{LD}} \approx 250^\circ\text{C}$ for $60\%$ acetone, driven by preferential flash vaporization of the lower boiling point fraction.
3. **Transition Boiling Explosive Fragmentation:**
   At intermediate temperatures ($130^\circ\text{C} \le T \le 220^\circ\text{C}$), volatile binary mixtures exhibit severe explosive boiling where localized vapor explosions eject secondary micro-droplets, producing fragmentation counts up to $N > 15$.
4. **Restitution Coefficient Jump:**
   The restitution coefficient $e = \sqrt{H_{\text{rebound}}/h_{\text{drop}}}$ displays a sharp Heaviside-like step from $e = 0$ (deposition/sticking) to $e \approx 0.35\text{–}0.52$ at the dynamic Leidenfrost threshold.

---

## 📖 Citation & Contact

If you utilize this dataset, computer vision pipeline, or figures in your research, please reference:

```bibtex
@dataset{leidenfrost_droplet_dynamics_2026,
  author    = {Pablo and Research Collaborators},
  title     = {Experimental Dataset on Leidenfrost Droplet Impact Dynamics Across 10 Binary Liquid Formulations},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/pablo-2047/leidenfrost}
}
```
