"""
Surgical Rectification Script for:
  Water + 20% Ethanol at 220°C:
    1) 08_10_2022_2.9cm / 220_C001H001S0001
       Baseline previously at y = 547 px. True physical plate plateau at y = 571 px.
       t0 = 20 (raw frame 20, 000021.jpg), t_contact = 21 (raw frame 21, 000022.jpg).
       D0 = 105.0 px, scale = 33.6495 px/mm.
    2) 08_10_2022_1.3cm / 220_C001H001S0001
       Baseline previously at y = 558 px. True physical plate plateau at y = 581 px.
       t0 = 18 (raw frame 18, 000019.jpg), t_contact = 19 (raw frame 19, 000020.jpg).
       D0 = 110.0 px, scale = 35.2518 px/mm.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from analyze_case import reconstruct_droplet_contour, detect_burst_fragments

BASE_EXP_DIR = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost"
BASE_ANALYSIS_DIR = r"C:\Users\Pablo\Desktop\Analysis"
MASTER_PLOT2_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
MASTER_PHYSICS_CSV = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"

CAMERA_FPS = 4000.0
STANDARD_DROPLET_DIA_MM = 3.1204
TOTAL_FRAMES_TO_PROCESS = 250

CASES_CONFIG = [
    {
        "fluid": "20 perc ethanol",
        "height_folder": "08_10_2022_2.9cm",
        "case_folder": "220_C001H001S0001",
        "y_contact": 571,
        "t0": 20,
        "t_contact": 21,
        "d0_eq_px": 105.0,
        "scale_px_per_mm": 33.6495,
        "y_top_cut": 135
    },
    {
        "fluid": "20 perc ethanol",
        "height_folder": "08_10_2022_1.3cm",
        "case_folder": "220_C001H001S0001",
        "y_contact": 581,
        "t0": 18,
        "t_contact": 19,
        "d0_eq_px": 110.0,
        "scale_px_per_mm": 35.2518,
        "y_top_cut": 135
    }
]


def process_case(cfg):
    fluid = cfg["fluid"]
    h_fld = cfg["height_folder"]
    c_fld = cfg["case_folder"]
    y_contact = cfg["y_contact"]
    t0 = cfg["t0"]
    t_contact = cfg["t_contact"]
    d0_eq = cfg["d0_eq_px"]
    scale = cfg["scale_px_per_mm"]
    mm_per_px = 1.0 / scale
    y_top_cut = cfg["y_top_cut"]

    raw_dir = os.path.join(BASE_EXP_DIR, fluid, h_fld, c_fld)
    out_dir = os.path.join(BASE_ANALYSIS_DIR, fluid, h_fld, c_fld)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not files:
        print(f"[!] No images in {raw_dir}")
        return

    print(f"\n=======================================================")
    print(f"[*] Processing {fluid} / {h_fld} / {c_fld}")
    print(f"    Baseline: y = {y_contact} px, t0 = {t0}, t_contact = {t_contact}")
    print(f"    D0 = {d0_eq} px, Scale = {scale:.4f} px/mm")
    print(f"=======================================================")

    img0 = cv2.imread(files[0])
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    H, W = img0_gray.shape

    bg90_idx = min(90, len(files) - 1)
    bg90_gray = cv2.imread(files[bg90_idx], cv2.IMREAD_GRAYSCALE)

    bg = np.zeros_like(img0_gray)
    bg[:y_contact - 4, :] = np.maximum(img0_gray[:y_contact - 4, :], bg90_gray[:y_contact - 4, :])
    bg[y_contact - 4:, :] = img0_gray[y_contact - 4:, :]

    records = []
    x_hint = W / 2.0
    end_f = min(len(files), t0 + TOTAL_FRAMES_TO_PROCESS)

    for f_idx in range(t0, end_f):
        orig_fname = os.path.basename(files[f_idx])
        rot = cv2.imread(files[f_idx])
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

        rel_step = f_idx - t0
        frame_num = rel_step + 1
        time_s = rel_step / CAMERA_FPS
        time_ms = time_s * 1000.0
        step_name = f"t0+{rel_step:02d}"
        status = "PRE-IMPACT (AIRBORNE)" if rel_step == 0 else "AIRBORNE"

        y_effective_clip = y_contact
        if rel_step == 0:
            col = np.mean(gray[:, int(W / 2 - 50):int(W / 2 + 50)], axis=1)
            dark_rows = np.where((col < 180) & (np.arange(H) < y_contact - 3))[0]
            y_effective_clip = int(dark_rows.max()) if len(dark_rows) > 0 else y_contact - 10

        c, solid = reconstruct_droplet_contour(gray, bg, y_contact, y_top_cut, x_hint, y_effective_clip)
        sat_contours, sat_stats = detect_burst_fragments(gray, bg, y_contact, y_top_cut, solid, x_hint)

        d_contact_px = 0
        d_contact_mm = 0.0
        beta_contact = 0.0
        total_contact_px = 0
        total_contact_mm = 0.0
        d_spread_px = 0
        d_spread_mm = 0.0
        beta_spread = 0.0
        height_px = 0
        height_mm = 0.0
        norm_height = 0.0
        com_height_px = 0.0
        com_height_mm = 0.0
        norm_com_height = 0.0
        primary_com_height_mm = 0.0
        fragment_count = (1 if c is not None else 0) + len(sat_stats)
        burst_status = "BURST" if fragment_count >= 2 else ("INTACT" if fragment_count == 1 else "NONE")
        mean_sat_dia_mm = float(np.mean([s["d_eq_px"] for s in sat_stats]) / scale) if sat_stats else 0.0

        x_com = None
        y_com = None
        best_left_x = None
        best_right_x = None
        y_top = None
        x_top = None

        if c is not None:
            c[:, 0, 1] = np.minimum(c[:, 0, 1], y_contact)

            pts = c[:, 0, :]
            min_y_idx = int(np.argmin(pts[:, 1]))
            y_top = int(pts[min_y_idx, 1])
            x_top = int(np.median(pts[pts[:, 1] == y_top, 0]))
            y_bot = int(pts[:, 1].max())
            x_hint = float(np.median(pts[:, 0]))

            height_px = max(0, y_contact - y_top)
            height_mm = height_px / scale
            norm_height = height_px / d0_eq

            ys_solid = np.where(solid.any(axis=1))[0]
            if len(ys_solid) > 0:
                sub_solid = solid[ys_solid, :] > 0
                w_rows = sub_solid.sum(axis=1).astype(np.float64)
                weights = w_rows ** 2
                w_sum_prim = float(weights.sum())
                if w_sum_prim > 0:
                    y_com_prim = float(np.dot(ys_solid, weights) / w_sum_prim)
                    x_indices = np.arange(W, dtype=np.float64)
                    x_mids = (sub_solid @ x_indices) / np.maximum(1.0, w_rows)
                    x_com_prim = float(np.dot(x_mids, weights) / w_sum_prim)
                    prim_h_px = max(0.0, float(y_contact) - y_com_prim)
                    primary_com_height_mm = prim_h_px / scale

                    total_w_sum = w_sum_prim + sum(s["w_sum"] for s in sat_stats)
                    y_com = (w_sum_prim * y_com_prim + sum(s["w_sum"] * s["y_cm"] for s in sat_stats)) / total_w_sum
                    x_com = (w_sum_prim * x_com_prim + sum(s["w_sum"] * s["x_cm"] for s in sat_stats)) / total_w_sum
                    com_height_px = max(0.0, float(y_contact) - y_com)
                    com_height_mm = com_height_px / scale
                    norm_com_height = com_height_px / d0_eq

            d_spread_px = int(pts[:, 0].max() - pts[:, 0].min() + 1)
            d_spread_mm = d_spread_px / scale
            beta_spread = d_spread_px / d0_eq

            if rel_step >= 1:
                contact_band = solid[max(0, y_contact - 6):y_contact + 1, :]
                xs_c = np.where(contact_band.any(axis=0))[0]
                if len(xs_c) >= 3 and y_bot >= y_contact - 6:
                    best_left_x = int(xs_c[0])
                    best_right_x = int(xs_c[-1])
                    d_contact_px = best_right_x - best_left_x + 1
                    d_contact_mm = d_contact_px / scale
                    beta_contact = d_contact_px / d0_eq
                    status = "IN CONTACT"

                total_contact_px = d_contact_px + sum(s["contact_px"] for s in sat_stats)
                total_contact_mm = total_contact_px / scale

        canvas = rot.copy()
        cv2.line(canvas, (0, y_contact), (W - 1, y_contact), (0, 0, 255), 1, cv2.LINE_AA)

        if c is not None:
            cv2.drawContours(canvas, [c], -1, (255, 220, 0), 1, cv2.LINE_AA)
        if sat_contours:
            cv2.drawContours(canvas, sat_contours, -1, (0, 140, 255), 1, cv2.LINE_AA)

        cv2.line(canvas, (0, y_contact), (W - 1, y_contact), (0, 0, 255), 1, cv2.LINE_AA)

        guide_x = None
        if y_top is not None:
            guide_x = int(best_right_x + 24) if best_right_x is not None else int(x_hint + d_spread_px // 2 + 24)
            guide_x = min(W - 180, max(40, guide_x))
            cv2.line(canvas, (x_top, y_top), (guide_x + 5, y_top), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x, y_top), (guide_x, y_contact), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x - 5, y_top), (guide_x + 5, y_top), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x - 5, y_contact), (guide_x + 5, y_contact), (0, 255, 255), 1, cv2.LINE_AA)

        if x_com is not None and y_com is not None:
            cx_i, cy_i = int(round(x_com)), int(round(y_com))
            cv2.circle(canvas, (cx_i, cy_i), 5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx_i, cy_i), 5, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx_i - 8, cy_i), (cx_i + 8, cy_i), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx_i, cy_i - 8), (cx_i, cy_i + 8), (255, 0, 255), 1, cv2.LINE_AA)
            if guide_x is not None:
                cv2.line(canvas, (cx_i, cy_i), (guide_x + 4, cy_i), (255, 0, 255), 1, cv2.LINE_AA)

        if status == "IN CONTACT" and best_left_x is not None:
            cy = y_contact
            cv2.line(canvas, (best_left_x, cy), (best_right_x, cy), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(canvas, (best_left_x, cy - 6), (best_left_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (best_right_x, cy - 6), (best_right_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)

        # Magnified Inset
        crop_half_w = max(170, int(d_spread_px // 2 + 55))
        cx1 = max(0, min(W - 2 * crop_half_w, int(x_hint) - crop_half_w))
        cx2 = min(W, cx1 + 2 * crop_half_w)
        if y_top is not None and y_top < y_contact - 155:
            cy1 = max(0, y_top - 25)
            cy2 = min(H, max(cy1 + 200, y_contact + 20))
        else:
            cy1 = max(0, y_contact - 175)
            cy2 = min(H, y_contact + 25)

        crop = canvas[cy1:cy2, cx1:cx2]
        inset_h, inset_w = 260, 430
        if crop.size > 0:
            inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
            ix1, iy1 = W - inset_w - 20, 20
            ix2, iy2 = ix1 + inset_w, iy1 + inset_h
            canvas[iy1:iy2, ix1:ix2] = inset
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (255, 255, 255), 2)
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy1 + 24), (30, 30, 30), -1)
            cv2.putText(canvas, "Magnified Droplet & Contact View", (ix1 + 10, iy1 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD Overlay Box
        hud_w, hud_h = 390, 160
        cv2.rectangle(canvas, (15, 15), (15 + hud_w, 15 + hud_h), (0, 0, 0), -1)
        cv2.rectangle(canvas, (15, 15), (15 + hud_w, 15 + hud_h), (0, 180, 255), 2)

        f_num_str = f"Frame #{frame_num:03d} ({step_name}) | t = {time_ms:.2f} ms ({int(CAMERA_FPS)} fps) | {orig_fname}"

        cv2.putText(canvas, f"Case: {h_fld} / {c_fld}", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f_num_str, (25, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

        status_color = (0, 255, 0) if status == "IN CONTACT" else (0, 180, 255)
        c_str = f"Contact (Dc) : {d_contact_px}px | {d_contact_mm:.2f}mm | Beta={beta_contact:.3f} [{status}]"
        cv2.putText(canvas, c_str, (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

        h_str = f"Apex H (H) : {height_px}px | {height_mm:.2f}mm | H/D0={norm_height:.3f}"
        cv2.putText(canvas, h_str, (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA)

        cm_str = f"3D CoM (Hcm) : {com_height_px:.1f}px | {com_height_mm:.2f}mm | Hcm/D0={norm_com_height:.3f}"
        cv2.putText(canvas, cm_str, (25, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 255), 1, cv2.LINE_AA)

        sp_str = f"Spread (Dsp) : {d_spread_px}px | {d_spread_mm:.2f}mm | Beta_sp={beta_spread:.3f}"
        cv2.putText(canvas, sp_str, (25, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        sc_str = f"Scale (D0={d0_eq:.1f}px / {STANDARD_DROPLET_DIA_MM:.3f}mm): {scale:.3f} px/mm ({mm_per_px:.5f} mm/px)"
        cv2.putText(canvas, sc_str, (25, 153),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)

        # Baseline annotations near the bottom
        if status == "IN CONTACT" and best_left_x is not None:
            mid_c = (best_left_x + best_right_x) // 2
            cv2.putText(canvas, f"Dc = {d_contact_px} px ({d_contact_mm:.2f} mm)",
                        (mid_c - 70, y_contact + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if guide_x is not None and height_px > 0:
            cv2.putText(canvas, f"H = {height_px} px ({height_mm:.2f} mm)",
                        (guide_x + 10, int((y_top + y_contact) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
        if guide_x is not None and com_height_px > 0:
            cv2.putText(canvas, f"Hcm = {com_height_px:.1f} px ({com_height_mm:.2f} mm)",
                        (guide_x + 10, int((y_top + y_contact) / 2) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 100, 255), 1, cv2.LINE_AA)

        out_fname = f"frame_{frame_num:04d}_{step_name}_{time_ms:.2f}ms.jpg"
        out_path = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_path, canvas)
        
        orig_analysis_path = os.path.join(out_dir, orig_fname)
        cv2.imwrite(orig_analysis_path, canvas)

        records.append({
            'Frame_Number': frame_num,
            'Time_ms': round(time_ms, 2),
            'Time_s': round(time_s, 6),
            'Relative_Step': step_name,
            'Raw_Frame_Index': f_idx,
            'Original_Filename': orig_fname,
            'Analysed_Photo_Filename': out_fname,
            'Contact_Status': status,
            'Fragment_Count': fragment_count,
            'Burst_Status': burst_status,
            'Contact_Diameter_px': d_contact_px,
            'Contact_Diameter_mm': round(d_contact_mm, 2),
            'Beta_Contact_Dc_D0': round(beta_contact, 4),
            'Total_MultiDrop_Contact_mm': round(total_contact_mm, 2),
            'Spreading_Diameter_px': d_spread_px,
            'Spreading_Diameter_mm': round(d_spread_mm, 2),
            'Beta_Spread_Dsp_D0': round(beta_spread, 4),
            'Droplet_Height_px': height_px,
            'Droplet_Height_mm': round(height_mm, 2),
            'Normalized_Height_H_D0': round(norm_height, 4),
            'CoM_Height_px': round(com_height_px, 1),
            'CoM_Height_mm': round(com_height_mm, 2),
            'Normalized_CoM_Height_Hcm_D0': round(norm_com_height, 4),
            'Primary_Core_CoM_Height_mm': round(primary_com_height_mm, 2),
            'Mean_Satellite_Dia_mm': round(mean_sat_dia_mm, 2),
            'True_Contact_Line_y_px': y_contact,
            'PreImpact_D0_px': d0_eq,
            'Standard_D0_mm': STANDARD_DROPLET_DIA_MM,
            'Standard_Scale_px_per_mm': round(scale, 4),
            'mm_per_pixel': round(mm_per_px, 6),
            'Camera_FPS': CAMERA_FPS
        })

    df_out = pd.DataFrame(records)
    for dest_dir in [out_dir, raw_dir]:
        csv_out = os.path.join(dest_dir, "measurements_data.csv")
        xlsx_out = os.path.join(dest_dir, "measurements_data.xlsx")
        df_out.to_csv(csv_out, index=False)
        try:
            df_out.to_excel(xlsx_out, index=False)
        except Exception as e:
            print(f"  Note: could not save xlsx ({e})")
        print(f"  [+] Saved {len(df_out)} frames to {csv_out}")

    max_dc = int(df_out['Contact_Diameter_px'].max())
    max_dsp = int(df_out['Spreading_Diameter_px'].max())
    print(f"  [+] Max Dc = {max_dc} px ({max_dc/scale:.2f} mm, Beta_c={max_dc/d0_eq:.3f})")
    print(f"  [+] Max Dsp = {max_dsp} px ({max_dsp/scale:.2f} mm, Beta_sp={max_dsp/d0_eq:.3f})")

    # Update master summaries
    for master_path in [MASTER_PLOT2_CSV, MASTER_PHYSICS_CSV]:
        if not os.path.exists(master_path):
            continue
        df_m = pd.read_csv(master_path)
        mask = (df_m['Fluid_Folder'] == fluid) & \
               (df_m['Temperature_C'] == 220) & \
               (df_m['Drop_Height_Folder'] == h_fld)
        if mask.any():
            df_m.loc[mask, 'Surface_Level_y_px'] = y_contact
            df_m.loc[mask, 'Max_Contact_Dia_px'] = max_dc
            df_m.loc[mask, 'Max_Contact_Dia_mm'] = round(max_dc / scale, 3)
            df_m.loc[mask, 'Max_Beta_Contact_Dc_D0'] = round(max_dc / d0_eq, 4)
            if 'Max_Spread_Dia_px' in df_m.columns:
                df_m.loc[mask, 'Max_Spread_Dia_px'] = max_dsp
            if 'T0_Impact_Frame' in df_m.columns:
                df_m.loc[mask, 'T0_Impact_Frame'] = t0
            if 'Contact_Frame_Idx' in df_m.columns:
                df_m.loc[mask, 'Contact_Frame_Idx'] = t_contact
            df_m.to_csv(master_path, index=False)
            print(f"  [+] Updated master summary: {master_path}")


def main():
    for cfg in CASES_CONFIG:
        process_case(cfg)
    print("\n[SUCCESS] Both 220C cases processed and master summaries updated!")


if __name__ == "__main__":
    main()
