"""
High-Performance Surgical Rectification Pipeline for All 25% Methanol Cases:
Calibrates accurate physical substrate baselines across all 9 drop heights:
  - 27_08_2022_1.3cm:  Y_CONTACT = 696 px
  - 27_08_2022_2.9cm:  Y_CONTACT = 702 px
  - 27_08_2022_5.1cm:  Y_CONTACT = 712 px
  - 27_08_2022_8cm:    Y_CONTACT = 510 px
  - 27_08_2022_11.5cm: Y_CONTACT = 510 px
  - 27_08_2022_15.6cm: Y_CONTACT = 354 px
  - 27_08_2022_20.5cm:
      * 100°C - 190°C: Y_CONTACT = 584 px
      * 220°C - 400°C: Y_CONTACT = 396 px
  - 27_08_2022_25.8cm: Y_CONTACT = 394 px
  - 27_08_2022_32cm:   Y_CONTACT = 510 px

Runs in parallel using ProcessPoolExecutor across CPU cores.
Updates measurements_data.csv & .xlsx in Raw and Analysis folders.
Updates master summaries in Analysis/plot_2 and Analysis/plots.
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from analyze_case import reconstruct_droplet_contour, detect_burst_fragments

BASE_EXP_DIR = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost"
BASE_ANALYSIS_DIR = r"C:\Users\Pablo\Desktop\Analysis"
MASTER_PLOT2_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
MASTER_PHYSICS_CSV = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"

CAMERA_FPS = 4000.0
STANDARD_DROPLET_DIA_MM = 3.1204
TOTAL_FRAMES_TO_PROCESS = 250
Y_TOP_CUT = 135

def get_calibrated_y_contact(height_folder, temp_c):
    temp = float(temp_c)
    if height_folder.endswith("_1.3cm"):
        return 696
    elif height_folder.endswith("_2.9cm"):
        return 702
    elif height_folder.endswith("_5.1cm"):
        return 712
    elif height_folder.endswith("_8cm"):
        return 510
    elif height_folder.endswith("_11.5cm"):
        return 510
    elif height_folder.endswith("_15.6cm"):
        return 354
    elif height_folder.endswith("_20.5cm"):
        return 584 if temp < 210 else 396
    elif height_folder.endswith("_25.8cm"):
        return 394
    elif height_folder.endswith("_32cm"):
        return 510
    else:
        return 696


def process_single_case(case_info):
    h_fld = case_info['height_folder']
    c_fld = case_info['case_folder']
    temp_c = case_info['temp_c']
    t0 = int(case_info['t0'])
    d0_eq = float(case_info['d0_eq'])
    scale = float(case_info['scale'])
    mm_per_px = 1.0 / scale
    y_contact = get_calibrated_y_contact(h_fld, temp_c)

    raw_dir = os.path.join(BASE_EXP_DIR, "25 perc methanol", h_fld, c_fld)
    out_dir = os.path.join(BASE_ANALYSIS_DIR, "25 perc methanol", h_fld, c_fld)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not files:
        return None

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

        c, solid = reconstruct_droplet_contour(gray, bg, y_contact, Y_TOP_CUT, x_hint, y_effective_clip)
        sat_contours, sat_stats = detect_burst_fragments(gray, bg, y_contact, Y_TOP_CUT, solid, x_hint)

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

        x_com, y_com = None, None
        best_left_x, best_right_x = None, None
        y_top, x_top = None, None

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
        except Exception:
            pass

    max_dc = int(df_out['Contact_Diameter_px'].max())
    max_dsp = int(df_out['Spreading_Diameter_px'].max())
    max_h = int(df_out['Droplet_Height_px'].max())
    max_com = float(df_out['CoM_Height_px'].max())

    return {
        'height_folder': h_fld,
        'case_folder': c_fld,
        'temp_c': temp_c,
        'y_contact': y_contact,
        'max_dc': max_dc,
        'max_dc_mm': round(max_dc / scale, 3),
        'max_beta_c': round(max_dc / d0_eq, 4),
        'max_dsp': max_dsp,
        'max_dsp_mm': round(max_dsp / scale, 3),
        'max_beta_sp': round(max_dsp / d0_eq, 4),
        'max_h': max_h,
        'max_h_mm': round(max_h / scale, 3),
        'max_norm_h': round(max_h / d0_eq, 4),
        'max_com_px': round(max_com, 1),
        'max_com_mm': round(max_com / scale, 3),
        'max_norm_com': round(max_com / d0_eq, 4),
    }


def main(filter_height=None):
    t_start = time.time()
    print("=" * 65)
    print(f"[*] STARTING SURGICAL RECTIFICATION FOR 25% METHANOL (filter: {filter_height or 'ALL'})")
    print("=" * 65)

    df_m = pd.read_csv(MASTER_PLOT2_CSV)
    sub = df_m[df_m['Fluid_Folder'] == '25 perc methanol'].copy()
    if filter_height:
        sub = sub[sub['Drop_Height_Folder'] == filter_height]
    print(f"[*] Found {len(sub)} cases to process.")

    tasks = []
    for _, r in sub.iterrows():
        tasks.append({
            'height_folder': str(r['Drop_Height_Folder']),
            'case_folder': str(r['Case_Folder']),
            'temp_c': float(r['Temperature_C']),
            't0': int(r['Frame_1_Raw_Index_t0']),
            'd0_eq': float(r['PreImpact_D0_px']),
            'scale': float(r['Case_Scale_px_per_mm']),
        })

    max_workers = min(14, os.cpu_count() or 4)
    print(f"[*] Launching {len(tasks)} cases with {max_workers} worker processes...")

    completed_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_case, t): t for t in tasks}
        for f in as_completed(futures):
            res = f.result()
            if res:
                completed_results.append(res)
                print(f"  [+] Finished {res['height_folder']} / {res['case_folder']} "
                      f"(y_c={res['y_contact']}, Max_Dc={res['max_dc']}px/{res['max_dc_mm']}mm) "
                      f"[{len(completed_results)}/{len(tasks)}]")

    print(f"\n[*] All {len(completed_results)} cases processed in {time.time() - t_start:.1f}s.")

    # Update master summaries
    for master_path in [MASTER_PLOT2_CSV, MASTER_PHYSICS_CSV]:
        if not os.path.exists(master_path):
            continue
        df_target = pd.read_csv(master_path)
        for res in completed_results:
            mask = (df_target['Fluid_Folder'] == '25 perc methanol') & \
                   (df_target['Drop_Height_Folder'] == res['height_folder']) & \
                   (df_target['Case_Folder'] == res['case_folder'])
            if mask.any():
                df_target.loc[mask, 'Surface_Level_y_px'] = res['y_contact']
                df_target.loc[mask, 'Max_Contact_Dia_px'] = res['max_dc']
                df_target.loc[mask, 'Max_Contact_Dia_mm'] = res['max_dc_mm']
                df_target.loc[mask, 'Max_Beta_Contact_Dc_D0'] = res['max_beta_c']
                if 'Max_Spread_Dia_px' in df_target.columns:
                    df_target.loc[mask, 'Max_Spread_Dia_px'] = res['max_dsp']
                if 'Max_Spread_Dia_mm' in df_target.columns:
                    df_target.loc[mask, 'Max_Spread_Dia_mm'] = res['max_dsp_mm']
                if 'Max_Beta_Spread_Dsp_D0' in df_target.columns:
                    df_target.loc[mask, 'Max_Beta_Spread_Dsp_D0'] = res['max_beta_sp']
                if 'Max_Droplet_Height_px' in df_target.columns:
                    df_target.loc[mask, 'Max_Droplet_Height_px'] = res['max_h']
                if 'Max_Droplet_Height_mm' in df_target.columns:
                    df_target.loc[mask, 'Max_Droplet_Height_mm'] = res['max_h_mm']
                if 'Max_Normalized_Height_H_D0' in df_target.columns:
                    df_target.loc[mask, 'Max_Normalized_Height_H_D0'] = res['max_norm_h']
                if 'Max_3D_CoM_Height_px' in df_target.columns:
                    df_target.loc[mask, 'Max_3D_CoM_Height_px'] = res['max_com_px']
                if 'Max_3D_CoM_Height_mm' in df_target.columns:
                    df_target.loc[mask, 'Max_3D_CoM_Height_mm'] = res['max_com_mm']
                if 'Max_Normalized_3D_CoM_Hcm_D0' in df_target.columns:
                    df_target.loc[mask, 'Max_Normalized_3D_CoM_Hcm_D0'] = res['max_norm_com']

        df_target.to_csv(master_path, index=False)
        print(f"  [+] Saved updated master summary: {master_path}")

    print("\n[SUCCESS] Entire 25% Methanol dataset rectified successfully!")


if __name__ == "__main__":
    import sys
    f_h = sys.argv[1] if len(sys.argv) > 1 else None
    main(f_h)
