"""
Surgical Rectification Script for:
  Water + 20% Ethanol at 25.8 cm, 130°C:
    Folder: 20 perc ethanol / 08_10_2022_25.8 cm / 130_C001H001S0001

Root Cause:
  The automated tilt / baseline detector set y_contact to 491 px, which was 14 pixels
  too high (in the bright background sky above the physical substrate plate at y = 505 px).
  This caused the baseline to slice across the top sliver of the droplet (measuring H = 7 px
  and Dc = 99 px in Frame 6), and starting from Frame 7 as the droplet flattened, it dropped
  below y = 491 px, triggering false AIRBORNE status and 0 px contact diameter.

Surgical Fix:
  - Correct physical substrate baseline: y_contact = 505 px.
  - Correct touchdown alignment: t0 = 12 (Frame 1 pre-impact, 000013.jpg),
    t_contact = 13 (Frame 2 touchdown, 000014.jpg).
  - Reprocess all 250 frames with full HUD visualization and insets.
  - Update measurements_data.csv and measurements_data.xlsx.
  - Update master summaries in Analysis/plot_2 and Analysis/plots.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from analyze_case import detect_substrate_tilt, reconstruct_droplet_contour, detect_burst_fragments

BASE_RAW = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost\20 perc ethanol\08_10_2022_25.8 cm\130_C001H001S0001"
OUTPUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\20 perc ethanol\08_10_2022_25.8 cm\130_C001H001S0001"
MASTER_PLOT2_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
MASTER_PHYSICS_CSV = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"

CAMERA_FPS = 4000.0
STANDARD_DROPLET_DIA_MM = 3.1204
TOTAL_FRAMES_TO_PROCESS = 250

Y_CONTACT = 505
T0 = 12
T_CONTACT = 13
D0_EQ_PX = 105.0
SCALE_PX_PER_MM = 33.6493
MM_PER_PX = 1.0 / SCALE_PX_PER_MM
Y_TOP_CUT = 50


def run_rectification():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(BASE_RAW, "*.jpg")))
    if not files:
        raise RuntimeError(f"No files found in {BASE_RAW}")

    img0_bgr = cv2.imread(files[0])
    H, W = img0_bgr.shape[:2]
    y_tilt_ref, angle = detect_substrate_tilt(img0_bgr)
    M = cv2.getRotationMatrix2D((W / 2.0, y_tilt_ref), angle, 1.0)

    def read_deskewed_bgr_and_gray(idx):
        bgr = cv2.imread(files[idx])
        rot = cv2.warpAffine(bgr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        return rot, gray

    sample_idx = np.linspace(0, min(len(files) - 1, 180), 19, dtype=int)
    bg90 = np.percentile([read_deskewed_bgr_and_gray(i)[1] for i in sample_idx], 90, axis=0).astype(np.uint8)
    img0_gray = read_deskewed_bgr_and_gray(0)[1]
    bg = img0_gray.copy()
    bg[:Y_CONTACT - 4, :] = np.maximum(img0_gray[:Y_CONTACT - 4, :], bg90[:Y_CONTACT - 4, :])

    print("=======================================================")
    print("Rectifying: 20 perc ethanol / 08_10_2022_25.8 cm / 130_C001H001S0001")
    print(f"  Corrected Baseline: y_contact = {Y_CONTACT} px (was 491 px)")
    print(f"  Frame 1 (t=0.00 ms): Raw Index {T0} ({os.path.basename(files[T0])})")
    print(f"  Frame 2 (t=0.25 ms): Raw Index {T_CONTACT} ({os.path.basename(files[T_CONTACT])})")

    records = []
    x_hint = W / 2.0
    end_f = min(len(files), T0 + TOTAL_FRAMES_TO_PROCESS)

    for f_idx in range(T0, end_f):
        orig_fname = os.path.basename(files[f_idx])
        rot, gray = read_deskewed_bgr_and_gray(f_idx)

        rel_step = f_idx - T0
        frame_num = rel_step + 1
        time_s = rel_step / CAMERA_FPS
        time_ms = time_s * 1000.0
        step_name = f"t0+{rel_step:02d}"
        status = "PRE-IMPACT (AIRBORNE)" if rel_step == 0 else "AIRBORNE"

        y_effective_clip = Y_CONTACT
        if rel_step == 0:
            col = np.mean(gray[:, int(W / 2 - 50):int(W / 2 + 50)], axis=1)
            dark_rows = np.where((col < 180) & (np.arange(H) < Y_CONTACT - 3))[0]
            y_effective_clip = int(dark_rows.max()) if len(dark_rows) > 0 else Y_CONTACT - 10

        c, solid = reconstruct_droplet_contour(gray, bg, Y_CONTACT, Y_TOP_CUT, x_hint, y_effective_clip)
        sat_contours, sat_stats = detect_burst_fragments(gray, bg, Y_CONTACT, Y_TOP_CUT, solid, x_hint)

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
        mean_sat_dia_mm = float(np.mean([s["d_eq_px"] for s in sat_stats]) / SCALE_PX_PER_MM) if sat_stats else 0.0

        x_com = None
        y_com = None
        best_left_x = None
        best_right_x = None
        y_top = None
        x_top = None

        if c is not None:
            c[:, 0, 1] = np.minimum(c[:, 0, 1], Y_CONTACT)

            pts = c[:, 0, :]
            min_y_idx = int(np.argmin(pts[:, 1]))
            y_top = int(pts[min_y_idx, 1])
            x_top = int(np.median(pts[pts[:, 1] == y_top, 0]))
            y_bot = int(pts[:, 1].max())
            x_hint = float(np.median(pts[:, 0]))

            height_px = max(0, Y_CONTACT - y_top)
            height_mm = height_px / SCALE_PX_PER_MM
            norm_height = height_px / D0_EQ_PX

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
                    prim_h_px = max(0.0, float(Y_CONTACT) - y_com_prim)
                    primary_com_height_mm = prim_h_px / SCALE_PX_PER_MM

                    total_w_sum = w_sum_prim + sum(s["w_sum"] for s in sat_stats)
                    y_com = (w_sum_prim * y_com_prim + sum(s["w_sum"] * s["y_cm"] for s in sat_stats)) / total_w_sum
                    x_com = (w_sum_prim * x_com_prim + sum(s["w_sum"] * s["x_cm"] for s in sat_stats)) / total_w_sum
                    com_height_px = max(0.0, float(Y_CONTACT) - y_com)
                    com_height_mm = com_height_px / SCALE_PX_PER_MM
                    norm_com_height = com_height_px / D0_EQ_PX

            d_spread_px = int(pts[:, 0].max() - pts[:, 0].min() + 1)
            d_spread_mm = d_spread_px / SCALE_PX_PER_MM
            beta_spread = d_spread_px / D0_EQ_PX

            if f_idx >= T_CONTACT and y_effective_clip == Y_CONTACT:
                contact_band = solid[max(0, Y_CONTACT - 2):Y_CONTACT + 1, :]
                xs_c = np.where(contact_band.any(axis=0))[0]
                if len(xs_c) >= 3 and y_bot >= Y_CONTACT - 3:
                    best_left_x = int(xs_c[0])
                    best_right_x = int(xs_c[-1])
                    d_contact_px = best_right_x - best_left_x + 1
                    d_contact_mm = d_contact_px / SCALE_PX_PER_MM
                    beta_contact = d_contact_px / D0_EQ_PX
                    status = "IN CONTACT"

                total_contact_px = d_contact_px + sum(s["contact_px"] for s in sat_stats)
                total_contact_mm = total_contact_px / SCALE_PX_PER_MM

        canvas = rot.copy()
        cv2.line(canvas, (0, Y_CONTACT), (W - 1, Y_CONTACT), (0, 0, 255), 1, cv2.LINE_AA)

        if c is not None:
            cv2.drawContours(canvas, [c], -1, (255, 220, 0), 1, cv2.LINE_AA)
        if sat_contours:
            cv2.drawContours(canvas, sat_contours, -1, (0, 140, 255), 1, cv2.LINE_AA)

        cv2.line(canvas, (0, Y_CONTACT), (W - 1, Y_CONTACT), (0, 0, 255), 1, cv2.LINE_AA)

        guide_x = None
        if y_top is not None:
            guide_x = int(best_right_x + 24) if best_right_x is not None else int(x_hint + d_spread_px // 2 + 24)
            guide_x = min(W - 180, max(40, guide_x))
            cv2.line(canvas, (x_top, y_top), (guide_x + 5, y_top), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x, y_top), (guide_x, Y_CONTACT), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x - 5, y_top), (guide_x + 5, y_top), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (guide_x - 5, Y_CONTACT), (guide_x + 5, Y_CONTACT), (0, 255, 255), 1, cv2.LINE_AA)

        if x_com is not None and y_com is not None:
            cx_i, cy_i = int(round(x_com)), int(round(y_com))
            cv2.circle(canvas, (cx_i, cy_i), 5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx_i, cy_i), 5, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx_i - 8, cy_i), (cx_i + 8, cy_i), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx_i, cy_i - 8), (cx_i, cy_i + 8), (255, 0, 255), 1, cv2.LINE_AA)
            if guide_x is not None:
                cv2.line(canvas, (cx_i, cy_i), (guide_x + 4, cy_i), (255, 0, 255), 1, cv2.LINE_AA)

        if status == "IN CONTACT" and best_left_x is not None:
            cy = Y_CONTACT
            cv2.line(canvas, (best_left_x, cy), (best_right_x, cy), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(canvas, (best_left_x, cy - 6), (best_left_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (best_right_x, cy - 6), (best_right_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)

        # Magnified Inset
        crop_half_w = max(170, int(d_spread_px // 2 + 55))
        cx1 = max(0, min(W - 2 * crop_half_w, int(x_hint) - crop_half_w))
        cx2 = min(W, cx1 + 2 * crop_half_w)
        if y_top is not None and y_top < Y_CONTACT - 155:
            cy1 = max(0, y_top - 25)
            cy2 = min(H, max(cy1 + 200, Y_CONTACT + 20))
        else:
            cy1 = max(0, Y_CONTACT - 175)
            cy2 = min(H, Y_CONTACT + 25)
        inset_raw = canvas[cy1:cy2, cx1:cx2].copy()
        max_inset_h = 340
        scale_f = min(1.6, max_inset_h / float(max(1, cy2 - cy1)))
        inset_zoom = cv2.resize(inset_raw, (0, 0), fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC)
        ih, iw = inset_zoom.shape[:2]
        ix1 = W - iw - 20
        iy1 = 44
        cv2.rectangle(canvas, (ix1 - 3, iy1 - 26), (ix1 + iw + 3, iy1 + ih + 3), (30, 30, 30), -1)
        cv2.putText(canvas, "Magnified Droplet & Contact View", (ix1 + 6, iy1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[iy1:iy1 + ih, ix1:ix1 + iw] = inset_zoom
        cv2.rectangle(canvas, (ix1 - 1, iy1 - 1), (ix1 + iw, iy1 + ih), (0, 255, 255), 1)

        if guide_x is not None and y_top is not None:
            top_label_y = max(25, y_top + 14)
            cv2.putText(canvas, f"H = {height_px} px ({height_mm:.2f} mm)", (guide_x + 10, top_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 120, 255), 2, cv2.LINE_AA)
            if y_com is not None:
                com_label_y = max(top_label_y + 20, int(round(y_com)) + 4)
                cv2.putText(canvas, f"Hcm = {com_height_px:.1f} px ({com_height_mm:.2f} mm)", (guide_x + 10, com_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 0, 200), 2, cv2.LINE_AA)

        if status == "IN CONTACT" and best_left_x is not None:
            cv2.putText(canvas, f"Dc = {d_contact_px} px ({d_contact_mm:.2f} mm)", (max(20, best_left_x - 10), min(H - 15, Y_CONTACT + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        box_w, box_h = 540, 238
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (0, 220, 255), 2)
        cv2.putText(canvas, "Case: 08_10_2022_25.8 cm / 130_C001H001S0001", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame #{frame_num:03d} ({step_name}) | t = {time_ms:.2f} ms ({int(CAMERA_FPS)} fps) | {orig_fname}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)

        dc_color = (0, 255, 0) if status == "IN CONTACT" else (0, 165, 255)
        cv2.putText(canvas, f"Contact (Dc) : {d_contact_px}px | {d_contact_mm:.2f}mm | Beta={beta_contact:.3f} [{status}]", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.50, dc_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Apex H  (H) : {height_px}px | {height_mm:.2f}mm | H/D0={norm_height:.3f}", (30, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"3D CoM (Hcm) : {com_height_px:.1f}px | {com_height_mm:.2f}mm | Hcm/D0={norm_com_height:.3f}", (30, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Spread (Dsp) : {d_spread_px}px | {d_spread_mm:.2f}mm | Beta_sp={beta_spread:.3f}", (30, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Scale (D0={D0_EQ_PX:.1f}px / {STANDARD_DROPLET_DIA_MM:.3f}mm): {SCALE_PX_PER_MM:.3f} px/mm ({MM_PER_PX:.5f} mm/px)", (30, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

        out_fname = f"frame_{frame_num:04d}_{step_name}_{time_ms:.2f}ms.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        cv2.imwrite(out_path, canvas)

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
            'True_Contact_Line_y_px': Y_CONTACT,
            'PreImpact_D0_px': D0_EQ_PX,
            'Standard_D0_mm': STANDARD_DROPLET_DIA_MM,
            'Standard_Scale_px_per_mm': round(SCALE_PX_PER_MM, 4),
            'mm_per_pixel': round(MM_PER_PX, 6),
            'Camera_FPS': CAMERA_FPS
        })

    df_out = pd.DataFrame(records)
    csv_out = os.path.join(OUTPUT_DIR, "measurements_data.csv")
    xlsx_out = os.path.join(OUTPUT_DIR, "measurements_data.xlsx")
    df_out.to_csv(csv_out, index=False)
    try:
        df_out.to_excel(xlsx_out, index=False)
    except Exception as e:
        print(f"  Note: could not save xlsx ({e})")

    max_dc = int(df_out['Contact_Diameter_px'].max())
    max_dsp = int(df_out['Spreading_Diameter_px'].max())
    print(f"  [+] Saved {len(df_out)} frames to {csv_out}")
    print(f"  [+] Max Dc = {max_dc} px ({max_dc/SCALE_PX_PER_MM:.2f} mm, Beta_c={max_dc/D0_EQ_PX:.3f})")
    print(f"  [+] Max Dsp = {max_dsp} px ({max_dsp/SCALE_PX_PER_MM:.2f} mm, Beta_sp={max_dsp/D0_EQ_PX:.3f})")

    # Update master summaries
    for master_path in [MASTER_PLOT2_CSV, MASTER_PHYSICS_CSV]:
        if not os.path.exists(master_path):
            continue
        df_m = pd.read_csv(master_path)
        mask = (df_m['Fluid_Folder'] == '20 perc ethanol') & \
               (df_m['Temperature_C'] == 130) & \
               (df_m['Drop_Height_Folder'] == '08_10_2022_25.8 cm')
        if mask.any():
            df_m.loc[mask, 'Surface_Level_y_px'] = Y_CONTACT
            df_m.loc[mask, 'Max_Contact_Dia_px'] = max_dc
            df_m.loc[mask, 'Max_Contact_Dia_mm'] = round(max_dc / SCALE_PX_PER_MM, 3)
            df_m.loc[mask, 'Max_Beta_Contact_Dc_D0'] = round(max_dc / D0_EQ_PX, 4)
            if 'Max_Spread_Dia_px' in df_m.columns:
                df_m.loc[mask, 'Max_Spread_Dia_px'] = max_dsp
            if 'T0_Impact_Frame' in df_m.columns:
                df_m.loc[mask, 'T0_Impact_Frame'] = T0
            if 'Contact_Frame_Idx' in df_m.columns:
                df_m.loc[mask, 'Contact_Frame_Idx'] = T_CONTACT
            df_m.to_csv(master_path, index=False)
            print(f"  [+] Updated master summary: {master_path}")


if __name__ == "__main__":
    run_rectification()
    print("\n20% ETHANOL 25.8 CM 130C RECTIFIED SUCCESSFULLY!")
