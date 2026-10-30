"""
Surgical Rectification Script for ONLY:
  Water + 20% Acetone, 25.8 cm release height, 130°C
  (20 perc acetone / 09_11_2022_25.8cm / 130_C001H001S0001)

Why the original analyze_case.py failed on this case:
  1. Optical Baseline Misalignment:
     - The automated tilt detector encountered a faint lamp shadow/meniscus at y = 473 on the image
       margins and set y_contact = y_first_occ + 3 = 476 px.
     - However, the true physical substrate plate surface in this experiment rig is at y = 482 px
       (identically matching the 160°C, 190°C, and 220°C runs in the exact same 09_11_2022_25.8cm series).
     - Setting y_contact = 476 px placed the substrate boundary 6 pixels too high, slicing off the
       lower portion of the impacting droplet!
  2. Severe Physical Artifacts Caused by the 6 px Offset:
     - In Frame 6 (t0+05, 1.25 ms, 130_C001H001S0001000014.jpg), the spreading contact diameter was
       under-measured at Dc = 158 px instead of the true contact neck of Dc = 346 px, and apex height
       was reported as only H = 14 px.
     - As the droplet flattened onto the substrate, its apex dropped below row 476.
     - Consequently, in Frames 8, 9, 10 (1.75 - 2.25 ms), the entire droplet was below row 476, causing
       the algorithm to detect 0 droplet pixels, reporting Droplet_Height = 0, Dc = 0, Dsp = 0, and
       falsely flipping the status to AIRBORNE!
     - In Frame 11, the spreading disc barely reached row 475, producing a violent non-physical jump
       from Dc = 0 to Dc = 251 px, followed by massive oscillations.

Surgical Fix:
  - Correct the physical substrate contact baseline to y_contact = 482 px.
  - Re-process all 43 frames of the impact event (Raw Index 8 to 50):
      * Accurate deskewing and composite background subtraction.
      * Extraction of true droplet contour clamped strictly at y <= 482 px (zero border intrusion).
      * Re-assertion of flush red baseline at y = 482 px.
      * Physical contact diameter, spread diameter, apex height, and 3D CoM calculations.
      * High-resolution annotated image generation with HUD box and 1.6x magnified contact inset.
  - Update local measurements_data.csv and measurements_data.xlsx.
  - Update master_summary_plot_2_rectified.csv (Surface_Level_y_px = 482, Max_Contact_Dia_px = 487 px).
  - Regenerate Category 1 & Category 7 plots in plot_2 and dashboard index.html, then sync to GitHub.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import analyze_case as ac
from analyze_case import detect_substrate_tilt, reconstruct_droplet_contour, detect_burst_fragments

INPUT_DIR = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost\20 perc acetone\09_11_2022_25.8cm\130_C001H001S0001"
OUTPUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\20 perc acetone\09_11_2022_25.8cm\130_C001H001S0001"
MASTER_PLOT2_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"

CAMERA_FPS = 4000.0
STANDARD_DROPLET_DIA_MM = 3.1204


def rectify_case():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    if not files:
        raise RuntimeError(f"No files found in {INPUT_DIR}")

    img0_bgr = cv2.imread(files[0])
    H, W = img0_bgr.shape[:2]
    y_tilt_ref, angle = detect_substrate_tilt(img0_bgr)
    M = cv2.getRotationMatrix2D((W / 2.0, y_tilt_ref), angle, 1.0)

    def read_deskewed_bgr_and_gray(idx):
        bgr = cv2.imread(files[idx])
        rot = cv2.warpAffine(bgr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        return rot, gray

    # True physical contact line for 09_11_2022_25.8cm series
    y_contact = 482
    y_top_cut = 135

    sample_idx = np.linspace(0, min(len(files) - 1, 180), 19, dtype=int)
    bg90 = np.percentile([read_deskewed_bgr_and_gray(i)[1] for i in sample_idx], 90, axis=0).astype(np.uint8)
    img0_gray = read_deskewed_bgr_and_gray(0)[1]
    bg = img0_gray.copy()
    bg[:y_contact - 4, :] = np.maximum(img0_gray[:y_contact - 4, :], bg90[:y_contact - 4, :])

    t_contact = 9  # Raw index 9 (000010.jpg) is touchdown
    t0 = 8         # Raw index 8 (000009.jpg) is Frame 1 (just before contact)

    d0_eq_px = 106.0
    scale_px_per_mm = 33.970
    mm_per_px = 1.0 / scale_px_per_mm

    case_name = os.path.basename(os.path.normpath(INPUT_DIR))
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(INPUT_DIR)))

    print(f"Rectifying Case: {parent_dir} / {case_name}")
    print(f"  Setting Corrected Baseline: y_contact = {y_contact} px")
    print(f"  Frame 1 (t=0.00 ms) : Raw Index {t0} ({os.path.basename(files[t0])})")
    print(f"  Frame 2 (t=0.25 ms) : Raw Index {t_contact} ({os.path.basename(files[t_contact])})")

    records = []
    x_hint = W / 2.0

    for f_idx in range(t0, len(files)):
        orig_fname = os.path.basename(files[f_idx])
        rot, gray = read_deskewed_bgr_and_gray(f_idx)

        rel_step = f_idx - t0
        frame_num = rel_step + 1
        time_s = rel_step / CAMERA_FPS
        time_ms = time_s * 1000.0
        step_name = f"t0+{rel_step:02d}"
        status = "PRE-IMPACT (AIRBORNE)" if rel_step == 0 else "AIRBORNE"

        y_effective_clip = y_contact
        if rel_step == 0:
            # Droplet tip is at row 447 in pre-impact Frame 1
            y_effective_clip = 447

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
        mean_sat_dia_mm = float(np.mean([s["d_eq_px"] for s in sat_stats]) / scale_px_per_mm) if sat_stats else 0.0

        x_com = None
        y_com = None
        best_left_x = None
        best_right_x = None
        y_top = None
        x_top = None

        if c is not None:
            # Clamp contour points to never dip below y_contact
            c[:, 0, 1] = np.minimum(c[:, 0, 1], y_contact)

            pts = c[:, 0, :]
            min_y_idx = int(np.argmin(pts[:, 1]))
            y_top = int(pts[min_y_idx, 1])
            x_top = int(np.median(pts[pts[:, 1] == y_top, 0]))
            y_bot = int(pts[:, 1].max())
            x_hint = float(np.median(pts[:, 0]))

            height_px = max(0, y_contact - y_top)
            height_mm = height_px / scale_px_per_mm
            norm_height = height_px / d0_eq_px

            # 3D Axisymmetric Volume Center of Mass
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
                    primary_com_height_mm = prim_h_px / scale_px_per_mm

                    total_w_sum = w_sum_prim + sum(s["w_sum"] for s in sat_stats)
                    y_com = (w_sum_prim * y_com_prim + sum(s["w_sum"] * s["y_cm"] for s in sat_stats)) / total_w_sum
                    x_com = (w_sum_prim * x_com_prim + sum(s["w_sum"] * s["x_cm"] for s in sat_stats)) / total_w_sum
                    com_height_px = max(0.0, float(y_contact) - y_com)
                    com_height_mm = com_height_px / scale_px_per_mm
                    norm_com_height = com_height_px / d0_eq_px

            d_spread_px = int(pts[:, 0].max() - pts[:, 0].min() + 1)
            d_spread_mm = d_spread_px / scale_px_per_mm
            beta_spread = d_spread_px / d0_eq_px

            # Contact line measurement
            if f_idx >= t_contact and y_effective_clip == y_contact:
                contact_band = solid[max(0, y_contact - 2):y_contact + 1, :]
                xs_c = np.where(contact_band.any(axis=0))[0]
                if len(xs_c) >= 3 and y_bot >= y_contact - 3:
                    best_left_x = int(xs_c[0])
                    best_right_x = int(xs_c[-1])
                    d_contact_px = best_right_x - best_left_x + 1
                    d_contact_mm = d_contact_px / scale_px_per_mm
                    beta_contact = d_contact_px / d0_eq_px
                    status = "IN CONTACT"

                total_contact_px = d_contact_px + sum(s["contact_px"] for s in sat_stats)
                total_contact_mm = total_contact_px / scale_px_per_mm

        # Render full-frame annotated visualization
        canvas = rot.copy()

        # Red true contact line across substrate
        cv2.line(canvas, (0, y_contact), (W - 1, y_contact), (0, 0, 255), 1, cv2.LINE_AA)

        # Primary droplet contour in cyan
        if c is not None:
            cv2.drawContours(canvas, [c], -1, (255, 220, 0), 1, cv2.LINE_AA)
        if sat_contours:
            cv2.drawContours(canvas, sat_contours, -1, (0, 140, 255), 1, cv2.LINE_AA)

        # Re-assert red substrate baseline flush
        cv2.line(canvas, (0, y_contact), (W - 1, y_contact), (0, 0, 255), 1, cv2.LINE_AA)

        # Height measurement guide
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

        # Green contact line
        if status == "IN CONTACT" and best_left_x is not None:
            cy = y_contact
            cv2.line(canvas, (best_left_x, cy), (best_right_x, cy), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(canvas, (best_left_x, cy - 6), (best_left_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (best_right_x, cy - 6), (best_right_x, cy + 6), (0, 255, 255), 1, cv2.LINE_AA)

        # Magnified Droplet & Contact View Inset
        crop_half_w = max(170, int(d_spread_px // 2 + 55))
        cx1 = max(0, min(W - 2 * crop_half_w, int(x_hint) - crop_half_w))
        cx2 = min(W, cx1 + 2 * crop_half_w)
        if y_top is not None and y_top < y_contact - 155:
            cy1 = max(0, y_top - 25)
            cy2 = min(H, max(cy1 + 200, y_contact + 20))
        else:
            cy1 = max(0, y_contact - 175)
            cy2 = min(H, y_contact + 25)
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

        # Labels next to measurement guide
        if guide_x is not None and y_top is not None:
            top_label_y = max(25, y_top + 14)
            cv2.putText(canvas, f"H = {height_px} px ({height_mm:.2f} mm)", (guide_x + 10, top_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 120, 255), 2, cv2.LINE_AA)
            if y_com is not None:
                com_label_y = max(top_label_y + 20, int(round(y_com)) + 4)
                cv2.putText(canvas, f"Hcm = {com_height_px:.1f} px ({com_height_mm:.2f} mm)", (guide_x + 10, com_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 0, 200), 2, cv2.LINE_AA)

        if status == "IN CONTACT" and best_left_x is not None:
            cv2.putText(canvas, f"Dc = {d_contact_px} px ({d_contact_mm:.2f} mm)", (max(20, best_left_x - 10), min(H - 15, y_contact + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # HUD Overlay Box
        box_w, box_h = 540, 238
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (0, 220, 255), 2)
        cv2.putText(canvas, f"Case: {parent_dir} / {case_name}", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame #{frame_num:03d} ({step_name}) | t = {time_ms:.2f} ms ({int(CAMERA_FPS)} fps) | {orig_fname}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)

        dc_color = (0, 255, 0) if status == "IN CONTACT" else (0, 165, 255)
        cv2.putText(canvas, f"Contact (Dc) : {d_contact_px:3d}px | {d_contact_mm:.2f}mm | Beta={beta_contact:.3f} [{status}]", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.49, dc_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Apex H  (H)  : {height_px:3d}px | {height_mm:.2f}mm | H/D0={norm_height:.3f}", (30, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"3D CoM (Hcm) : {com_height_px:5.1f}px | {com_height_mm:.2f}mm | Hcm/D0={norm_com_height:.3f}", (30, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 130, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Spread (Dsp) : {d_spread_px:3d}px | {d_spread_mm:.2f}mm | Beta_sp={beta_spread:.3f}", (30, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Scale (D0={d0_eq_px:.1f}px / {STANDARD_DROPLET_DIA_MM:.3f}mm): {scale_px_per_mm:.3f} px/mm ({mm_per_px:.5f} mm/px)", (30, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 255, 170), 1, cv2.LINE_AA)

        out_fname = f"frame_{frame_num:04d}_{step_name}_{time_ms:.2f}ms.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        records.append({
            "Frame_Number": frame_num,
            "Time_ms": round(time_ms, 2),
            "Time_s": round(time_s, 6),
            "Relative_Step": rel_step,
            "Raw_Frame_Index": f_idx,
            "Original_Filename": orig_fname,
            "Analysed_Photo_Filename": out_fname,
            "Contact_Status": status,
            "Fragment_Count": fragment_count,
            "Burst_Status": burst_status,
            "Contact_Diameter_px": int(d_contact_px),
            "Contact_Diameter_mm": round(float(d_contact_mm), 3),
            "Beta_Contact_Dc_D0": round(float(beta_contact), 4),
            "Total_MultiDrop_Contact_mm": round(float(total_contact_mm), 3),
            "Spreading_Diameter_px": int(d_spread_px),
            "Spreading_Diameter_mm": round(float(d_spread_mm), 3),
            "Beta_Spread_Dsp_D0": round(float(beta_spread), 4),
            "Droplet_Height_px": int(height_px),
            "Droplet_Height_mm": round(float(height_mm), 3),
            "Normalized_Height_H_D0": round(float(norm_height), 4),
            "CoM_Height_px": round(float(com_height_px), 2),
            "CoM_Height_mm": round(float(com_height_mm), 3),
            "Normalized_CoM_Height_Hcm_D0": round(float(norm_com_height), 4),
            "Primary_Core_CoM_Height_mm": round(float(primary_com_height_mm), 3),
            "Mean_Satellite_Dia_mm": round(float(mean_sat_dia_mm), 3),
            "True_Contact_Line_y_px": y_contact,
            "PreImpact_D0_px": d0_eq_px,
            "Standard_D0_mm": STANDARD_DROPLET_DIA_MM,
            "Standard_Scale_px_per_mm": round(scale_px_per_mm, 4),
            "mm_per_pixel": round(mm_per_px, 6),
            "Camera_FPS": int(CAMERA_FPS)
        })

    df_out = pd.DataFrame(records)
    csv_out_path = os.path.join(OUTPUT_DIR, "measurements_data.csv")
    xlsx_out_path = os.path.join(OUTPUT_DIR, "measurements_data.xlsx")
    df_out.to_csv(csv_out_path, index=False)
    df_out.to_excel(xlsx_out_path, index=False)
    print(f"[+] Saved {len(df_out)} frames to {csv_out_path} and {xlsx_out_path}")

    # Update master_summary_plot_2_rectified.csv
    if os.path.exists(MASTER_PLOT2_CSV):
        df_master = pd.read_csv(MASTER_PLOT2_CSV)
        mask = (
            (df_master["Fluid_Folder"] == "20 perc acetone") &
            (df_master["Drop_Height_Folder"] == "09_11_2022_25.8cm") &
            (df_master["Temperature_C"] == 130.0)
        )
        if mask.any():
            max_contact_px = int(df_out["Contact_Diameter_px"].max())
            max_contact_mm = round(max_contact_px / scale_px_per_mm, 3)
            max_beta_c = round(max_contact_px / d0_eq_px, 4)
            max_spread_px = int(df_out["Spreading_Diameter_px"].max())
            max_spread_mm = round(max_spread_px / scale_px_per_mm, 3)
            max_beta_sp = round(max_spread_px / d0_eq_px, 4)
            max_h_px = int(df_out["Droplet_Height_px"].max())
            max_h_mm = round(max_h_px / scale_px_per_mm, 3)
            max_h_norm = round(max_h_px / d0_eq_px, 4)

            df_master.loc[mask, "Surface_Level_y_px"] = y_contact
            df_master.loc[mask, "Max_Contact_Dia_px"] = max_contact_px
            df_master.loc[mask, "Max_Contact_Dia_mm"] = max_contact_mm
            df_master.loc[mask, "Max_Beta_Contact_Dc_D0"] = max_beta_c
            df_master.loc[mask, "Max_Spread_Dia_px"] = max_spread_px
            df_master.loc[mask, "Max_Spread_Dia_mm"] = max_spread_mm
            df_master.loc[mask, "Max_Beta_Spread_Dsp_D0"] = max_beta_sp
            df_master.loc[mask, "Max_Droplet_Height_px"] = max_h_px
            df_master.loc[mask, "Max_Droplet_Height_mm"] = max_h_mm
            df_master.loc[mask, "Max_Normalized_Height_H_D0"] = max_h_norm

            df_master.to_csv(MASTER_PLOT2_CSV, index=False)
            print(f"[+] Updated {MASTER_PLOT2_CSV} for 20 perc acetone 25.8cm 130C")


if __name__ == "__main__":
    rectify_case()
