"""
High-Performance Surgical Rectification Pipeline for All Pure Water Cases:
Calibrates exact physical substrate baselines:
  - 18_08_22_water_1.3cm:  Y_CONTACT = 740 px
  - 18_08_22_water_3cm:    Y_CONTACT = 325 px (T <= 130), 315 px (T <= 160), 706 px (190-370), 729 px (400)
  - 18_08_22_water_5.1cm:  Y_CONTACT = 738 px (T=100), 818 px (T >= 130)
  - 18_08_22_water_8cm:    Y_CONTACT = 850 px (T=100), 838 px (T >= 130)
  - 18_08_22_water_11.5cm: Y_CONTACT = 842 px (T <= 160), 865 px (T >= 190)
  - 18_08_22_water_15.6cm: Y_CONTACT = 448 px (T <= 250), 435 px (T >= 280)
  - 18_08_22_water_20.5cm: Y_CONTACT = 428 px (T <= 250), 405 px (T >= 280)
  - 18_08_22_water_25.8cm: Y_CONTACT = 430 px
  - 18_08_22_water_32cm:   Y_CONTACT = 370 px

Features:
  - Dynamically detects true first impact frame (t0).
  - Unbiased water segmentation preserving transparent water cores without im < 145 restriction.
  - Full-width lateral spreading detection without artificial x_hint +/- 320 limits.
  - Multi-threaded processing via ProcessPoolExecutor.
  - Updates measurements_data.csv/.xlsx in Raw and Analysis folders.
  - Updates master_summary_plot_2_rectified.csv and master_summary_with_physics.csv.
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from analyze_case import detect_burst_fragments

BASE_EXP_DIR = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost"
BASE_ANALYSIS_DIR = r"C:\Users\Pablo\Desktop\Analysis"
MASTER_PLOT2_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"
MASTER_PHYSICS_CSV = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"

CAMERA_FPS = 4000.0
STANDARD_DROPLET_DIA_MM = 3.1204
TOTAL_FRAMES_TO_PROCESS = 250
Y_TOP_CUT = 40


def get_calibrated_y_contact(height_folder, temp_c=None):
    temp = float(temp_c) if temp_c is not None else 130.0
    if "25.8cm" in height_folder:
        return 430
    elif "20.5cm" in height_folder:
        return 428 if temp <= 250 else 405
    elif "15.6cm" in height_folder:
        return 448 if temp <= 250 else 435
    elif "11.5cm" in height_folder:
        return 842 if temp <= 160 else 865
    elif "5.1cm" in height_folder:
        return 738 if temp <= 100 else 818
    elif "32cm" in height_folder:
        return 370
    elif "1.3cm" in height_folder:
        return 740
    elif "_3cm" in height_folder or height_folder.endswith("3cm"):
        return 325 if temp <= 130 else (315 if temp <= 160 else (729 if temp >= 400 else 706))
    elif "_8cm" in height_folder or height_folder.endswith("8cm"):
        return 850 if temp <= 100 else 838
    else:
        return 448


def detect_case_t0(files, bg, y_c, y_top_cut=40):
    """
    Detects true first impact frame t0 by scanning free-fall trajectory.
    Triggers when droplet's bottom reaches y_c - 4 or lateral expansion begins.
    """
    d0_ws = []
    t_contact = None
    for idx in range(min(len(files), 140)):
        im = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        diff = cv2.subtract(bg, im)
        sil = ((diff[y_top_cut:y_c + 1, 150:1130] > 25) & (bg[y_top_cut:y_c + 1, 150:1130] > 55)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 600]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            real_bot = y + y_top_cut + h
            if real_bot < y_c - 12:
                d0_ws.append(w)
            mean_w = np.median(d0_ws) if len(d0_ws) > 0 else 100.0
            if real_bot >= y_c - 3 or w >= 1.25 * mean_w:
                t_contact = idx
                break
    return t_contact


def reconstruct_water_droplet_contour(im, bg, y_contact, y_top_cut, y_effective_clip=None):
    """
    Robust segmentation and contour reconstruction for pure water droplets,
    preserving transparent liquid cores while strictly rejecting thin horizontal
    surface shimmer slivers and isolated frame border noise.
    """
    clip_y = y_contact if y_effective_clip is None else min(y_contact, int(y_effective_clip))
    diff = cv2.subtract(bg, im)

    sil = ((diff > 25) & (bg > 55)).astype(np.uint8) * 255
    sil[clip_y + 1:, :] = 0
    sil[:y_top_cut, :] = 0
    sil[:, :20] = 0
    sil[:, im.shape[1] - 20:] = 0

    split_y = max(y_top_cut + 2, clip_y - 12)
    top_part = cv2.morphologyEx(sil[:split_y, :], cv2.MORPH_CLOSE, np.ones((5, 45), np.uint8))
    bot_part = cv2.morphologyEx(sil[split_y:, :], cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
    combined = np.vstack([top_part, bot_part])

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        # Strictly reject thin horizontal shimmer slivers (e.g. height < 11 and aspect ratio > 4.5)
        if bh < 11 and (bw > 25 or bw / max(1, bh) > 4.5):
            continue
        if cv2.contourArea(c) > 180:
            valid_cnts.append(c)

    if not valid_cnts:
        return None, None

    main_c = max(valid_cnts, key=cv2.contourArea)
    mx, my, mw, mh = cv2.boundingRect(main_c)

    droplet_mask = np.zeros_like(im)
    cv2.drawContours(droplet_mask, [main_c], -1, 255, -1)
    for c in valid_cnts:
        if c is main_c:
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cy < my + mh + 6 and cy + ch > my - 6:
            if cx <= mx + mw + 40 and cx + cw >= mx - 40:
                cv2.drawContours(droplet_mask, [c], -1, 255, -1)

    # Scanline horizontal fill ONLY within the bounds of the droplet
    solid = np.zeros_like(im)
    ys = np.where(droplet_mask.any(axis=1))[0]
    for y in ys:
        if y > clip_y:
            continue
        xs = np.where(droplet_mask[y] > 0)[0]
        if len(xs) >= 2:
            solid[y, xs[0]:xs[-1] + 1] = 255
        elif len(xs) == 1:
            solid[y, xs[0]] = 255

    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    solid[clip_y + 1:, :] = 0

    final_cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not final_cnts:
        return None, None
    best_c = max(final_cnts, key=cv2.contourArea)
    return best_c, solid


def process_single_case(case_info):
    h_fld = case_info['height_folder']
    c_fld = case_info['case_folder']
    temp_c = case_info['temp_c']
    d0_eq = float(case_info['d0_eq'])
    scale = float(case_info['scale'])
    mm_per_px = 1.0 / scale
    y_contact = get_calibrated_y_contact(h_fld, temp_c)
    y_top_cut = 120 if "1.3cm" in h_fld else Y_TOP_CUT

    raw_dir = os.path.join(BASE_EXP_DIR, "pure water", h_fld, c_fld)
    out_dir = os.path.join(BASE_ANALYSIS_DIR, "pure water", h_fld, c_fld)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not files:
        return None

    img0 = cv2.imread(files[0])
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    H, W = img0_gray.shape

    sample_idx = np.linspace(0, min(len(files) - 1, 180), 19, dtype=int)
    bg90_gray = np.percentile([cv2.imread(files[i], cv2.IMREAD_GRAYSCALE) for i in sample_idx], 90, axis=0).astype(np.uint8)

    bg = np.zeros_like(img0_gray)
    bg[:y_contact - 4, :] = np.maximum(img0_gray[:y_contact - 4, :], bg90_gray[:y_contact - 4, :])
    bg[y_contact - 4:, :] = img0_gray[y_contact - 4:, :]

    # Dynamically detect true first impact frame
    t0_detected = detect_case_t0(files, bg, y_contact, y_top_cut)
    t0 = t0_detected if t0_detected is not None else int(case_info['t0'])
    t0_preimpact_file = os.path.basename(files[t0])

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

        c, solid = reconstruct_water_droplet_contour(gray, bg, y_contact, y_top_cut, y_effective_clip)
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
                contact_band = solid[max(0, y_contact - 8):y_contact + 1, :]
                xs_c = np.where(contact_band.any(axis=0))[0]
                if len(xs_c) >= 3 and y_bot >= y_contact - 8:
                    best_left_x = int(xs_c[0])
                    best_right_x = int(xs_c[-1])
                    d_contact_px = min(d_spread_px, best_right_x - best_left_x + 1)
                    d_contact_mm = d_contact_px / scale
                    beta_contact = d_contact_px / d0_eq
                    status = "IN CONTACT"
                else:
                    # check slightly wider band if droplet reached within 12 px
                    band_wide = solid[max(0, y_contact - 12):y_contact + 1, :]
                    xs_cw = np.where(band_wide.any(axis=0))[0]
                    if len(xs_cw) >= 5 and y_bot >= y_contact - 12:
                        best_left_x = int(xs_cw[0])
                        best_right_x = int(xs_cw[-1])
                        d_contact_px = min(d_spread_px, best_right_x - best_left_x + 1)
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

        if best_left_x is not None and best_right_x is not None:
            cv2.line(canvas, (best_left_x, y_contact), (best_right_x, y_contact), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.line(canvas, (best_left_x, y_contact - 5), (best_left_x, y_contact + 5), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(canvas, (best_right_x, y_contact - 5), (best_right_x, y_contact + 5), (0, 255, 0), 2, cv2.LINE_AA)

        # Inset zoom
        crop_w, crop_h = 320, 160
        roi_cx = int(x_hint)
        roi_cy = int(y_contact)
        x1 = max(0, min(W - crop_w, roi_cx - crop_w // 2))
        y1 = max(0, min(H - crop_h, roi_cy - crop_h // 2))
        x2, y2 = x1 + crop_w, y1 + crop_h

        inset_crop = canvas[y1:y2, x1:x2]
        if inset_crop.shape[0] > 0 and inset_crop.shape[1] > 0:
            inset_big = cv2.resize(inset_crop, (crop_w * 2, crop_h * 2), interpolation=cv2.INTER_LINEAR)
            ix2 = W - 20
            ix1 = ix2 - crop_w * 2
            iy1 = 20
            iy2 = iy1 + crop_h * 2
            canvas[iy1:iy2, ix1:ix2] = inset_big
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (40, 40, 40), 2)
            cv2.putText(canvas, "Magnified Droplet & Contact View", (ix1 + 10, iy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Telemetry
        box_w, box_h = 420, 260
        bx1, by1 = 15, 15
        bx2, by2 = bx1 + box_w, by1 + box_h
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 255), 2)

        c_color = (0, 255, 0) if status == "IN CONTACT" else (0, 180, 255)
        lines = [
            (f"Case: {h_fld} / {c_fld}", (255, 255, 255), 0.52, 2),
            (f"Frame #{frame_num:03d} ({step_name}) | t = {time_ms:.2f} ms ({CAMERA_FPS:.0f} fps) | {orig_fname}", (180, 180, 180), 0.38, 1),
            (f"Contact (Dc) : {d_contact_px}px | {d_contact_mm:.2f}mm | Beta={beta_contact:.3f} [{status}]", c_color, 0.46, 2),
            (f"Apex H (H) : {height_px}px | {height_mm:.2f}mm | H/D0={norm_height:.3f}", (0, 240, 255), 0.46, 2),
            (f"3D CoM (Hcm) : {com_height_px:.1f}px | {com_height_mm:.2f}mm | Hcm/D0={norm_com_height:.3f}", (255, 100, 255), 0.46, 2),
            (f"Spread (Dsp) : {d_spread_px}px | {d_spread_mm:.2f}mm | Beta_sp={beta_spread:.3f}", (200, 200, 200), 0.42, 1),
            (f"Scale (D0={d0_eq:.1f}px / {STANDARD_DROPLET_DIA_MM:.3f}mm): {scale:.3f} px/mm ({mm_per_px:.5f} mm/px)", (140, 140, 140), 0.38, 1),
        ]
        curr_y = by1 + 25
        for text, color, font_scale, thickness in lines:
            cv2.putText(canvas, text, (bx1 + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            curr_y += 32

        out_fname = f"frame_{rel_step:04d}_{step_name}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_fname), canvas)

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
        't0': t0,
        't0_preimpact_file': t0_preimpact_file,
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
    print(f"[*] STARTING SURGICAL RECTIFICATION FOR PURE WATER (filter: {filter_height or 'ALL'})")
    print("=" * 65)

    df_m = pd.read_csv(MASTER_PLOT2_CSV)
    sub = df_m[df_m['Fluid_Folder'] == 'pure water'].copy()
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
                      f"(y_c={res['y_contact']}, t0={res['t0']}, Max_Dc={res['max_dc']}px/{res['max_dc_mm']}mm) "
                      f"[{len(completed_results)}/{len(tasks)}]")

    print(f"\n[*] All {len(completed_results)} cases processed in {time.time() - t_start:.1f}s.")

    # Update master summaries
    for master_path in [MASTER_PLOT2_CSV, MASTER_PHYSICS_CSV]:
        if not os.path.exists(master_path):
            continue
        df_target = pd.read_csv(master_path)
        for res in completed_results:
            mask = (df_target['Fluid_Folder'] == 'pure water') & \
                   (df_target['Drop_Height_Folder'] == res['height_folder']) & \
                   (df_target['Case_Folder'] == res['case_folder'])
            if mask.any():
                df_target.loc[mask, 'Surface_Level_y_px'] = res['y_contact']
                df_target.loc[mask, 'Contact_Line_y_px'] = res['y_contact']
                df_target.loc[mask, 'Frame_1_Raw_Index_t0'] = res['t0']
                if 'Frame_1_PreImpact_File' in df_target.columns:
                    df_target.loc[mask, 'Frame_1_PreImpact_File'] = res['t0_preimpact_file']
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
                    df_target.loc[mask, 'Max_Droplet_Height_mm'] = res['max_h_mm']
                    df_target.loc[mask, 'Max_Normalized_Height_H_D0'] = res['max_norm_h']
                if 'Max_3D_CoM_Height_px' in df_target.columns:
                    df_target.loc[mask, 'Max_3D_CoM_Height_px'] = res['max_com_px']
                    df_target.loc[mask, 'Max_3D_CoM_Height_mm'] = res['max_com_mm']
                    df_target.loc[mask, 'Max_Normalized_3D_CoM_Hcm_D0'] = res['max_norm_com']

        df_target.to_csv(master_path, index=False)
        print(f"[*] Updated {master_path} successfully.")

    print("\n[SUCCESS] Surgical rectification of Pure Water complete!")


if __name__ == "__main__":
    import sys
    f_h = sys.argv[1] if len(sys.argv) > 1 else None
    main(f_h)
