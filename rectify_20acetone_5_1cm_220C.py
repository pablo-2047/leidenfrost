"""
Surgical Rectification Script for ONLY:
  Water + 20% Acetone, 5.1 cm height, 220°C
  (20 perc acetone / 09_11_2022_5.1cm / 220_C001H001S0001)

Why the original analyze_case.py failed on this case:
  1. Baseline Inaccuracy (y_contact = 726 px was 3 pixels too low):
     At 220°C (transition/Leidenfrost boiling regime), a thermal shimmer and specular mirror-reflection
     band exists on rows y = 724..726 on the heated polished substrate. The original code set
     y_contact = 726, cutting right into this reflection and shadow layer.
  2. First Contact Touchdown Inflation (Frame 2 / t0+01, 0.25 ms):
     When the incoming spherical droplet touched down, specular reflection flare along y = 726
     spread horizontally from x = 621 to x = 746. Unbounded horizontal morphology and dilation
     spread this further to x = 619..748, reporting Dc = 130 px (Beta = 1.000, 3.120 mm)!
     This erroneously reported that on the very first contact frame (0.25 ms), the contact diameter
     had already instantly equaled the full pre-impact diameter D0 = 130 px of the entire spherical droplet!
     In reality, rows above the shimmer layer (y = 715..720) show an incoming droplet body width of
     92-94 px, and at the true contact interface (Y_CONTACT = 723), the actual liquid contact neck is
     Dc = 96 px (Beta = 0.738).
  3. Recoil Arch Bridging (Frames 28 and 29 / t0+27 and t0+28):
     During droplet recoil, the retracting droplet formed a central liquid body and elevated lateral lobes
     separated by a 49-pixel wide open white air arch (x = 680..728). Unbounded full-row scanline filling
     bridged across this open air space, falsely reporting Dc = 189 px (instead of the genuine 84 px central neck).
  4. Late Recoil Column Shadow Bridging (Frames 32 to 37):
     As the droplet pulled into a tall recoiling vertical column (Apex H = 105..175 px), row y = 726
     bridged the central stem with the substrate reflection flare under the hovering outer lobes,
     artificially sustaining Dc between 117 and 142 px, before instantly collapsing to 0 at Frame 38.

Surgical Fix:
  - Sets the baseline strictly to Y_CONTACT = 723 px (above the y = 724..726 reflection/shimmer band,
    leaving 0 false positives during airborne pre-impact and liftoff).
  - Topological exterior flood-fill reconstructs the true droplet silhouette without bridging open air arches.
  - Contact detection strictly requires verified liquid at Y_CONTACT = 723 supported by genuine droplet body
    directly above in rows 715..720, eliminating horizontal specular flares.
  - Internal vapor bubbles up to 31 px within the continuous puddle are bridged, while wide open air arches
    (> 35 px) during recoil remain completely unbridged.
  - Frame 2 touchdown is accurately measured as Dc = 96 px (Beta = 0.738), Frames 28-29 recoil arch as Dc = 84 px,
    column necking cleanly tracks down to Frame 37, and Frame 38 cleanly registers liftoff (Dc = 0, AIRBORNE).
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import analyze_case as ac
from analyze_case import detect_substrate_tilt

INPUT_DIR = r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost\20 perc acetone\09_11_2022_5.1cm\220_C001H001S0001"
OUTPUT_DIR = r"C:\Users\Pablo\Desktop\Analysis\20 perc acetone\09_11_2022_5.1cm\220_C001H001S0001"
MASTER_PHYSICS_CSV = r"C:\Users\Pablo\Desktop\Analysis\plots\master_summary_with_physics.csv"
PLOT2_RECTIFIED_CSV = r"C:\Users\Pablo\Desktop\Analysis\plot_2\master_summary_plot_2_rectified.csv"

Y_CONTACT = 723  # Raised by 3 px from 726 to 723 to sit strictly above the y=724..726 reflection/shimmer floor


def reconstruct_5_1cm_220C_contour(im, bg, img0, y_contact, y_top_cut, is_contact_frame=True):
    """
    Reconstructs the true outer silhouette of the primary droplet and any airborne satellite fragments
    above y_contact = 723 using topological interior hole filling.
    Prevents unbounded bridging across open white air and measures strictly the true contiguous
    liquid contact foot of the droplet at Y_CONTACT = 723.
    """
    H, W = im.shape[:2]
    diff = cv2.subtract(bg, im)
    diff0 = cv2.subtract(img0, im)

    sil = ((diff > 40) & (im < 135) & (bg > 55)).astype(np.uint8) * 255
    sil[y_contact + 1:, :] = 0
    sil[:y_top_cut, :] = 0
    sil[:, :300] = 0
    sil[:, 900:] = 0

    # Close small rim breaks (up to 7 px) so internal backlight glare inside lobes is enclosed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, kernel)
    closed[y_contact + 1:, :] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num <= 1:
        return None, [], None, None, None

    # Identify primary droplet (largest connected component) and valid airborne satellites
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_lbl = 1 + int(np.argmax(areas))
    main_mask = (labels == main_lbl).astype(np.uint8) * 255

    sat_contours = []
    for lbl in range(1, num):
        if lbl == main_lbl:
            continue
        area = stats[lbl, cv2.CC_STAT_AREA]
        top_y = stats[lbl, cv2.CC_STAT_TOP]
        h_i = stats[lbl, cv2.CC_STAT_HEIGHT]
        bot_y = top_y + h_i - 1
        # Keep genuine airborne satellite droplets (exclude tiny noise specks or substrate residue)
        if area >= 25 and bot_y < y_contact - 4:
            sat_m = (labels == lbl).astype(np.uint8) * 255
            ff_s = sat_m.copy()
            ff_s_mask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(ff_s, ff_s_mask, (0, 0), 255)
            sat_solid = cv2.bitwise_or(sat_m, cv2.bitwise_not(ff_s))
            cnts_s, _ = cv2.findContours(sat_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts_s:
                sat_contours.append(max(cnts_s, key=cv2.contourArea))

    best_lx, best_rx = None, None
    if is_contact_frame:
        # Check droplet body in clean layer y_contact - 8 to y_contact - 3 (rows 715..720)
        # to ensure contact foot has genuine liquid above it and is not an optical reflection artifact
        has_body_clean = main_mask[y_contact - 8:y_contact - 3, :].any(axis=0)

        # Liquid detection at y_contact:
        # Dense liquid (gray < 95) OR translucent liquid bridge (gray < 125) with clean droplet body directly above
        is_liquid = (diff0 > 50) & (img0 > 120) & ((im < 95) | ((im < 125) & has_body_clean)) & (main_mask > 0)
        strict_foot = is_liquid.astype(np.uint8) * 255
        
        # Require liquid presence strictly at y_contact with body directly above
        band = (strict_foot[y_contact, :] > 0) & has_body_clean

        # Bridge internal vapor bubbles up to 31 px wide inside wetted footprint (only where droplet body is above)
        band_closed = cv2.morphologyEx(band.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8))[0]
        band_closed = (band_closed > 0) & has_body_clean
        band_open = cv2.morphologyEx((band_closed.astype(np.uint8) * 255).reshape(1, -1), cv2.MORPH_OPEN, np.ones((1, 3), np.uint8))[0]
        xs_all = np.where(band_open > 0)[0]

        if len(xs_all) >= 3:
            splits = np.where(np.diff(xs_all) > 1)[0] + 1
            runs = np.split(xs_all, splits)
            best_run = max(runs, key=len)
            if len(best_run) >= 3:
                best_lx, best_rx = int(best_run[0]), int(best_run[-1])
                main_mask[y_contact - 1:y_contact + 1, best_lx:best_rx + 1] = 255
                # Strip any 1-2 px substrate shimmer/reflection tail outside [best_lx..best_rx]
                for r_y in range(y_contact - 2, y_contact + 1):
                    row_xs = np.where(main_mask[r_y, :] > 0)[0]
                    for rx_i in row_xs:
                        if (rx_i < best_lx or rx_i > best_rx) and not main_mask[y_contact - 6:y_contact - 2, rx_i].any():
                            main_mask[r_y, rx_i] = 0

    # Fill enclosed internal glare holes inside the main droplet via floodFill from exterior (0, 0)
    ff = main_mask.copy()
    ff_mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    solid = cv2.bitwise_or(main_mask, holes)
    solid[y_contact + 1:, :] = 0

    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, sat_contours, None, None, None
    best_c = max(cnts, key=cv2.contourArea)
    return best_c, sat_contours, solid, best_lx, best_rx


def run_rectification():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    if not files:
        raise RuntimeError(f"No files found in {INPUT_DIR}")

    orig_csv_path = os.path.join(OUTPUT_DIR, "measurements_data.csv")
    df_orig = pd.read_csv(orig_csv_path)
    d0_eq_px = float(df_orig["PreImpact_D0_px"].iloc[0])
    scale_px_per_mm = float(df_orig["Standard_Scale_px_per_mm"].iloc[0])
    mm_per_px = float(df_orig["mm_per_pixel"].iloc[0])
    std_d0_mm = float(df_orig["Standard_D0_mm"].iloc[0])
    fps = float(df_orig["Camera_FPS"].iloc[0])
    start_idx = int(df_orig["Raw_Frame_Index"].iloc[0])  # 20 (220_C001H001S0001000021.jpg)
    t_contact = start_idx + 1  # 21 (Frame #002 / t0+01, 220_C001H001S0001000022.jpg)

    img0_bgr = cv2.imread(files[0])
    H, W = img0_bgr.shape[:2]
    y_tilt_ref, angle = detect_substrate_tilt(img0_bgr)
    M = cv2.getRotationMatrix2D((W / 2, y_tilt_ref), angle, 1.0)

    def read_deskewed_bgr_and_gray(idx):
        bgr = cv2.imread(files[idx])
        rot = cv2.warpAffine(bgr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        return rot, gray

    rot0, img0 = read_deskewed_bgr_and_gray(0)
    y_top_cut = 135
    y_contact = Y_CONTACT

    sample_idx = np.linspace(0, min(len(files) - 1, 180), 19, dtype=int)
    bg90 = np.percentile([read_deskewed_bgr_and_gray(i)[1] for i in sample_idx], 90, axis=0).astype(np.uint8)
    bg = img0.copy()
    bg[:y_contact - 4, :] = np.maximum(img0[:y_contact - 4, :], bg90[:y_contact - 4, :])

    case_name = os.path.basename(os.path.normpath(INPUT_DIR))
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(INPUT_DIR)))

    records = []
    num_frames = len(df_orig)
    x_hint = float(W / 2.0)

    print(f"Beginning surgical rectification for {parent_dir} / {case_name} ({num_frames} frames)...")

    for i in range(num_frames):
        f_idx = start_idx + i
        frame_num = i + 1
        rel_step = f_idx - start_idx
        step_name = f"t0+{rel_step:02d}"
        time_ms = rel_step * (1000.0 / fps)
        time_s = time_ms / 1000.0
        orig_fname = os.path.basename(files[f_idx])
        out_photo_name = f"frame_{frame_num:04d}_{step_name}_{time_ms:.2f}ms.jpg"

        rot, gray = read_deskewed_bgr_and_gray(f_idx)
        is_contact = (f_idx >= t_contact)

        c, sat_contours, solid, best_left_x, best_right_x = reconstruct_5_1cm_220C_contour(
            gray, bg, img0, y_contact, y_top_cut, is_contact_frame=is_contact
        )

        status = "PRE-IMPACT (AIRBORNE)" if f_idx < t_contact else "AIRBORNE"
        d_contact_px = 0
        d_contact_mm = 0.0
        beta_contact = 0.0
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
        fragment_count = (1 if c is not None else 0) + len(sat_contours)
        burst_status = "BURST" if fragment_count > 1 else ("INTACT" if fragment_count == 1 else "NONE")
        mean_sat_dia_mm = 0.0
        if sat_contours:
            sat_dias = [2.0 * np.sqrt(max(1.0, cv2.contourArea(sc)) / np.pi) / scale_px_per_mm for sc in sat_contours]
            mean_sat_dia_mm = float(np.mean(sat_dias))

        x_com, y_com = None, None
        y_top, x_top = None, None

        if c is not None:
            pts = c[:, 0, :]
            min_y_idx = int(np.argmin(pts[:, 1]))
            y_top = int(pts[min_y_idx, 1])
            x_top = int(np.median(pts[pts[:, 1] == y_top, 0]))
            x_hint = float(np.median(pts[:, 0]))

            height_px = max(0, y_contact - y_top)
            height_mm = height_px / scale_px_per_mm
            norm_height = height_px / d0_eq_px

            ys_solid = np.where(solid.any(axis=1))[0]
            if len(ys_solid) > 0:
                sub_solid = solid[ys_solid, :] > 0
                w_rows = sub_solid.sum(axis=1).astype(np.float64)
                weights = w_rows ** 2
                w_sum_prim = float(weights.sum())
                if w_sum_prim > 0:
                    y_com = float(np.dot(ys_solid, weights) / w_sum_prim)
                    x_indices = np.arange(W, dtype=np.float64)
                    x_mids = (sub_solid @ x_indices) / np.maximum(1.0, w_rows)
                    x_com = float(np.dot(x_mids, weights) / w_sum_prim)
                    com_height_px = max(0.0, float(y_contact) - y_com)
                    com_height_mm = com_height_px / scale_px_per_mm
                    primary_com_height_mm = com_height_mm
                    norm_com_height = com_height_px / d0_eq_px

            d_spread_px = int(pts[:, 0].max() - pts[:, 0].min() + 1)
            d_spread_mm = d_spread_px / scale_px_per_mm
            beta_spread = d_spread_px / d0_eq_px

            if is_contact and best_left_x is not None and best_right_x is not None:
                d_contact_px = int(best_right_x - best_left_x + 1)
                d_contact_mm = d_contact_px / scale_px_per_mm
                beta_contact = d_contact_px / d0_eq_px
                total_contact_mm = d_contact_mm
                status = "IN CONTACT"
                if d_spread_px < d_contact_px:
                    d_spread_px = d_contact_px
                    d_spread_mm = d_contact_mm
                    beta_spread = beta_contact

        canvas = rot.copy()
        cv2.line(canvas, (0, y_contact), (W - 1, y_contact), (0, 0, 255), 1, cv2.LINE_AA)
        if c is not None:
            cv2.drawContours(canvas, [c], -1, (255, 220, 0), 1, cv2.LINE_AA)
        for sc in sat_contours:
            cv2.drawContours(canvas, [sc], -1, (0, 140, 255), 1, cv2.LINE_AA)

        guide_x = None
        if y_top is not None:
            right_most_main = int(c[:, 0, 0].max()) if c is not None else int(x_hint)
            guide_x = int(max(right_most_main, best_right_x if best_right_x is not None else 0) + 24)
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
        if ix1 > 540 and (y_top is None or y_top > iy1 + ih + 15 or int(x_hint) + d_spread_px // 2 < ix1 - 15):
            cv2.rectangle(canvas, (ix1 - 3, iy1 - 26), (ix1 + iw + 3, iy1 + ih + 3), (30, 30, 30), -1)
            cv2.putText(canvas, "Magnified Droplet & Contact View", (ix1 + 6, iy1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            canvas[iy1:iy1 + ih, ix1:ix1 + iw] = inset_zoom
            cv2.rectangle(canvas, (ix1 - 1, iy1 - 1), (ix1 + iw, iy1 + ih), (0, 255, 255), 1)

        if y_top is not None and guide_x is not None:
            top_label_y = max(25, y_top + 14)
            cv2.putText(canvas, f"H = {height_px} px ({height_mm:.2f} mm)",
                        (guide_x + 10, top_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 120, 255), 2, cv2.LINE_AA)
            if y_com is not None:
                com_label_y = max(top_label_y + 20, int(round(y_com)) + 4)
                cv2.putText(canvas, f"Hcm = {com_height_px:.1f} px ({com_height_mm:.2f} mm)",
                            (guide_x + 10, com_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 0, 200), 2, cv2.LINE_AA)

        if status == "IN CONTACT" and best_left_x is not None:
            cv2.putText(canvas, f"Dc = {d_contact_px} px ({d_contact_mm:.2f} mm)",
                        (max(20, best_left_x - 10), min(H - 15, y_contact + 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        box_w, box_h = 540, 238
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (18, 18), (18 + box_w, 18 + box_h), (0, 220, 255), 2)

        cv2.putText(canvas, f"Case: {parent_dir} / {case_name}", (30, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame #{frame_num:03d} ({step_name}) | t = {time_ms:.2f} ms (4000 fps) | {orig_fname}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)

        status_color = (0, 255, 0) if status == "IN CONTACT" else (0, 165, 255)
        cv2.putText(canvas, f"Contact (Dc) : {d_contact_px:3d}px | {d_contact_mm:.2f}mm | Beta={beta_contact:.3f} [{status}]",
                    (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.49, status_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Apex H  (H)  : {height_px:3d}px | {height_mm:.2f}mm | H/D0={norm_height:.3f}",
                    (30, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"3D CoM (Hcm) : {com_height_px:5.1f}px | {com_height_mm:.2f}mm | Hcm/D0={norm_com_height:.3f}",
                    (30, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 130, 255), 2, cv2.LINE_AA)
        drops_tag = f" [Drops:{fragment_count}]" if fragment_count > 1 else ""
        cv2.putText(canvas, f"Spread (Dsp) : {d_spread_px:3d}px | {d_spread_mm:.2f}mm | Beta_sp={beta_spread:.3f}{drops_tag}",
                    (30, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Scale (D0={d0_eq_px:.1f}px / {std_d0_mm:.3f}mm): {scale_px_per_mm:.3f} px/mm ({mm_per_px:.5f} mm/px)",
                    (30, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 255, 170), 1, cv2.LINE_AA)

        out_photo_path = os.path.join(OUTPUT_DIR, out_photo_name)
        cv2.imwrite(out_photo_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        records.append({
            "Frame_Number": int(frame_num),
            "Time_ms": round(time_ms, 4),
            "Time_s": round(time_s, 6),
            "Relative_Step": step_name,
            "Raw_Frame_Index": int(f_idx),
            "Original_Filename": orig_fname,
            "Analysed_Photo_Filename": out_photo_name,
            "Contact_Status": status,
            "Fragment_Count": int(fragment_count),
            "Burst_Status": burst_status,
            "Contact_Diameter_px": int(d_contact_px),
            "Contact_Diameter_mm": round(d_contact_mm, 3),
            "Beta_Contact_Dc_D0": round(beta_contact, 4),
            "Total_MultiDrop_Contact_mm": round(total_contact_mm, 3),
            "Spreading_Diameter_px": int(d_spread_px),
            "Spreading_Diameter_mm": round(d_spread_mm, 3),
            "Beta_Spread_Dsp_D0": round(beta_spread, 4),
            "Droplet_Height_px": int(height_px),
            "Droplet_Height_mm": round(height_mm, 3),
            "Normalized_Height_H_D0": round(norm_height, 4),
            "CoM_Height_px": round(com_height_px, 2),
            "CoM_Height_mm": round(com_height_mm, 3),
            "Normalized_CoM_Height_Hcm_D0": round(norm_com_height, 4),
            "Primary_Core_CoM_Height_mm": round(primary_com_height_mm, 3),
            "Mean_Satellite_Dia_mm": round(mean_sat_dia_mm, 3),
            "True_Contact_Line_y_px": int(y_contact),
            "PreImpact_D0_px": round(d0_eq_px, 2),
            "Standard_D0_mm": round(std_d0_mm, 4),
            "Standard_Scale_px_per_mm": round(scale_px_per_mm, 4),
            "mm_per_pixel": round(mm_per_px, 6),
            "Camera_FPS": int(fps),
        })

    df_new = pd.DataFrame(records)
    df_new.to_csv(orig_csv_path, index=False)
    xlsx_path = os.path.join(OUTPUT_DIR, "measurements_data.xlsx")
    df_new.to_excel(xlsx_path, index=False)
    print(f"[+] Updated {orig_csv_path} and {xlsx_path} ({len(df_new)} frames)")
    print(f"    First 10 post-contact frames Dc (px): {list(df_new.iloc[1:11]['Contact_Diameter_px'])}")
    print(f"    Frames 25..38 Dc (px): {list(df_new.iloc[24:38]['Contact_Diameter_px'])}")
    print(f"    Max Contact Diameter: {df_new['Contact_Diameter_px'].max()} px ({df_new['Contact_Diameter_mm'].max():.3f} mm, Beta_c = {df_new['Beta_Contact_Dc_D0'].max():.3f})")

    for csv_p in [MASTER_PHYSICS_CSV, PLOT2_RECTIFIED_CSV]:
        if os.path.exists(csv_p):
            df_m = pd.read_csv(csv_p)
            mask_m = (
                (df_m["Fluid_Folder"] == "20 perc acetone")
                & (df_m["Drop_Height_Folder"] == "09_11_2022_5.1cm")
                & (df_m["Case_Folder"] == "220_C001H001S0001")
            )
            if mask_m.any():
                df_m.loc[mask_m, "Contact_Line_y_px"] = Y_CONTACT
                df_m.loc[mask_m, "Max_Contact_Dia_px"] = int(df_new["Contact_Diameter_px"].max())
                df_m.loc[mask_m, "Max_Contact_Dia_mm"] = round(float(df_new["Contact_Diameter_mm"].max()), 4)
                df_m.loc[mask_m, "Max_Beta_Contact_Dc_D0"] = round(float(df_new["Beta_Contact_Dc_D0"].max()), 4)
                df_m.loc[mask_m, "Max_Spread_Dia_px"] = int(df_new["Spreading_Diameter_px"].max())
                df_m.loc[mask_m, "Max_Spread_Dia_mm"] = round(float(df_new["Spreading_Diameter_mm"].max()), 4)
                df_m.loc[mask_m, "Max_Beta_Spread_Dsp_D0"] = round(float(df_new["Beta_Spread_Dsp_D0"].max()), 4)
                df_m.loc[mask_m, "Max_Normalized_3D_CoM_Hcm_D0"] = round(float(df_new["Normalized_CoM_Height_Hcm_D0"].max()), 4)
                df_m.to_csv(csv_p, index=False)
                print(f"[+] Surgically updated 1 row in {csv_p}")


if __name__ == "__main__":
    run_rectification()
