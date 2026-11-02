import sys, os
sys.path.append(r"C:\Users\Pablo\Desktop\0_3_Experiment data_leidenfrost")
from rectify_user_requested_baselines import process_single_case

task = {
    'fluid': '40 perc acetone',
    'height_folder': '09_11_2022_15.6cm',
    'case_folder': '100_C001H001S0001',
    'temp_c': 100.0,
    'y_contact': 524,
    'd0_eq_px': 111.0,
    'scale_px_per_mm': 35.5724
}
print("Rectifying 40 perc acetone 15.6cm 100C with y_contact=524...")
res = process_single_case(task)
print("Result:", res)
