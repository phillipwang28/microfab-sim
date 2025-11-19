import os, csv
import numpy as np
from deal_grove import deal_grove_thickness
from diffusion import *
from visual import plot_dopant_profiles, draw_wafer_cross_section
from datetime import datetime

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Oxides (B, B/A values determined from 1965 Deal-Grove Paper)
# wet oxidation
field = deal_grove_thickness(B_nm2_per_min=4600, B_over_A_nm_per_min=20.354,  t_min=100)

# dry oxidation
gate  = deal_grove_thickness(B_nm2_per_min=450,   B_over_A_nm_per_min=5,    t_min=50)
inter = deal_grove_thickness(B_nm2_per_min=322.5, B_over_A_nm_per_min=2.53, t_min=40)

# Build a suffix describing this run (rounded nm values)
field_nm = int(round(field))
gate_nm  = int(round(gate))
inter_nm = int(round(inter))

# Optional: timestamp so files are always unique
ts = datetime.now().strftime("%Y%m%d-%H%M%S")

suffix = f"field{field_nm}nm_gate{gate_nm}nm_inter{inter_nm}nm_{ts}"

with open("data/oxide_table.csv","w",newline="", encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["Parameter","Value"])
    w.writerow(["Field oxide (nm)", f"{field:.1f}"])
    w.writerow(["Gate oxide (nm)", f"{gate:.1f}"])
    w.writerow(["Intermediate oxide (nm)", f"{inter:.1f}"])

# Dopants (dummy values)
x_um = np.linspace(0, 2.0, 2001)
NA_bg = 1e15*np.ones_like(x_um)  # wafer p background

# Phosphorus diffusion (constant source at 1000C for 20 min)
D_P = D_cm2_s(D0=10.5, Ea_eV=3.69, T_C=1000)
P = const_source_erfc(Cs=1e21, D=D_P, t_s=20*60, x_um=x_um)

# Boron implant + anneal, then add background
B_impl = implant_gaussian(dose_cm2=5e13, Rp_um=0.05, dR_um=0.02, x_um=x_um)
D_B   = D_cm2_s(D0=0.76, Ea_eV=3.46, T_C=1000)
B_imp_anneal = anneal_broaden(B_impl, D_B, 30*60, x_um)
B = np.minimum(1e21, NA_bg + B_imp_anneal)

# Junction depth vs background
xj = junction_depth(x_um, NA_bg, P)

# Build unique filenames that encode oxide parameters
dopant_png = f"figures/dopant_profiles_{suffix}.png"
wafer_png  = f"figures/wafer_cross_section_{suffix}.png"

# DEBUG: show exactly where files are going  # NEW
print("Writing dopant plot to:", os.path.abspath(dopant_png))     # NEW
print("Writing wafer cross-section to:", os.path.abspath(wafer_png))  # NEW

# Save figure
plot_dopant_profiles(x_um, B, P, dopant_png)

# Save wafer view (shows oxide stacks + shades n+ to depth xj)
draw_wafer_cross_section(
    [field, gate, inter],
    oxide_labels=["Field", "Gate", "Intermediate"],
    xj_um=xj,
    out_png=wafer_png,
)

# ---- 3) Aluminum sheet resistance (bulk rho) ----
rho_Al = 2.65e-8  # Ω·m
t_Al   = 100e-9   # 100 nm
Rs_Al  = rho_Al/t_Al

with open("data/doping_table.csv","w",newline="", encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["Parameter","Value"])
    w.writerow(["Boron peak concentration (atoms/cm³)", f"{B.max():.3e}"])
    w.writerow(["Depth of boron peak (μm)", f"{x_um[np.argmax(B)]:.4f}"])
    w.writerow(["Phosphorus peak concentration (atoms/cm³)", f"{P.max():.3e}"])
    w.writerow(["Depth of phosphorus peak (μm)", f"{x_um[np.argmax(P)]:.4f}"])
    w.writerow(["Junction depth (μm)", f"{xj:.4f}"])
    # crude Rs estimates if you want: use mobility model of your choice later

with open("data/al_sheet_resistance.txt","w", encoding="utf-8-sig") as f:
    f.write(f"Aluminum sheet resistance (Ω/□): {Rs_Al:.3f}\n")

print("Done → figures/ and data/")
