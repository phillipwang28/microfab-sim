import math

def deal_grove_thickness(B_nm2_per_min, B_over_A_nm_per_min, t_min):
    # Solve x^2 + (B/A) x - B t = 0  → x = [-A + sqrt(A^2 + 4 B t)]/2
    A_nm = B_nm2_per_min / max(B_over_A_nm_per_min, 1e-12)
    disc = A_nm*A_nm + 4*B_nm2_per_min*t_min
    return (-A_nm + math.sqrt(disc)) / 2.0  # nm