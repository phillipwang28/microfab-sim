import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# -------------------------------
# 1) Dopant profile plot (semilog)
# -------------------------------
def plot_dopant_profiles(x_um, B, P, out_png, title="Dopant Profiles"):
    plt.figure(figsize=(6, 4))
    plt.semilogy(x_um, B, label="Boron")
    plt.semilogy(x_um, P, label="Phosphorus")
    plt.xlabel("Depth (μm)")
    plt.ylabel("Concentration (atoms/cm³)")
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4, alpha=0.4)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png

# --------------------------------------
# 2) Wafer cross-section (not to scale)
# --------------------------------------
def draw_wafer_cross_section(
    oxide_nm_list,
    oxide_labels=None,
    xj_um=None,
    out_png="figures/wafer_cross_section.png",
    fig_size=(7, 2.2),
):
    """
    oxide_nm_list: list of oxide thicknesses in nm (bottom -> top).
    oxide_labels : list of strings matching oxide_nm_list (optional).
    xj_um       : if set, hatches an n+ region to that depth (visual only).
    """
    if oxide_labels is None:
        oxide_labels = [f"Oxide {i+1}" for i in range(len(oxide_nm_list))]

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    W, H = 6.0, 1.0  # drawing canvas units

    # Silicon base
    si_h = 0.62 * H
    ax.add_patch(
        Rectangle((0, 0), W, si_h, facecolor="#bfbfbf", edgecolor="black", linewidth=1.0)
    )
    ax.text(W - 0.1, 0.03, "Si", ha="right", va="bottom", fontsize=10)

    # Optional n+ shaded region (left third)
    if xj_um is not None and np.isfinite(xj_um) and xj_um > 0:
        # visual mapping: 1 μm depth ~ 90% of Si height
        frac = min(0.9, xj_um / 1.0)
        ax.add_patch(
            Rectangle(
                (0, 0), W * 0.33, si_h * frac, fill=False, hatch="///", linewidth=0, edgecolor=None
            )
        )
        ax.text(W * 0.33 + 0.05, min(si_h * frac + 0.02, si_h - 0.02), f"n+ to ~{xj_um:.2f} μm",
                ha="left", va="bottom", fontsize=9)

    # Oxides on top (scaled visually; not to scale)
    y = si_h
    # scale: map 0–max_nm to 0–0.35H gently
    safe_thks = [t if (t is not None and t >= 0) else 0.0 for t in oxide_nm_list]
    max_nm = max(1.0, max(safe_thks))
    scale = 0.35 * H / (np.sqrt(max_nm) + 1e-9)  # sqrt scaling keeps very thick oxides reasonable

    for t_nm, label in zip(safe_thks, oxide_labels):
        # visual height via sqrt scale; clamp a minimum so lines are visible
        t_vis = max(0.012 * H, np.sqrt(t_nm) * scale)
        ax.add_patch(
            Rectangle(
                (0, y),
                W,
                t_vis,
                facecolor="#d9ecff",
                edgecolor="black",
                linewidth=1.0,
            )
        )
        # annotate thickness
        ax.text(0.04 * W, y + t_vis / 2, f"{label}: {t_nm:.0f} nm", ha="left", va="center", fontsize=9)
        y += t_vis

    # cosmetics
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title("Wafer Cross-Section (not to scale)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png

# --------------------------------------
# 3) Oxidation animation (x(t) supplied)
# --------------------------------------
def animate_oxidation(times_min, thickness_nm, out_mp4="figures/oxidation.mp4", fps=24):
    """
    times_min: 1D array of times [min]
    thickness_nm: 1D array same length; oxide thickness vs time
    Renders a simple growing oxide slab on silicon.
    """
    times_min = np.asarray(times_min)
    thickness_nm = np.asarray(thickness_nm)
    assert times_min.shape == thickness_nm.shape and times_min.ndim == 1

    fig, ax = plt.subplots(figsize=(7, 2.2))
    W, H = 6.0, 1.0
    si_h = 0.62 * H

    # static silicon base
    si_rect = Rectangle((0, 0), W, si_h, facecolor="#bfbfbf", edgecolor="black", linewidth=1.0)
    ax.add_patch(si_rect)

    # dynamic oxide rectangle (height updated per frame)
    ox_rect = Rectangle((0, si_h), W, 0.01, facecolor="#d9ecff", edgecolor="black", linewidth=1.0)
    ax.add_patch(ox_rect)

    t_text = ax.text(W - 0.05, 0.03, "", ha="right", va="bottom", fontsize=10)

    # pre-compute visual scale
    max_nm = max(1.0, float(thickness_nm.max()))
    scale = 0.35 * H / (np.sqrt(max_nm) + 1e-9)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title("Field Oxidation (not to scale)")

    def update(i):
        t_nm = thickness_nm[i]
        h_vis = max(0.012 * H, np.sqrt(max(t_nm, 0.0)) * scale)
        ox_rect.set_height(h_vis)
        t_text.set_text(f"t = {times_min[i]:.1f} min, x ≈ {t_nm:.0f} nm")
        return (ox_rect, t_text)

    frames = len(times_min)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    try:
        anim.save(out_mp4, dpi=200)
    finally:
        plt.close(fig)
    return out_mp4