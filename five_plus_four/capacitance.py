"""Quick capacitance estimate for the islands of a planar Qiskit Metal design.

Zero-thickness metal on the boundary between the substrate and vacuum sees
the plain Coulomb potential with the average permittivity (eps_r + 1) / 2, so
the charge on each metal pixel follows from one convolution. The metal in a
window is cut into square pixels, every separate piece of metal is a
conductor, and the charges for "this conductor at 1 V, every other at 0 V"
are solved for with conjugate gradients (FFT matrix-vector products).

The result is the Maxwell capacitance matrix row for each probed conductor:
the diagonal entry is C_sigma (what sets E_C), the off-diagonal entries are
the coupling capacitances (negative, by convention).

Junctions are left out, so each island is its own conductor. Metal cut by the
window edge (resonators, control lines) is held at 0 V along with the ground.
Checked against the thin square plate (C = 0.367 * 4 pi eps a): 0.2 % low at
a 1 um grid. Use it for sizing, then confirm with Q3D / LOM.
"""

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg
from shapely.geometry import box
from shapely.ops import unary_union

EPS0_F_PER_UM = 8.854e-18


def metal_raster(design, window_mm, pixel_um):
    """Boolean metal map of the window (x0, x1, y0, y1) in mm, pixel centres."""
    x0, x1, y0, y1 = window_mm
    etch, metal = [], []
    for table_name, table in design.qgeometry.tables.items():
        if table_name == "junction":
            continue
        for _, row in table.iterrows():
            geom = row.geometry
            if table_name == "path":
                geom = geom.buffer(row.width / 2, cap_style=2, join_style=2)
            (etch if bool(row["subtract"]) else metal).append(geom)
    win = box(x0, y0, x1, y1)
    all_metal = unary_union(
        [win.difference(unary_union(etch)), unary_union(metal).intersection(win)]
    )
    step = pixel_um / 1e3
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    shapely.prepare(all_metal)
    return shapely.contains_xy(all_metal, *np.meshgrid(xs, ys)), xs, ys


def capacitance_rows(design, window_mm, probes_mm, pixel_um=1.5, eps_r=11.45):
    """Maxwell capacitance rows (fF) for the conductors under each probe point.

    Args:
        design: built Qiskit Metal design
        window_mm: (x0, x1, y0, y1) region to solve, mm
        probes_mm: {name: (x, y)}, a point inside each conductor of interest
        pixel_um: grid size; keep it under half the narrowest gap or strip

    Returns:
        {name: {other_name: C (fF)}}; rows[name][name] is C_sigma
    """
    mask, xs, ys = metal_raster(design, window_mm, pixel_um)
    labels, _ = ndimage.label(mask)
    ids = {}
    for name, (x, y) in probes_mm.items():
        ids[name] = labels[np.argmin(abs(ys - y)), np.argmin(abs(xs - x))]
        if ids[name] == 0:
            raise ValueError(f"probe {name} at ({x}, {y}) mm is not on metal")

    ny, nx = mask.shape
    h = pixel_um
    iy = np.arange(-ny + 1, ny) * h
    ix = np.arange(-nx + 1, nx) * h
    r = np.hypot(*np.meshgrid(iy, ix, indexing="ij"))
    # Potential at a pixel from unit charge density on another pixel (centre
    # approximation), and on itself (exact for a uniformly charged square)
    kernel = np.where(r > 0, h * h / np.maximum(r, 1e-12), 4 * h * np.log(1 + np.sqrt(2)))
    kernel /= 4 * np.pi * EPS0_F_PER_UM * (eps_r + 1) / 2
    kernel_f = np.fft.rfft2(kernel)
    on = np.flatnonzero(mask)

    def potential(sigma):
        s = np.zeros((ny, nx))
        s.flat[on] = sigma
        v = np.fft.irfft2(np.fft.rfft2(s, s=kernel.shape) * kernel_f, s=kernel.shape)
        return v[ny - 1 : 2 * ny - 1, nx - 1 : 2 * nx - 1].flat[on]

    op = LinearOperator((on.size, on.size), matvec=potential, dtype=float)
    labels_on = labels.flat[on]
    rows = {}
    for name, cid in ids.items():
        sigma, info = cg(op, (labels_on == cid).astype(float), rtol=1e-5, maxiter=3000)
        if info:
            print(f"warning: {name} did not converge ({info})")
        charge_ff = sigma * h * h * 1e15
        rows[name] = {n2: charge_ff[labels_on == c2].sum() for n2, c2 in ids.items()}
    return rows
