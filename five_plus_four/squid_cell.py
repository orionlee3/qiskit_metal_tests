"""SQUID cells for the GDS export, drawn to the MIT LL SQP3.1 (BISQuit) rules.

Each junction is a lead_width lead, a taper taper_length long down to the
finger, a finger finger_length long, then a Dolan gap before the
counter-electrode; taper + finger + gap is 2.0 um (SE.DJ.L.1/L.2), as in MIT
LL's JJ cells. Two of them sit side by side in the arms of a rectangular loop.
The island lands on its metal through one of MIT LL's claw contacts, the
ground through two, one each side of the flux-line T:

                                             ground metal
                                       +==<|  claw
    island                             |
    metal       +====== arm, J1 =======+
     claw |>====+         loop         |    <- flux T beyond the ground edge
                +====== arm, J2 =======+
                                       |
                                       +==<|  claw
          |<-------------- gap_um -------------->|

Cell coordinates are um, with +x pointing from the island to ground and the
origin at the middle of the gap. The cell must be placed unrotated, on a
junction line that runs along +x, so both fingers point along +x with the
finger to the left of its gap in chip coordinates (SE.DJ.4):
squid_origin_um() checks the line and gives the snapped placement origin.

The claw (load_claw(), from MIT LL's JJconnection cell) needs its keyhole cut
out of the base metal, and the base metal to reach 0.2 um past the claw's SE
on every side: squid_geometry() returns both (m1_cut, m1_add) for the base
metal layout, and sqp_drc() applies them before checking.

Layers: junction metal on QSE1 (20/0), one QSE1_device marker (60/0) per
junction covering exactly its taper + finger + gap. Every vertex is on the
5 nm grid (SE.DEV.1).
"""

import math
import re

import gdstk
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

SE_LAYER = 20  # QSE1, drawing (datatype 0)
SE_DEVICE_LAYER = 60  # QSE1_device (ISE1_device = 160 on an interposer)
M1_LAYER = 5  # QM1
CHIP_EDGE_LAYER = 703  # QChipEdge
GRID_UM = 0.005  # SE.DEV.1
M1_GRID_UM = 0.05  # recommended for base metal (DRM 2.7)
SE_CONTACT_AREA_UM2 = 42.0  # SE.MET.O.1
_EPS = 1e-6
_CELL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def on_grid(x, grid=GRID_UM):
    return round(round(x / grid) * grid, 6)


def _floor_grid(x, grid=GRID_UM):
    return round(math.floor(x / grid + _EPS) * grid, 6)


def validate_cell_name(name):
    if not _CELL_NAME.fullmatch(name):
        raise ValueError(
            f"cell name {name!r}: must start with a letter and use only A-Z, a-z, 0-9, _"
        )


def squid_origin_um(coords_mm, name=""):
    """Placement of the cell for a Metal junction line [(x0, y0), (x1, y1)]
    in mm, which must run along +x (island to ground).

    The island and ground edges are snapped to the base-metal grid, as the
    base metal is when it is written, so the claws sit exactly on them.

    Returns:
        ((x, y), gap_um): the cell origin (um, on the 5 nm grid) and the
        snapped island-to-ground gap
    """
    (x0, y0), (x1, y1) = coords_mm
    if abs(y1 - y0) > 1e-9 or x1 <= x0:
        raise ValueError(
            f"{name}: junction line {coords_mm} does not run along +x; the SQUID "
            "cell is placed unrotated, so its gap would not be along x (SE.DJ.4)"
        )
    xi, xg = on_grid(x0 * 1e3, M1_GRID_UM), on_grid(x1 * 1e3, M1_GRID_UM)
    return (round((xi + xg) / 2, 6), on_grid(y0 * 1e3, M1_GRID_UM)), round(xg - xi, 6)


def load_claw(path, cell_prefix="JJconnection"):
    """MIT LL's claw contact from their GDS: the first cell whose name starts
    with cell_prefix. In its coordinates the base metal lies at x <= 0 and
    the SE reaches out past it along +x; its lead (a 2 um path, left out here)
    starts at x = 1 um.

    Returns:
        dict(se=..., m1=..., footprint=...): shapely polygons of the claw's
        SE, its base metal (the pad with the keyhole cut out) and the pad's
        outline
    """
    lib = gdstk.read_gds(path)
    cell = next((c for c in lib.cells if c.name.startswith(cell_prefix)), None)
    if cell is None:
        raise ValueError(f"{path}: no cell named {cell_prefix}*")
    scale = lib.unit / 1e-6
    se, m1 = [], []
    for p in cell.polygons:
        poly = Polygon([(on_grid(x * scale), on_grid(y * scale)) for x, y in p.points])
        if (p.layer, p.datatype) == (SE_LAYER, 0):
            se.append(poly)
        elif (p.layer, p.datatype) == (M1_LAYER, 0):
            m1.append(poly)
    se, m1 = unary_union(se), unary_union(m1)
    footprint = box(*m1.bounds)
    if m1.geom_type != "Polygon" or se.geom_type != "Polygon":
        raise ValueError(f"{path}: {cell.name} is not one SE and one M1 polygon")
    return dict(se=se, m1=m1, footprint=footprint, name=cell.name)


def squid_geometry(
    finger_um,
    gap_um,
    claw,
    ground_claw_y_um,
    lead_width_um=2.0,
    taper_length_um=0.50,
    finger_length_um=1.36,
    dolan_gap_um=0.140,
    counter_length_um=1.5,
    loop_width_um=10.0,
    loop_length_um=5.0,
    metal_margin_um=0.2,
    name="",
):
    """Shapes of one symmetric SQUID with fingers finger_um wide spanning gap_um.

    Args:
        finger_um: junction finger width, on the 5 nm grid
        gap_um: island-to-ground distance (length of the Metal junction line)
        claw: load_claw() result
        ground_claw_y_um: |y| of the two ground claws; keep them (and
            metal_margin_um around them) clear of the flux-line T's etch
        lead_width_um: leads, arms, bars and counter-electrodes (SE.W.1)
        taper_length_um, finger_length_um, dolan_gap_um: the junction, along x
        counter_length_um: counter-electrode, from the gap to the ground bar
        loop_width_um, loop_length_um: inside size of the loop (across, along)
        metal_margin_um: base metal kept beyond the claw's SE (SE.MET.O.4
            needs 0.1 um)

    Returns:
        dict: nets {"island", "ground"} as shapely polygons, devices (the
        SE_device markers), the junction parts, and m1_add / m1_cut: base
        metal to add round each claw, and its keyhole to cut out
    """
    if abs(on_grid(finger_um) - finger_um) > _EPS:
        raise ValueError(f"{name}: finger {finger_um} um is off the 5 nm grid")
    w = lead_width_um
    jj_len = round(taper_length_um + finger_length_um + dolan_gap_um, 6)
    x_island, x_ground = round(-gap_um / 2, 6), round(gap_um / 2, 6)
    if abs(on_grid(x_ground) - x_ground) > _EPS:
        raise ValueError(f"{name}: gap {gap_um} um puts the metal edges off the 5 nm grid")
    y_arm = loop_width_um / 2 + w / 2  # arm centre lines
    y_out = y_arm + w / 2
    lead_a = round(loop_length_um - jj_len - counter_length_um, 6)
    if lead_a < 0:
        raise ValueError(f"{name}: loop_length_um is too short for the junctions")

    # Claws: island one facing +x at the island edge, ground ones mirrored.
    # MIT LL's 2 um lead leaves each claw 1 um past its metal edge.
    fx0, fy0, _, fy1 = claw["footprint"].bounds
    lead_start = 1.0

    def place(geom, x0, y0, mirror):
        g = affinity.scale(geom, -1, 1, origin=(0, 0)) if mirror else geom
        return affinity.translate(g, x0, y0)

    claws = [(x_island, 0.0, False)] + [(x_ground, s * ground_claw_y_um, True) for s in (-1, 1)]
    if ground_claw_y_um + fy0 - metal_margin_um < y_out + w:
        raise ValueError(f"{name}: ground claws overlap the loop; raise ground_claw_y_um")
    head = box(fx0 - metal_margin_um, fy0 - metal_margin_um, 0, fy1 + metal_margin_um)
    keyhole = claw["footprint"].difference(claw["m1"])

    bar_g = (round(x_ground - lead_start - 2 * w, 6), round(x_ground - lead_start - w, 6))
    bar_i = (round(bar_g[0] - loop_length_um - w, 6), round(bar_g[0] - loop_length_um, 6))
    if bar_i[0] < x_island + lead_start - _EPS:
        raise ValueError(f"{name}: a {gap_um:.2f} um gap is too short for the loop")
    shapes = {
        "island": [
            place(claw["se"], x_island, 0, False),
            box(x_island + lead_start, -w / 2, bar_i[0], w / 2),  # island lead
            box(*bar_i[:1], -y_out, bar_i[1], y_out),
        ],
        "ground": [box(bar_g[0], -ground_claw_y_um - w / 2, bar_g[1], ground_claw_y_um + w / 2)],
    }
    for s in (-1, 1):
        yc = s * ground_claw_y_um
        shapes["ground"] += [
            place(claw["se"], x_ground, yc, True),
            box(bar_g[1], yc - w / 2, x_ground - lead_start, yc + w / 2),
        ]

    x_taper = round(bar_i[1] + lead_a, 6)
    x_finger = round(x_taper + taper_length_um, 6)
    x_tip = round(x_finger + finger_length_um, 6)
    x_counter = round(x_tip + dolan_gap_um, 6)
    junctions = []
    for yc in (-y_arm, y_arm):
        y_lo = _floor_grid(yc - finger_um / 2)
        y_hi = round(y_lo + finger_um, 6)
        # SE.DJ.1 / SE.DJ.2.DFM
        assert min(yc + w / 2 - y_hi, y_lo - (yc - w / 2)) >= 0.25 - _EPS, (
            f"{name}: counter-electrode extends < 0.25 um past the finger (SE.DJ.1)")
        assert max(yc + w / 2 - y_hi, y_lo - (yc - w / 2)) <= 2.0 + _EPS, (
            f"{name}: taper extends > 2 um past the finger (SE.DJ.2.DFM)")
        taper = Polygon([(x_taper, yc - w / 2), (x_finger, y_lo),
                         (x_finger, y_hi), (x_taper, yc + w / 2)])
        finger = box(x_finger, y_lo, x_tip, y_hi)
        if lead_a > 0:
            shapes["island"].append(box(bar_i[1], yc - w / 2, x_taper, yc + w / 2))
        shapes["island"] += [taper, finger]
        shapes["ground"].append(box(x_counter, yc - w / 2, bar_g[0], yc + w / 2))
        junctions.append(dict(
            taper=taper,
            finger=finger,
            gap=box(x_tip, y_lo, x_counter, y_hi),
            device=Polygon([(x_taper, yc - w / 2), (x_finger, y_lo), (x_counter, y_lo),
                            (x_counter, y_hi), (x_finger, y_hi), (x_taper, yc + w / 2)]),
            base_net="ground",
        ))
    return dict(
        name=name,
        finger_um=finger_um,
        nets={k: unary_union(v) for k, v in shapes.items()},
        devices=[j["device"] for j in junctions],
        junctions=junctions,
        m1_add=unary_union([place(head, x, y, m) for x, y, m in claws]),
        m1_cut=unary_union([place(keyhole, x, y, m) for x, y, m in claws]),
        island_head=place(head, x_island, 0, False),
    )


def _gdstk_polygons(geom, layer, datatype=0, grid=GRID_UM):
    """gdstk polygons of a shapely (Multi)Polygon, vertices on the grid."""
    out = []
    for g in getattr(geom, "geoms", [geom]):
        outer = [(on_grid(x, grid), on_grid(y, grid)) for x, y in list(g.exterior.coords)[:-1]]
        out.append(gdstk.Polygon(outer, layer=layer, datatype=datatype))
        if g.interiors:
            holes = [gdstk.Polygon([(on_grid(x, grid), on_grid(y, grid)) for x, y in list(r.coords)[:-1]])
                     for r in g.interiors]
            # gdstk cuts each hole open to the outline; snap the cut vertices too
            cut = gdstk.boolean(out[-1], holes, "not", precision=GRID_UM / 10,
                                layer=layer, datatype=datatype)
            out = out[:-1] + [gdstk.Polygon([(on_grid(x, grid), on_grid(y, grid)) for x, y in c.points],
                                            layer=layer, datatype=datatype) for c in cut]
    return out


def squid_cell(name, finger_um, gap_um, claw, ground_claw_y_um, layer=SE_LAYER,
               datatype=0, sedevice_layer=SE_DEVICE_LAYER, **geometry):
    """One symmetric-SQUID cell (gdstk.Cell) with fingers finger_um wide
    spanning gap_um. Other keyword arguments go to squid_geometry().

    Each side of the SQUID (island claw, lead, bar, arm leads, tapers and
    fingers; counter-electrodes, ground bar and claws) is one polygon, merged
    with a gdstk boolean "or", on layer/datatype; each junction's SE_device
    marker is on sedevice_layer, datatype 0. The claws' base metal (m1_add,
    m1_cut from squid_geometry) is not in the cell: it belongs to the base
    metal layout.
    """
    validate_cell_name(name)
    geo = squid_geometry(finger_um, gap_um, claw, ground_claw_y_um, name=name, **geometry)
    cell = gdstk.Cell(name)
    for net in geo["nets"].values():
        merged = gdstk.boolean(_gdstk_polygons(net, layer, datatype), [], "or",
                               precision=GRID_UM / 10, layer=layer, datatype=datatype)
        if len(merged) != 1:
            raise RuntimeError(f"{name}: one side of the SQUID is {len(merged)} pieces")
        cell.add(*_gdstk_polygons(Polygon(merged[0].points), layer, datatype))
    for dev in geo["devices"]:
        cell.add(*_gdstk_polygons(dev, sedevice_layer, 0))
    return cell


def write_squid_library(path, cells, claw, layer=SE_LAYER, sedevice_layer=SE_DEVICE_LAYER):
    """Write a GDS file (um units, 0.5 nm precision) of SQUID cells.

    Args:
        path: output .gds path
        cells: {cell_name: dict(finger_um=..., gap_um=..., ground_claw_y_um=...,
            other squid_geometry() keywords)}
        claw: load_claw() result
        layer: GDS layer for the junction metal (datatype 0)
        sedevice_layer: GDS layer for the SE_device markers
    """
    lib = gdstk.Library(unit=1e-6, precision=5e-10)
    for name, spec in cells.items():
        lib.add(squid_cell(name, claw=claw, layer=layer, sedevice_layer=sedevice_layer, **spec))
    lib.write_gds(path)


def snap_m1(geom_um):
    """Base metal (shapely, um) snapped to the base-metal grid."""
    return shapely.set_precision(geom_um, M1_GRID_UM)


def write_layout_gds(path, top_name, squids, m1_um, chip_size_um):
    """GDS of a layout for MIT LL: um units, 2000 dbu/um, origin at the chip
    centre, and no shapes in the top cell (the Valid Shape Check flags them).

    Args:
        path: output .gds path
        top_name: top cell name
        squids: {cell_name: (origin_xy_um, gdstk.Cell from squid_cell)}; each
            is placed unrotated at its origin
        m1_um: base metal (shapely, um, already snapped), on QM1 5/0
        chip_size_um: (width, height) of the QChipEdge rectangle (703/0)
    """
    validate_cell_name(top_name)
    lib = gdstk.Library(unit=1e-6, precision=5e-10)
    top = lib.new_cell(top_name)
    metal = lib.new_cell(f"{top_name}_M1")
    metal.add(*_gdstk_polygons(m1_um, M1_LAYER, 0, grid=M1_GRID_UM))
    edge = lib.new_cell(f"{top_name}_CHIPEDGE")
    w, h = chip_size_um
    edge.add(gdstk.rectangle((-w / 2, -h / 2), (w / 2, h / 2), layer=CHIP_EDGE_LAYER))
    top.add(gdstk.Reference(metal), gdstk.Reference(edge))
    for cell_name, (origin, cell) in squids.items():
        lib.add(cell)
        top.add(gdstk.Reference(cell, origin=origin))
    # GDSII's own limit; gdstk's default (199) would split the ground plane
    # at new, off-grid vertices
    lib.write_gds(path, max_points=8190)


def _min_width_ok(geom, width):
    """True if every part of geom is at least `width` wide."""
    opened = geom.buffer(-width / 2 + _EPS, join_style=2).buffer(width / 2 - _EPS, join_style=2)
    return geom.difference(opened).area < 1e-6


def _vertices(geom):
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        yield from g.exterior.coords
        for ring in g.interiors:
            yield from ring.coords


def sqp_drc(squid, m1):
    """Check one squid_geometry() against the SQP3.1 rules it is drawn to.

    Args:
        squid: result of squid_geometry
        m1: shapely geometry of the base metal (um, in the cell's
            coordinates) around the SQUID, flux-line metal included

    Returns:
        list of (rule, status, detail), status "ok", "WARN" or "FAIL"
    """
    out = []

    def check(rule, ok, detail, warn_only=False):
        out.append((rule, "ok" if ok else ("WARN" if warn_only else "FAIL"), detail))

    devices = unary_union(squid["devices"])
    nets = {k: v.difference(devices) for k, v in squid["nets"].items()}
    m1_parts = list(getattr(m1, "geoms", [m1]))
    f = squid["finger_um"]

    # Grid, marker shape
    verts = [p for g in [*squid["nets"].values(), *squid["devices"]] for p in _vertices(g)]
    off = max(abs(c / GRID_UM - round(c / GRID_UM)) * GRID_UM for p in verts for c in p)
    check("SE.DEV.1", off < _EPS, f"all SE / SE_device vertices on 5 nm grid (worst {off * 1e3:.4f} nm)")
    n_v = [len(d.exterior.coords) - 1 for d in squid["devices"]]
    check("SE.DJ.3", all(4 <= n <= 6 for n in n_v), f"SE_device vertices {n_v}")

    for k, j in enumerate(squid["junctions"], start=1):
        fx0, fy0, fx1, fy1 = j["finger"].bounds
        gx0, gy0, gx1, gy1 = j["gap"].bounds
        tag = f"J{k}"
        check("SE.DEV.2", abs(gx1 - gx0 - 0.14) < _EPS and abs(j["gap"].area - (gx1 - gx0) * (gy1 - gy0)) < 1e-9,
              f"{tag} Dolan gap {gx1 - gx0:.3f} um, axis-aligned rectangle")
        dx0, _, dx1, _ = j["device"].bounds
        check("SE.DJ.L.1/2", abs(dx1 - dx0 - 2.0) < _EPS, f"{tag} taper + finger + gap {dx1 - dx0:.3f} um")
        check("SE.DJ.4", fx1 <= gx0 + _EPS and abs(fy0 - gy0) < _EPS and abs(fy1 - gy1) < _EPS,
              f"{tag} finger left of the gap, gap along x")
        se = unary_union(list(squid["nets"].values()))
        check("SE.DJ.5", j["device"].difference(se).symmetric_difference(j["gap"]).area < 1e-9,
              f"{tag} SE_device minus SE is exactly the gap rectangle")
        parts = unary_union([j["taper"], j["finger"], j["gap"]])
        check("SE.DJ.5", j["device"].symmetric_difference(parts).area < 1e-9,
              f"{tag} SE_device is exactly taper + finger + gap")
        base = squid["nets"][j["base_net"]]
        probe = box(gx1, fy0 - 0.25, gx1 + 0.01, fy1 + 0.25)
        check("SE.DJ.1", base.buffer(_EPS).contains(probe), f"{tag} base extends >= 0.25 um past finger in y")
        _, ty0, _, ty1 = j["taper"].bounds
        past = max(fy0 - ty0, ty1 - fy1)
        check("SE.DJ.2.DFM", past <= 2.0 + _EPS, f"{tag} taper extends {past:.3f} um past finger in y", True)
    check("SE.DJ.W.1", 0.100 - _EPS <= f <= 10.0, f"finger width {f:.3f} um")
    check("SE.DJ.W.1.DFM", f >= 0.200 - _EPS, f"finger width {f:.3f} um (DFM >= 0.200)", True)

    d = devices.distance(m1)
    check("SE.DEV.MET.S.1", d >= 2.0 - _EPS, f"SE_device to M1 {d:.2f} um (>= 2.0, 4.0 recommended)")
    if d >= 2.0 - _EPS:
        check("SE.DEV.MET.S.1 (rec.)", d >= 4.0 - _EPS, f"SE_device to M1 {d:.2f} um", True)

    # Leads and wiring, outside SE_device
    for net, g in nets.items():
        check("SE.W.1", _min_width_ok(g, 0.5) and g.buffer(-5.0 - _EPS).is_empty,
              f"{net} SE width within 0.5-10 um")
    d = nets["island"].distance(nets["ground"])
    check("SE.S.1", d >= 2.0 - _EPS, f"island-side to ground-side SE {d:.3f} um")
    whole = unary_union([*squid["nets"].values(), devices])
    holes = [Polygon(r) for g in getattr(whole, "geoms", [whole]) for r in g.interiors]
    check("SE.S.1", len(holes) == 1 and _min_width_ok(holes[0], 2.0),
          "SQUID loop arms >= 2 um apart")
    for net, g in nets.items():
        r = 1.0 - 1e-4
        closed = g.buffer(r, join_style=2, mitre_limit=10).buffer(-r, join_style=2, mitre_limit=10)
        notch = max((p.area for p in getattr(closed.difference(g), "geoms", [closed.difference(g)])),
                    default=0.0)
        check("SE.S.1", notch < 1e-3, f"{net} SE notches >= 2 um (largest gap filled {notch:.4f} um^2)")

    # Contacts and SE-to-metal spacing
    for net, g in squid["nets"].items():
        touched = [p for p in m1_parts if p.intersects(g)]
        others = [p for p in m1_parts if not p.intersects(g)]
        if others:
            d = nets[net].distance(unary_union(others))
            check("SE.MET.S.1", d >= 2.0 - _EPS, f"{net} SE to other M1 {d:.2f} um")
        # SE off the metal stays 2 um from every metal edge, except where it
        # runs out of a contact (overlapping is permitted). The 10 nm on the
        # exclusion covers the buffer's polygonised arcs.
        free = nets[net].difference(m1).difference(g.intersection(m1).buffer(2.01))
        d = free.distance(m1) if not free.is_empty else math.inf
        check("SE.MET.S.1", d >= 2.0 - 1e-4, f"{net} SE away from its contacts to M1 {min(d, 99):.2f} um")
        if len(touched) != 1:
            check("SE.MET.O", False, f"{net} SE touches {len(touched)} M1 shapes, expected 1")
            continue
        metal = touched[0]
        for k, ov in enumerate(getattr(g.intersection(metal), "geoms", [g.intersection(metal)])):
            tag = f"{net} contact {k + 1}"
            check("SE.MET.O.1", ov.area >= SE_CONTACT_AREA_UM2 - _EPS, f"{tag} overlap {ov.area:.1f} um^2")
            check("SE.MET.O.2", _min_width_ok(ov, 2.0), f"{tag} overlap >= 2.0 um wide")
            n_ov = len(shapely.simplify(ov, 0).exterior.coords) - 1
            check("SE.MET.O.5/O.6", n_ov >= 7, f"{tag} enclosed metal has {n_ov} vertices (>= 7)")
            if n_ov >= 7:
                check("SE.MET.O.5/O.6 (rec.)", n_ov >= 12, f"{tag} {n_ov} vertices (12 recommended)", True)
        # Where the SE outline crosses the metal edge. Away from those points,
        # metal edges under the SE must lie >= 0.5 um inside it (O.3) and SE
        # edges over the metal >= 0.1 um inside the metal (O.4).
        crossing = g.boundary.intersection(metal.boundary)
        coincident = sum(p.length for p in getattr(crossing, "geoms", [crossing])
                         if p.geom_type in ("LineString", "MultiLineString"))
        check("SE.MET.O.3/O.4", coincident < _EPS, f"{net} coincident SE and M1 edges {coincident:.3f} um")
        m1_under = metal.boundary.intersection(g).difference(crossing.buffer(0.5))
        d3 = m1_under.distance(g.boundary) if not m1_under.is_empty else math.inf
        check("SE.MET.O.3", d3 >= 0.5 - 1e-4, f"{net} SE extends {min(d3, 99):.2f} um past the metal edge")
        se_over = g.boundary.intersection(metal).difference(crossing.buffer(0.1))
        d4 = se_over.distance(metal.boundary) if not se_over.is_empty else math.inf
        check("SE.MET.O.4", d4 >= 0.1 - 1e-4, f"{net} metal extends {min(d4, 99):.2f} um past SE")
    return out
