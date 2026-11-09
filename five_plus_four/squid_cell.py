"""SQUID cells for the GDS export, built from the reference Dolan junction.

Each junction is drawn like the reference cell: a lead_width lead, a taper
taper_length long down to the finger, a finger finger_length long, then a
bridge_gap of resist bridge before the counter-electrode. Two of them sit
side by side in the arms of a rectangular loop:

    island                                         ground
    metal |                                       |  metal
          |      +======== arm, junction J1 ======+=+
       ===|======+                                | |  <- ground bar sits on
          |      |           loop                 | |     the ground metal
          |      +======== arm, junction J2 ======+=+
          |<--------------- gap_um -------------->|

Cell coordinates are um, with +x pointing from the island to ground (the
direction of the Metal junction line) and the origin at the middle of the gap,
which is where the GDS renderer places the cell. The loop's ground-side bar
lies on the strip of ground between the pocket and the flux line, so the loop
is as close to the flux line as it can be; keep ground_overlap_um under the
flux line's inductive gap. Both fingers point along +x, the evaporation axis.
"""

import gdstk


def squid_cell(
    name,
    finger_um,
    gap_um,
    layer=2,
    datatype=0,
    lead_width_um=2.0,
    taper_length_um=0.48,
    finger_length_um=1.36,
    bridge_gap_um=0.15,
    island_overlap_um=4.0,
    ground_overlap_um=2.0,
    loop_width_um=10.0,
    loop_length_um=8.0,
):
    """One symmetric-SQUID cell with fingers finger_um wide spanning gap_um.

    Args:
        name: cell name, matching the component's gds_cell_name
        finger_um: junction finger width
        gap_um: island-to-ground distance (length of the Metal junction line)
        layer, datatype: GDS layer for the junction metal
        lead_width_um, taper_length_um, finger_length_um, bridge_gap_um: the
            reference junction's lead, taper, finger and bridge
        island_overlap_um: how far the island lead runs onto the island
        ground_overlap_um: how far the ground-side bar runs onto the ground
        loop_width_um, loop_length_um: inside size of the loop (across, along),
            from the island-side bar to the ground edge

    Returns:
        gdstk.Cell
    """
    w = lead_width_um
    x_ground = gap_um / 2
    x_island = -gap_um / 2
    y_arm = loop_width_um / 2 + w / 2  # arm centre lines

    bar_b = (x_ground + ground_overlap_um - w, x_ground + ground_overlap_um)
    bar_a = (bar_b[0] - loop_length_um - w, bar_b[0] - loop_length_um)
    if bar_a[0] <= x_island + 0.5:
        raise ValueError(
            f"{name}: a {gap_um:.2f} um gap is too short for the loop; "
            "shrink loop_length_um"
        )
    y_out = y_arm + w / 2

    cell = gdstk.Cell(name)

    def rect(x0, x1, y0, y1):
        cell.add(gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype))

    rect(x_island - island_overlap_um, bar_a[0], -w / 2, w / 2)  # island lead
    rect(*bar_a, -y_out, y_out)
    rect(*bar_b, -y_out, y_out)

    # Junction centred along each arm: lead, taper, finger, bridge, counter-electrode
    junction_len = taper_length_um + finger_length_um + bridge_gap_um
    x_taper = bar_a[1] + (loop_length_um - junction_len) / 2
    x_finger = x_taper + taper_length_um
    x_tip = x_finger + finger_length_um
    x_counter = x_tip + bridge_gap_um
    for yc in (-y_arm, y_arm):
        rect(bar_a[1], x_taper, yc - w / 2, yc + w / 2)
        cell.add(
            gdstk.Polygon(
                [
                    (x_taper, yc - w / 2),
                    (x_finger, yc - finger_um / 2),
                    (x_finger, yc + finger_um / 2),
                    (x_taper, yc + w / 2),
                ],
                layer=layer,
                datatype=datatype,
            )
        )
        rect(x_finger, x_tip, yc - finger_um / 2, yc + finger_um / 2)
        rect(x_counter, bar_b[0], yc - w / 2, yc + w / 2)
    return cell


def write_squid_library(path, cells, layer=2):
    """Write a GDS file (um units) of SQUID cells.

    Args:
        path: output .gds path
        cells: {cell_name: dict(finger_um=..., gap_um=...)}
        layer: GDS layer for the junction metal
    """
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    for name, spec in cells.items():
        lib.add(squid_cell(name, spec["finger_um"], spec["gap_um"], layer=layer))
    lib.write_gds(path)
