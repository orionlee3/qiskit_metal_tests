"""MIT LL wirebond pads, at the positions of MIT LL's 20-pin package.

HIERARCHY_EXAMPLE_v3_1_1.gds (supplied with the SQP3.1 rules) shows the
standard pad locations of MIT LL's packages; its chip TEST04 is the 20-pin
package: 6 SSMB pads on each of the north and south edges, 4 K-connector pads
on each of the east and west edges. Each pad there is a 160 x 200 um
rectangle (the BP.WL.1.DFM size) whose back is 200 um in from the chip edge
(125 um for the outer pads of the north and south edges), tapering over
200 um to a 10 um trace. TEST01 shows the ground cut round it: 130 um past
each side of the pad, 100 um behind it, and narrowing along the taper to the
CPW gap.

load_package_pads() reads the pads from that file, and MitllLaunchpad draws
one: Metal's LaunchpadWirebond with its own gap behind the pad.
"""

import gdstk
import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond


class MitllLaunchpad(LaunchpadWirebond):
    """MIT LL's wirebond pad. As LaunchpadWirebond: (0, 0) is the end of the
    taper, the pad behind it along -x, the pin ``tie`` at the end of the lead.

    Options (on top of LaunchpadWirebond's):
        * pad_back_gap: ground cut behind the pad (pad_gap is the sides')
    """

    default_options = Dict(
        lead_length="25um",
        pad_width="160um",
        pad_height="200um",
        pad_gap="130um",
        pad_back_gap="100um",
        taper_height="200um",
    )

    component_metadata = Dict(short_name="LP", _qgeometry_table_poly="True")

    def make(self):
        p = self.p
        tw, tg, lead = p.trace_width / 2, p.trace_gap, p.lead_length
        pw, ph, taper = p.pad_width / 2, p.pad_height, p.taper_height
        side, back = p.pad_gap, p.pad_back_gap
        pad = draw.Polygon([
            (0, tw), (-taper, pw), (-(taper + ph), pw), (-(taper + ph), -pw),
            (-taper, -pw), (0, -tw), (lead, -tw), (lead, tw),
        ])
        pocket = draw.Polygon([
            (0, tw + tg), (-taper, pw + side), (-(taper + ph + back), pw + side),
            (-(taper + ph + back), -(pw + side)), (-taper, -(pw + side)),
            (0, -(tw + tg)), (lead, -(tw + tg)), (lead, tw + tg),
        ])
        pin = draw.LineString([(lead, tw), (lead, -tw)])
        pin, pad, pocket = draw.translate(
            draw.rotate([pin, pad, pocket], p.orientation, origin=(0, 0)), p.pos_x, p.pos_y
        )
        self.add_qgeometry("poly", dict(launch_pad=pad), layer=p.layer)
        self.add_qgeometry("poly", dict(pocket=pocket), subtract=True, layer=p.layer)
        self.add_pin("tie", pin.coords, p.trace_width)


def load_package_pads(path, cell="TEST04"):
    """The pads of an MIT LL package chip in the example GDS.

    Returns {edge: [(along_um, back_um), ...]} for edges "N", "E", "S", "W":
    each pad's centre along its edge (from the chip centre; west to east, or
    south to north) and the distance from the chip edge to the back of the
    pad. Coordinates are relative to the chip centre.
    """
    lib = gdstk.read_gds(path)
    chip = next(c for c in lib.cells if c.name == cell).copy(cell + "_flat").flatten()
    (x0, y0), (x1, y1) = next(p for p in chip.polygons if p.layer == 703).bounding_box()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    pads = {"N": [], "E": [], "S": [], "W": []}
    for poly in chip.polygons:
        if (poly.layer, poly.datatype) != (5, 0) or len(poly.points) != 4:
            continue
        (a0, b0), (a1, b1) = poly.bounding_box()
        a0, a1, b0, b1 = a0 - cx, a1 - cx, b0 - cy, b1 - cy
        if not np.isclose(abs(poly.area()), (a1 - a0) * (b1 - b0)):
            continue  # a taper, not a pad
        # The pad's long side points at its edge; the nearest edge is the one.
        edges = {"N": half - b1, "S": b0 + half, "E": half - a1, "W": a0 + half}
        edge = min(edges, key=edges.get)
        along = (a0 + a1) / 2 if edge in "NS" else (b0 + b1) / 2
        pads[edge].append((round(along, 3), round(edges[edge], 3)))
    # The example draws every pad polygon twice.
    return {edge: sorted(set(found)) for edge, found in pads.items()}


def pad_placement(edge, along_um, back_um, chip_size_mm, **options):
    """MitllLaunchpad options for a pad from load_package_pads(), on a square
    chip centred on the origin: the pad's back back_um in from the edge,
    facing the chip centre. Extra options (in um, e.g. taper_height="100um")
    override the defaults and are passed on."""
    o = {**MitllLaunchpad.default_options, **options}
    depth_um = back_um + sum(float(o[k].removesuffix("um")) for k in ("pad_height", "taper_height"))
    inset_mm = chip_size_mm / 2 - depth_um / 1e3  # chip centre to the taper's end
    along_mm = along_um / 1e3
    x, y, orientation = {
        "N": (along_mm, inset_mm, "270"),
        "S": (along_mm, -inset_mm, "90"),
        "E": (inset_mm, along_mm, "180"),
        "W": (-inset_mm, along_mm, "0"),
    }[edge]
    return dict(options, pos_x=f"{x:.4f}mm", pos_y=f"{y:.4f}mm", orientation=orientation)
