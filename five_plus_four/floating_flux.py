"""Floating flux line: a hairpin that passes under a SQUID and never touches ground.

          ground (pocket edge, SQUID just above)
          - - - - - - - - - - - - - - - - - -     <- inductive_gap of ground
             +=======================+            <- bias current runs along here
             ||                     ||
             ||                     ||
          west_leg               east_leg        pins point south

Current goes in one launchpad, along the top segment under the SQUID, and out
through a second launchpad. With no connection to ground on chip, the return
current flows back through the other leg instead of spreading through the
ground plane, where it would thread neighbouring SQUIDs (flux crosstalk).

``pos_x``/``pos_y`` is the point on the pocket edge directly under the SQUID,
so the segment sits at the same place as the T of ``TransmonCrossFL``'s
grounded flux line. Keep ``leg_pitch`` comfortably wider than the routed CPW
etch (cpw_width + 2 * cpw_gap) so the ground between the legs stays connected.
"""

from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent


class FloatingFluxLine(QComponent):
    """Hairpin flux line with two pins, for a floating (differential) bias.

    Default Options:
        * width          - trace width
        * gap            - etch on each side of the trace
        * inductive_gap  - ground left between the pocket edge and the etch
        * leg_pitch      - centre-to-centre spacing of the two legs
        * leg_length     - how far the legs run down before the pins
    """

    default_options = Dict(
        width="5um",
        gap="3um",
        inductive_gap="3um",
        leg_pitch="40um",
        leg_length="30um",
    )
    """Default drawing options"""

    component_metadata = Dict(short_name="FFL", _qgeometry_table_path="True")
    """Component metadata"""

    TOOLTIP = """Floating hairpin flux line (two launchpads, no ground return)."""

    def make(self):
        """Define the way the options are turned into QGeometry."""
        p = self.p
        a = p.leg_pitch / 2
        y_seg = -(p.inductive_gap + p.gap + p.width / 2)
        y_pin = y_seg - p.leg_length

        hairpin = draw.LineString([(-a, y_pin), (-a, y_seg), (a, y_seg), (a, y_pin)])
        west_leg = draw.LineString([(-a, y_seg), (-a, y_pin)])
        east_leg = draw.LineString([(a, y_seg), (a, y_pin)])

        parts = [hairpin, west_leg, east_leg]
        parts = draw.rotate(parts, p.orientation, origin=(0, 0))
        parts = draw.translate(parts, p.pos_x, p.pos_y)
        [hairpin, west_leg, east_leg] = parts

        self.add_qgeometry("path", {"hairpin": hairpin}, width=p.width, layer=p.layer)
        self.add_qgeometry(
            "path",
            {"hairpin_sub": hairpin},
            width=p.width + 2 * p.gap,
            subtract=True,
            layer=p.layer,
        )
        for name, leg in (("west_leg", west_leg), ("east_leg", east_leg)):
            self.add_pin(
                name, points=leg.coords, width=p.width, gap=p.gap, input_as_norm=True
            )
