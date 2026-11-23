"""TransmonCrossFL with a length of its own for each arm.

Metal's cross has one cross_length for all four arms, and its claw and flux
line are placed from it. Here each arm can have its own length, so the arms
that reach into the couplers can be longer than the readout and SQUID arms.
The etch gap is the same all round, so each arm's pocket tip is its metal tip
plus cross_gap.

Arms are named as in TransmonCross, in the component's own (unrotated) frame:
west (-x), north (+y), east (+x), south (-y). The SQUID and flux line are on
the south arm; a claw goes on the arm its connector_location selects.
"""

from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.qubits.transmon_cross import connector_arm_name
from qiskit_metal.qlibrary.qubits.transmon_cross_fl import TransmonCrossFL

ARMS = ("west", "north", "east", "south")


class TransmonCrossArms(TransmonCrossFL):
    """TransmonCrossFL with per-arm lengths.

    Options (on top of TransmonCrossFL's):
        * arm_lengths: centre-to-tip metal length of each arm (west, north,
          east, south); an empty value means cross_length
    """

    default_options = Dict(arm_lengths=Dict(west="", north="", east="", south=""))

    component_metadata = Dict(
        short_name="Cross",
        _qgeometry_table_poly="True",
        _qgeometry_table_junction="True",
        _qgeometry_table_path="True",
    )

    def arm_length(self, arm):
        """Parsed metal length of one arm."""
        value = self.options.arm_lengths[arm]
        return self.parse_value(value) if value else self.p.cross_length

    def make_pocket(self):
        p = self.p
        w, n, e, s = (self.arm_length(arm) for arm in ARMS)
        cross_line = draw.shapely.ops.unary_union(
            [draw.LineString([(0, n), (0, -s)]), draw.LineString([(e, 0), (-w, 0)])]
        )
        cross = cross_line.buffer(p.cross_width / 2, cap_style=2)
        cross_etch = cross.buffer(p.cross_gap, cap_style=3, join_style=2)
        rect_jj = draw.LineString([(0, -s), (0, -s - p.cross_gap)])
        polys = draw.translate(
            draw.rotate([cross, cross_etch, rect_jj], p.orientation, origin=(0, 0)), p.pos_x, p.pos_y
        )
        cross, cross_etch, rect_jj = polys
        self.add_qgeometry("poly", dict(cross=cross), chip=p.chip)
        self.add_qgeometry("poly", dict(cross_etch=cross_etch), subtract=True, chip=p.chip)
        self.add_qgeometry("junction", dict(rect_jj=rect_jj), width=p.cross_width, chip=p.chip)

    def _with_cross_length(self, arm, make, *args):
        # TransmonCross places claws and the flux line from cross_length:
        # draw them with the length of the arm they sit on.
        saved = self.options.cross_length
        self.options.cross_length = self.options.arm_lengths[arm] or saved
        try:
            make(*args)
        finally:
            self.options.cross_length = saved

    def make_connection_pad(self, name):
        arm = connector_arm_name(self.p.connection_pads[name].connector_location)
        self._with_cross_length(arm, super().make_connection_pad, name)

    def make_flux_line(self):
        self._with_cross_length("south", super().make_flux_line)
