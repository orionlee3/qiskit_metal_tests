"""Tunable coupler shaped like a capital H lying on its side (an I-beam).

          <---------- bar_length ---------->
          ==================================   ^ bar_width
                       |        |              v
     slot  ->          |  web   |          <-  slot   ^ slot_height
                       |        |                     v
          ==================================
                          [JJ]                  SQUID below the bottom bar
                         --T--                  flux line T, pin points south

The two slots open to the left and right so a neighbouring qubit's arm tip
(with its own etched pocket) can sit between the bars. All of the coupler's
metal then sits around the arm tips it is meant to couple to, rather than
spending half its area on north/south arms that only couple to the readout
and the ground plane.

The pocket etched around the coupler follows the metal outline (a square-
cornered buffer of pocket_gap), so the slot interiors stay ground plane. That
leaves a strip of ground between the coupler and each qubit's pocket.

The flux line is the same symmetric T as on ``TransmonCrossFL``, so routes
can attach to the ``flux_line`` pin in exactly the same way.
"""

from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent


class TransmonHCoupler(QComponent):
    """I-beam ("sideways H") grounded transmon coupler with a flux line.

    Default Options:
        * bar_length  - length of the top and bottom bars (sets slot reach)
        * bar_width   - thickness of each bar
        * slot_height - opening between the two bars (qubit pocket + ground + gap)
        * web_width   - width of the vertical web joining the bars
        * pocket_gap  - etch between the coupler metal and the ground plane
        * jj_width    - drawn width of the junction / SQUID sheet
        * make_fl     - add the flux line below the SQUID
        * fl_options  - flux-line T options, as on ``TransmonCrossFL``
    """

    default_options = Dict(
        bar_length="210um",
        bar_width="15um",
        slot_height="140um",
        web_width="80um",
        pocket_gap="15um",
        jj_width="20um",
        make_fl=True,
        fl_options=Dict(
            t_top="15um",
            t_offset="0um",
            t_inductive_gap="3um",
            t_width="5um",
            t_gap="3um",
        ),
    )
    """Default drawing options"""

    component_metadata = Dict(
        short_name="TC",
        _qgeometry_table_poly="True",
        _qgeometry_table_path="True",
        _qgeometry_table_junction="True",
    )
    """Component metadata"""

    TOOLTIP = """I-beam (sideways H) tunable coupler with a flux line."""

    def make(self):
        """Define the way the options are turned into QGeometry."""
        self.make_pocket()
        if self.options.make_fl:
            self.make_flux_line()

    def _outer_half_height(self):
        p = self.p
        return p.slot_height / 2 + p.bar_width

    def make_pocket(self):
        """Draw the I-beam island, its conformal etch and the junction."""
        p = self.p
        y_in = p.slot_height / 2
        y_out = self._outer_half_height()

        top_bar = draw.rectangle(p.bar_length, p.bar_width, 0, (y_in + y_out) / 2)
        bottom_bar = draw.rectangle(p.bar_length, p.bar_width, 0, -(y_in + y_out) / 2)
        web = draw.rectangle(p.web_width, p.slot_height, 0, 0)
        island = draw.union([top_bar, bottom_bar, web])

        # join_style=2 (mitre) keeps the etch corners square
        pocket = island.buffer(p.pocket_gap, join_style=2)

        # SQUID bridges the gap from the bottom bar down to ground
        jj = draw.LineString([(0, -y_out), (0, -y_out - p.pocket_gap)])

        polys = [island, pocket, jj]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)
        [island, pocket, jj] = polys

        self.add_qgeometry("poly", dict(island=island), chip=p.chip, layer=p.layer)
        self.add_qgeometry(
            "poly", dict(pocket=pocket), subtract=True, chip=p.chip, layer=p.layer
        )
        self.add_qgeometry(
            "junction", dict(rect_jj=jj), width=p.jj_width, chip=p.chip, layer=p.layer
        )

    def make_flux_line(self):
        """Symmetric T flux line under the SQUID (same as TransmonCrossFL)."""
        p = self.p
        pf = p.fl_options

        h_line = draw.LineString([(-pf.t_top / 2, 0), (pf.t_top / 2, 0)])
        v_line = draw.LineString([(pf.t_offset, 0), (pf.t_offset, -0.03)])
        parts = draw.translate(
            [h_line, v_line],
            0,
            -(
                self._outer_half_height()
                + p.pocket_gap
                + pf.t_inductive_gap
                + pf.t_width / 2
                + pf.t_gap
            ),
        )
        parts = draw.rotate(parts, p.orientation, origin=(0, 0))
        parts = draw.translate(parts, p.pos_x, p.pos_y)
        [h_line, v_line] = parts

        self.add_qgeometry(
            "path",
            {"h_line": h_line, "v_line": v_line},
            width=pf.t_width,
            layer=p.layer,
        )
        self.add_qgeometry(
            "path",
            {"h_line_sub": h_line, "v_line_sub": v_line},
            width=pf.t_width + 2 * pf.t_gap,
            subtract=True,
            layer=p.layer,
        )
        self.add_pin(
            "flux_line",
            points=v_line.coords,
            width=pf.t_width,
            gap=pf.t_gap,
            input_as_norm=True,
        )
