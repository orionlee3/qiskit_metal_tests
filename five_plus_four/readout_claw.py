"""The readout claw of a TransmonCross, drawn on its own.

The claw (a U of metal around the tip of the qubit's north arm, plus the short
CPW stub its resonator connects to) is drawn exactly where TransmonCross /
TransmonCrossFL would draw it, but the qubit and its etched pocket are left
out: the ground plane fills their place. Seen from the claw at the readout
frequencies, the qubit's arm is close to ground anyway (its capacitance to
ground is much larger than its capacitance to the claw), so the resonator sees
nearly the same load as on the full chip.

Where the claw sits depends only on the qubit's etched pocket, not on the
metal arm inside it, so it takes the pocket's arm width and its tip distance
(qubit centre to the etched tip of the arm) instead of the cross dimensions.
"""

from qiskit_metal import Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross


class ReadoutClaw(TransmonCross):
    """Readout claw only; pos_x/pos_y is the centre of the qubit it reads out.

    Options (on top of TransmonCross's connection_pads options):
        * pocket_width: width of the qubit's etched arm pocket
        * pocket_tip: qubit centre to the etched tip of the arm
    """

    default_options = Dict(pocket_width="89.1um", pocket_tip="260um")

    component_metadata = Dict(short_name="Claw", _qgeometry_table_poly="True")

    def make(self):
        # TransmonCross places the claw from cross_length + cross_gap (the
        # pocket tip) and sizes it from cross_width + 2 cross_gap (the pocket
        # width). A zero gap gives the same numbers from the pocket alone.
        self.options.cross_width = self.options.pocket_width
        self.options.cross_gap = "0um"
        self.options.cross_length = self.options.pocket_tip
        self.make_connection_pads()
