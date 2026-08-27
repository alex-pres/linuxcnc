#!/usr/bin/env python3
"""A ``GLCanon`` that ``gcode.parse`` can drive with no running LinuxCNC.

Supplies only the handful of interpreter queries ``StatMixin`` would normally
answer from the live status channel. The standalone ``gcode`` module has no
tool table, so fixtures parsed through here must avoid T/M6 and G43; tests that
need a tool change drive ``change_tool``/``tool_offset`` on the canon directly.

Importing this module needs the built ``gcode`` extension, so the tests that
use it run against a built tree with the run-in-place environment sourced.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "lib", "python"))

import gcode                                            # noqa: E402

import rs274.glcanon                                    # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

#: The two dwell colours the canon appends, plus the three drawn categories.
#: Values are arbitrary but distinct, so a test can tell one from another.
COLORS = {
    "traverse": (0.3, 0.5, 0.5), "traverse_alpha": 1 / 3.,
    "straight_feed": (1.0, 1.0, 1.0), "straight_feed_alpha": 1 / 3.,
    "arc_feed": (1.0, 1.0, 1.0), "arc_feed_alpha": 0.5,
    "dwell": (1.0, 0.5, 0.5), "m1xx": (0.5, 0.5, 1.0),
}


class _Progress:
    def nextphase(self, unused): pass
    def progress(self): pass


class HeadlessCanon(rs274.glcanon.GLCanon):
    """``GLCanon`` with the status-channel queries stubbed out.

    ``axis_mask`` is all nine letters so a fixture may use U/V/W (the foam
    geometries) and A/B/C (the rotary ones) without a second harness.
    """

    def __init__(self, geometry="XYZ", **kw):
        rs274.glcanon.GLCanon.__init__(self, COLORS, geometry, **kw)
        self.progress = _Progress()

    #: pocket -> the 14-tuple ``StatMixin.get_tool`` returns. Empty means no
    #: tool table, which is the standalone gcode module's normal state; a
    #: fixture using ``G43 H<n>`` needs an entry, or the interpreter walks off
    #: the end of one that is not there.
    TOOLS: dict = {}

    def get_external_length_units(self): return 1.0
    def get_external_angular_units(self): return 1.0
    def get_axis_mask(self): return 0x1ff
    def get_block_delete(self): return False
    def get_tool(self, pocket):
        return self.TOOLS.get(int(pocket), (-1,) + (0.0,) * 12 + (0,))


def tool(tool_id, zoffset=0.0, xoffset=0.0, diameter=0.0):
    """One row of a tool table, in the order ``StatMixin.get_tool`` returns."""
    return ((tool_id, xoffset, 0.0, zoffset) + (0.0,) * 6
            + (diameter, 0.0, 0.0) + (0,))


def parse(path, geometry="XYZ", ro=None, **kw):
    """Parse ``path`` (absolute, or a name under ``fixtures/``) into a canon.

    ``ro`` stands in for what the widget hands over in ``set_canon``: the
    rotation offsets the renderer transforms with. It has to be applied
    *before* the parse, because the C side compiles them once, at parse start,
    and converts every point on the way in.

    Note that AXIS and gremlin both reverse the ini's GEOMETRY string before
    using it (``"!CXYZ"`` becomes ``"ZYXC!"``), so a fixture standing in for a
    real config should be given the reversed form.
    """
    if not os.path.isabs(path):
        path = os.path.join(FIXTURES, path)
    canon = HeadlessCanon(geometry, **kw)
    if ro is not None:
        canon.configure_program_geometry(geometry, ro, bool(kw.get("is_foam")))
    with tempfile.NamedTemporaryFile(suffix=".var") as var:
        canon.parameter_file = var.name
        result, seq = gcode.parse(path, canon, "", "")
    if result > gcode.MIN_ERROR:
        raise AssertionError("%s: %s at line %s"
                             % (path, gcode.strerror(result), seq))
    return canon
