#!/usr/bin/env python3
"""The program's recorded order, checked against the program array.

``GLCanon`` used to keep ``traverse``/``feed``/``arcfeed`` and add
``moves``/``move_cats``: the same tuple objects in emission order plus a byte
naming each one's category. ``retire-canon-move-lists`` removed all five - the
array is now the only record, and these tests pin the same three properties
against it directly:

  * the array's ``kinds``/``lines`` interleave the categories in emission
    order, with no move added or dropped;
  * the recorded order really does interleave the categories along one
    continuous trajectory, which is the premise the single trajectory buffer
    rests on;
  * a tool change, a tool-offset change, a rigid tap and a suppressed region
    are recorded (or not) exactly as before.

Needs the built ``gcode`` extension.
"""
import os
import sys
import unittest

import numpy as np

import rs274.glcanon_bake as bake

sys.path.insert(0, os.path.dirname(__file__))

from headless_canon import COLORS, HeadlessCanon, REPO, parse   # noqa: E402,F401
from streams import write                                       # noqa: E402


def parsed(text, geometry="XYZ", **kw):
    """Parse a generated program; the file lives no longer than the parse."""
    path = write(text)
    try:
        return parse(path, geometry, **kw)
    finally:
        os.unlink(path)


#: Leading traverse, then one cut. The traverse moves the tool without
#: drawing, so every program below starts its array with one record vertex.
PREAMBLE = "G20 G17 G90\nG0 X0 Y0 Z0\nG1 F10 X1 Y0\n"


PROGRAM = os.path.join(os.path.dirname(__file__), "fixtures", "order_mixed.ngc")


class CanonOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canon = parse(PROGRAM)

    def test_highlight_unaffected(self):
        """highlight() still finds a line the program actually drew."""
        c = self.canon
        geometry = c.program_geometry
        drawn = geometry.kinds <= bake.LAST_DRAWN_KIND
        lineno = int(geometry.lines[drawn][len(geometry.lines[drawn]) // 2])
        x, y, z = c.highlight(lineno, "XYZ")
        for v in (x, y, z):
            self.assertIsInstance(float(v), float)

    def test_the_whole_program_is_one_unbroken_strip(self):
        """No tool change in the fixture, so nothing jumps after the start.

        This is the load-bearing claim: category changes must not fragment
        it. Drawn per category, the same moves would produce a strip per run.
        The one no-op is the record at the very first point, which is what
        starts the strip.
        """
        c = self.canon
        geometry = c.program_geometry
        kinds = geometry.kinds
        noops = np.flatnonzero(kinds == bake.KIND_NOOP)
        self.assertEqual(list(noops), [0])
        # k points for k-1 segments, sharing every interior vertex, plus the
        # one dwell record the fixture ends with.
        dwells = int((kinds == bake.KIND_DWELL).sum())
        self.assertEqual(len(geometry) - dwells, geometry.n_moves + 1)


class ChainBreaks(unittest.TestCase):
    """Tool changes and tool-offset changes: the only real trajectory breaks.

    ``M6`` with no ``T`` word still delivers a tool change, and ``G43.1`` a
    tool offset, so both are reachable from a real parse - which is the only
    way to reach them at all now that the canon does not draw.
    """

    def kinds(self, canon):
        return [int(k) for k in canon.program_geometry.kinds]

    def test_change_tool_records_a_jump(self):
        c = parsed(PREAMBLE + "M6\nG0 X5 Y5\nG1 X6 Y5\nM2\n")
        self.assertEqual(c.program_geometry.n_moves, 2)
        self.assertEqual(self.kinds(c),
                         [bake.KIND_NOOP, bake.KIND_FEED,
                          bake.KIND_TOOLCHANGE, bake.KIND_NOOP,
                          bake.KIND_FEED])

    def test_tool_offset_records_a_jump(self):
        c = parsed(PREAMBLE + "G43.1 Z0.5\nG1 X2 Y0\nM2\n")
        # One at the program's start, one where the offset moved the tool.
        self.assertEqual(self.kinds(c).count(bake.KIND_NOOP), 2)

    def test_rigid_tap_records_both_moves(self):
        c = parsed(PREAMBLE + "S500 M3\nG33.1 Z-0.1 K0.05\nM2\n")
        # Down and back up the way it came, off the same chain point.
        self.assertEqual(c.program_geometry.n_moves, 3)
        self.assertEqual(self.kinds(c)[-2:], [bake.KIND_FEED, bake.KIND_FEED])

    def test_straight_probe_is_recorded(self):
        c = parsed(PREAMBLE + "G38.2 Z-0.2 F5\nM2\n")
        self.assertEqual(c.program_geometry.n_moves, 2)
        self.assertEqual(self.kinds(c)[-1], bake.KIND_FEED)

    def test_suppressed_moves_are_not_recorded(self):
        """(AXIS,hide) suppresses a move from the record entirely."""
        c = parsed(PREAMBLE + "(AXIS,hide)\nG1 X2 Y0\nG0 X3 Y0\n"
                   "(AXIS,show)\nM2\n")
        self.assertEqual(c.program_geometry.n_moves, 1)

    def test_a_hidden_move_does_not_move_the_chain_point(self):
        """The move after a hidden span starts where the last drawn one
        ended, so it draws a segment rather than a jump."""
        c = parsed(PREAMBLE + "(AXIS,hide)\nG1 X2 Y0\n(AXIS,show)\n"
                   "G1 X3 Y0\nM2\n")
        self.assertEqual(self.kinds(c).count(bake.KIND_NOOP), 1)


DWELL_PROGRAM = os.path.join(os.path.dirname(__file__), "fixtures",
                             "dwell_m1xx.ngc")


class DwellPalette(unittest.TestCase):
    """A real program with both dwell colours, parsed and baked.

    The bake's palette collection is unit-tested on synthetic items elsewhere;
    this checks the colours the canon actually appends for a ``G4`` and an
    ``M1xx``, which is where the two-entry claim comes from.
    """

    @classmethod
    def setUpClass(cls):
        cls.canon = parse(DWELL_PROGRAM)

    def test_canon_records_both_dwell_colours(self):
        c = self.canon
        self.assertEqual(len(c.dwells), 3)      # G4, M100, G4
        colours = [d[1] for d in c.dwells]
        self.assertEqual(colours[0], COLORS['dwell'])
        self.assertEqual(colours[1], COLORS['m1xx'])
        self.assertEqual(colours[2], COLORS['dwell'])

    def test_the_part_collects_two_entries_and_indexes_each_marker(self):
        part = bake.dwell_marker_part(self.canon.program_geometry)
        self.assertEqual(part["kind"], "program_array")
        # Exactly two distinct colours, in first-seen order.
        palette = part["palettes"][0]
        self.assertEqual(palette[0], tuple(COLORS['dwell']) + (1.0,))
        self.assertEqual(palette[1], tuple(COLORS['m1xx']) + (1.0,))
        self.assertEqual(set(palette[2:]), {(0.0, 0.0, 0.0, 1.0)},
                         "extra entries assigned")
        # Four vertices per marker: dwell, m1xx, then dwell reusing entry 0.
        kinds = part["attrs"]["kindtool"] & bake.KIND_MASK
        np.testing.assert_array_equal(kinds, [0] * 4 + [1] * 4 + [0] * 4)

    def test_dwells_stay_pickable_by_source_line(self):
        """Each marker's own line number survives into its own uint32 field."""
        part = bake.dwell_marker_part(self.canon.program_geometry)
        expected = [d[0] for d in self.canon.dwells]
        np.testing.assert_array_equal(part["attrs"]["line"][::4], expected)
        keys, _firsts, _counts = part["spans"]
        for lineno in expected:
            self.assertIn(lineno, list(keys))


if __name__ == "__main__":
    unittest.main()
