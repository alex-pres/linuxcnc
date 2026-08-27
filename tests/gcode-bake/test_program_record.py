#!/usr/bin/env python3
"""The program record a real parse produces: order, markers, tools.

``gcode.parse`` builds the whole preview in C++ and hands it over at the end;
what this file pins is the part of that no baked array can state on its own -
that a dwell record lands *between* the moves it happened between, that a
marker sits where the path does on every drawn plane, that a tool change
advances the ordinal the vertices after it carry, and that a canon nothing
ever parsed into still answers every question about its (empty) program.

The interleaving is the fact only the parse knows: the dwell table has no
positional relationship to the vertices, and the vertices cannot say what
happened between two of them.

Needs the built ``gcode`` extension.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import line9_reference as ref                             # noqa: E402
import rs274.glcanon_bake as bake                         # noqa: E402
from headless_canon import COLORS, HeadlessCanon, parse    # noqa: E402
from streams import write                                  # noqa: E402


def parsed(text, geometry="XYZ", **kw):
    """Parse a generated program; the file lives no longer than the parse."""
    path = write(text)
    try:
        return parse(path, geometry, **kw)
    finally:
        os.unlink(path)


#: Leading traverse, then one cut, so every program below starts its array
#: with one record vertex and one drawn segment.
PREAMBLE = "G20 G17 G90\nG0 X0 Y0 Z0\nG1 F10 X1 Y0\n"


# -- the GEOMETRY transform, against the independent reference ---------------
#
# ``line9_reference`` is a scalar transcription of the C ``vertex9``/``line9``
# expansion, pinned against the shipping C extension itself in
# ``test_line9_bake_reference.py``. The renderer's own transform is a third
# implementation, and this is what anchors it: a program whose nine-DOF
# endpoints are written out in the G-code, parsed, and compared vertex for
# vertex against what the reference says those endpoints become.

#: The chain of nine-DOF points the program below visits. A/B/C are held
#: constant and non-zero, so a rotary GEOMETRY letter really turns the points
#: while no move subdivides; the subdivision case has a program of its own.
AXIS_LETTERS = "XYZABCUVW"
STILL_ABC = (30.0, 45.0, 60.0)
TRANSFORM_PATH = [
    (0.100, 0.200, 0.300) + STILL_ABC + (0.700, 0.800, 0.900),
    (1.100, -0.200, 0.350) + STILL_ABC + (0.100, -0.200, 0.300),
    (-1.500, 2.250, -0.375) + STILL_ABC + (0.250, 0.125, -0.500),
    (2.000, 1.000, -1.000) + STILL_ABC + (1.000, -2.000, 3.000),
]

#: A rotary move: C turns 90 degrees, so it is drawn as a polyline rather than
#: a straight line, and the reference decides how many points that is.
ROTARY_PATH = [
    (1.000, 0.000, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.000, 0.000, 0.000, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0),
    (1.500, 0.500, 0.000, 0.0, 0.0, 205.0, 0.0, 0.0, 0.0),
]

#: Rotation offsets in both forms. Every rotary letter is unmasked, so a
#: GEOMETRY string that names one really rotates.
RO = bake.RotationOffsets(respect_offsets=True, coords="XYZABC")
REF_RO = ref.RotationOffsets(respect_offsets=True, coords="ABC")


def _word(letter, value):
    return "%s%.4f" % (letter, value)


def path_program(points):
    """G-code visiting ``points``: a leading traverse, then one cut each."""
    out = ["G20 G17 G90"]
    for i, point in enumerate(points):
        words = " ".join(_word(letter, value)
                         for letter, value in zip(AXIS_LETTERS, point))
        out.append(("G0 " if i == 0 else "G1 F10 ") + words)
    out.append("M2")
    return "\n".join(out) + "\n"


def reference_vertices(points, geometry):
    """What the reference says the program's one strip is."""
    segs = [(1, list(points[i]), list(points[i + 1]))
            for i in range(len(points) - 1)]
    strips = ref.draw_lines(geometry, segs, REF_RO)
    assert len(strips) == 1, "the fixture must be one unbroken strip"
    return np.asarray(strips[0][1], dtype=np.float64)


class TransformAgainstTheReference(unittest.TestCase):
    """The renderer's GEOMETRY transform and rotary subdivision, anchored."""

    #: Every transform shape the preview draws. ``;`` and ``!`` sit in
    #: non-terminal positions on purpose: a ``-`` before a letter that is a
    #: no-op must leave the sign pending for the next letter, and a list that
    #: only ever put them last is what once hid exactly that bug.
    GEOMETRIES = ["XYZ", "XYZUVW", "XZ", "XY", "UV", "-XYZ", "X-YZ",
                  "!XYZ", "XY;UV", "XYZA", "XYZB", "XYZC", "XYZABC",
                  "XYZ-AB", "!CXYZ"]

    def compare(self, points, geometry):
        canon = parsed(path_program(points), geometry, ro=RO)
        want = reference_vertices(points, geometry)
        got = canon.program_geometry.positions()
        self.assertEqual(len(got), len(want), "vertex count")
        # One float32 ULP at these magnitudes; the reference works in double.
        np.testing.assert_allclose(got, want, rtol=0, atol=5e-6)

    def test_every_geometry_string(self):
        for geometry in self.GEOMETRIES:
            with self.subTest(geometry=geometry):
                self.compare(TRANSFORM_PATH, geometry)

    def test_a_rotary_move_subdivides_as_the_reference_does(self):
        for geometry in ("XYZ", "XYZC", "!CXYZ"):
            with self.subTest(geometry=geometry):
                self.compare(ROTARY_PATH, geometry)

    def test_the_rotary_fixture_really_subdivides(self):
        """Otherwise the comparison above proves only that nothing turns."""
        canon = parsed(path_program(ROTARY_PATH), "XYZC", ro=RO)
        geometry = canon.program_geometry
        self.assertGreater(len(geometry) - geometry.n_moves, 1)

    def test_the_rotary_letters_are_no_ops_without_the_mask(self):
        """A GEOMETRY string may name a rotary axis the config does not turn.

        Then the letter contributes nothing, and the program is drawn exactly
        as the same string without it.
        """
        plain = parsed(path_program(TRANSFORM_PATH), "XYZC",
                       ro=bake.RotationOffsets())
        without = parsed(path_program(TRANSFORM_PATH), "XYZ",
                         ro=bake.RotationOffsets())
        np.testing.assert_array_equal(plain.program_geometry.positions(),
                                      without.program_geometry.positions())


class EmissionOrder(unittest.TestCase):
    """Dwells land between the moves they happened between."""

    def setUp(self):
        self.canon = parse("alternating_dwells.ngc")
        self.geometry = self.canon.program_geometry

    def test_the_fixture_alternates(self):
        self.assertEqual(len(self.canon.dwells), 5)

    def test_one_record_vertex_per_dwell_in_source_order(self):
        kinds = self.geometry.kinds
        lines = self.geometry.lines
        at = np.flatnonzero(kinds == bake.KIND_DWELL)
        self.assertEqual(len(at), len(self.canon.dwells))
        self.assertEqual([int(v) for v in lines[at]],
                         [d[0] for d in self.canon.dwells])

    def test_each_dwell_sits_between_its_neighbouring_moves(self):
        """The vertex before a dwell record is the end of the move that
        preceded it, and the vertex after is the end of the one that
        followed - which is the whole claim, since the position does not
        change across any of the three."""
        kinds = self.geometry.kinds
        lines = self.geometry.lines
        positions = self.geometry.positions()
        for i in np.flatnonzero(kinds == bake.KIND_DWELL):
            self.assertGreater(i, 0)
            self.assertLess(i, len(kinds) - 1)
            np.testing.assert_allclose(positions[i], positions[i - 1])
            self.assertLess(int(lines[i - 1]), int(lines[i]))
            self.assertLess(int(lines[i]), int(lines[i + 1]))

    def test_the_dwell_table_agrees_with_the_records(self):
        """A pick on a marker and a lookup in the array report the same line."""
        at = np.flatnonzero(self.geometry.kinds == bake.KIND_DWELL)
        for i, (lineno, _rgba, _plane, points) in zip(
                at, self.geometry.dwells):
            self.assertEqual(int(self.geometry.lines[i]), lineno)
            np.testing.assert_allclose(self.geometry.positions()[i],
                                       points[0], atol=5e-6)

    def test_dwell_positions_are_transformed(self):
        """The defect the change fixes: the pre-change marker bake was handed
        the GEOMETRY string and the rotation offsets and applied neither.

        ``UV`` is used rather than a lathe's mapping because a lathe does not
        actually show this. A GEOMETRY letter selects which of the nine
        degrees of freedom feeds the preview axis of the same name, so ``XZ``
        merely drops Y - the identity on any program whose Y is zero, which a
        turning program's is. Reading a marker's position off the raw machine
        coordinates is wrong everywhere and visible only where the string
        maps one axis onto another, negates one, or rotates.
        """
        canon = parse("foam_xyuv.ngc", "UV")
        geometry = canon.program_geometry
        self.assertEqual(len(geometry.dwells), len(canon.dwells))
        moved = 0
        for raw, (_l, _c, _p, points) in zip(canon.dwells, geometry.dwells):
            machine = (raw[2], raw[3], raw[4])
            # The record vertex and the marker are the same point, and that
            # point is on the path - which is the whole fix.
            if tuple(points[0]) != machine:
                moved += 1
        self.assertTrue(moved, "UV is the identity on every dwell here")

    def test_the_marker_sits_where_the_path_does(self):
        canon = parse("foam_xyuv.ngc", "UV")
        geometry = canon.program_geometry
        at = np.flatnonzero(geometry.kinds == bake.KIND_DWELL)
        for i, (_l, _c, _p, points) in zip(at, geometry.dwells):
            np.testing.assert_allclose(geometry.positions()[i], points[0],
                                       atol=5e-6)
            np.testing.assert_allclose(geometry.positions()[i],
                                       geometry.positions()[i - 1], atol=5e-6)


class FoamDwells(unittest.TestCase):
    """One transformed position per drawn plane."""

    def setUp(self):
        self.canon = parse("foam_xyuv.ngc", "XY;UV", is_foam=1)
        self.geometry = self.canon.program_geometry

    def test_the_canon_configured_two_planes(self):
        self.assertEqual(self.geometry.planes, ("XY", "UV"))

    def test_two_positions_per_dwell(self):
        for _l, _c, _p, points in self.geometry.dwells:
            self.assertEqual(len(points), 2)

    def test_the_two_positions_are_the_two_planes(self):
        """Not every dwell: the program passes through a point where the XY
        and UV columns agree, and there the two markers coincide - correctly."""
        differ = sum(1 for _l, _c, _p, points in self.geometry.dwells
                     if points[0] != points[1])
        self.assertTrue(differ, "no dwell distinguishes the two planes")

    def test_each_plane_is_its_own_columns(self):
        for i, geom in enumerate(self.geometry.planes):
            for (_l, _c, _p, points), at in zip(
                    self.geometry.dwells,
                    np.flatnonzero(self.geometry.kinds == bake.KIND_DWELL)):
                np.testing.assert_allclose(
                    self.geometry.positions(i)[at], points[i], atol=5e-6)


class ToolColumn(unittest.TestCase):
    """Tool changes drive the ordinal, and the table resolves it.

    ``M6`` with no ``T`` word changes to tool 0, which is the only tool change
    a headless parse can make: the standalone ``gcode`` module has no tool
    table, and a ``T`` word walks off the end of one that is not there.
    """

    def test_before_any_change_the_ordinal_is_the_initial_state(self):
        g = parsed(PREAMBLE + "M2\n").program_geometry
        self.assertEqual(set(int(t) for t in g.tools), {0})
        self.assertIsNone(g.tool_numbers[0])

    def test_a_change_advances_the_ordinal_and_records_the_number(self):
        g = parsed(PREAMBLE + "M6\nG0 X5 Y5\nG1 X6 Y5\nM2\n").program_geometry
        self.assertEqual(int(g.tools[-1]), 1)
        self.assertEqual(g.tool_numbers, [None, 0])

    def test_the_ordinal_advances_once_per_change(self):
        g = parsed(PREAMBLE + "M6\nG1 X2 Y0\nM6\nG1 X3 Y0\nM2\n"
                   ).program_geometry
        self.assertEqual([int(t) for t in g.tools][-1], 2)
        self.assertEqual(len(g.tool_numbers), 3)

    def test_the_change_and_the_jump_are_two_vertices(self):
        """A tool change is followed by a rapid to the new start, which is the
        move ``first_move`` suppresses - so the change record and the jump
        record are two separate vertices, in that order."""
        g = parsed(PREAMBLE + "M6\nG0 X5 Y5\nG1 X6 Y5\nM2\n").program_geometry
        self.assertEqual([int(k) for k in g.kinds][-3:],
                         [bake.KIND_TOOLCHANGE, bake.KIND_NOOP,
                          bake.KIND_FEED])

    def test_a_feed_straight_after_a_change_is_not_a_jump(self):
        """A feed does not honour ``first_move``, so it draws from where the
        tool was. Pinned because it is the case that looks like it should
        break the strip and does not."""
        g = parsed(PREAMBLE + "M6\nG1 X2 Y0\nM2\n").program_geometry
        self.assertEqual([int(k) for k in g.kinds][-2:],
                         [bake.KIND_TOOLCHANGE, bake.KIND_FEED])

    def test_the_spare_bits_of_the_kind_tool_word_stay_zero(self):
        """An unspecified bit is a bit some later reader finds a use for and a
        still later one finds already used."""
        g = parsed(PREAMBLE + "M6\nG1 X2 Y0\nM2\n").program_geometry
        self.assertEqual(int((g.kindtool & bake.SPARE_MASK).max()), 0)


class RemovedAttributesRaise(unittest.TestCase):
    """The lists are gone; reading one names its replacement.

    Reason and replacements: see ``retire-canon-move-lists``. ``dwells`` and
    ``tool_list`` are unaffected - bounded by event count, not move count.
    """

    def test_dwell_rows_unchanged(self):
        canon = parse("order_mixed.ngc")
        for row in canon.dwells:
            self.assertEqual(len(row), 6)
            self.assertIn(row[1], (COLORS['dwell'], COLORS['m1xx']))

    def test_tool_list_unchanged(self):
        canon = parsed(PREAMBLE + "M6\nG1 X2 Y0\nM6\nG1 X3 Y0\nM2\n")
        self.assertEqual(canon.tool_list, [0, 0])

    def test_each_removed_attribute_raises_naming_a_replacement(self):
        canon = parse("order_mixed.ngc")
        for name in ("traverse", "feed", "arcfeed", "moves", "move_cats",
                    "preview_zero_rxy"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as ctx:
                    getattr(canon, name)
                self.assertIn(name, str(ctx.exception))

    def test_unrotate_preview_is_gone(self):
        canon = parse("order_mixed.ngc")
        self.assertFalse(hasattr(canon, "unrotate_preview"))


class EmptyCanon(unittest.TestCase):
    """A canon that never parsed still has a complete, readable record."""

    def setUp(self):
        self.canon = HeadlessCanon()

    def test_the_geometry_exists_and_is_empty(self):
        g = self.canon.program_geometry
        self.assertEqual(len(g), 0)
        self.assertTrue(g.is_empty)
        self.assertEqual(g.dwells, [])
        self.assertEqual(g.toolchanges, [])
        self.assertEqual(g.tool_numbers, [None])
        self.assertEqual(len(g.positions()), 0)
        self.assertEqual(len(g.attrs), 0)

    def test_calc_extents_reports_zeroes(self):
        self.canon.calc_extents()
        for name in ("min_extents", "max_extents", "min_extents_notool",
                     "max_extents_notool", "min_extents_zero_rxy",
                     "max_extents_zero_rxy", "min_extents_notool_zero_rxy",
                     "max_extents_notool_zero_rxy"):
            self.assertEqual(list(getattr(self.canon, name)), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
