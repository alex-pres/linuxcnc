#!/usr/bin/env python3
"""The extents oracle, retired: recorded values, not a live differential.

``program-array-source-of-truth`` moved the eight extent vectors off
``gcode.calc_extents`` (C, a second traversal of a second copy of the move
data) onto an accumulation performed while the vertex array is filled, and
kept the C function and ``GLCanon.unrotate_preview()`` callable purely as a
differential oracle for that accumulation - which is what this file asserted
against, live, until now.

``retire-canon-move-lists`` removed the lists both of those read
(``traverse``/``feed``/``arcfeed``), so the oracle stopped being callable. Per
the migration order in design.md ("the extents oracle must be replaced before
the lists go, not after"), the values below were captured **from the still-
working oracle** - ``gcode.calc_extents`` over the live lists, and
``unrotate_preview()`` - while it was still callable, via a one-off script
against this same fixture corpus, and are pasted here as frozen literals. They
are not read off ``GLCanon.calc_extents()`` (the implementation under test):
recording expected values from the implementation being tested is exactly the
failure mode this ordering avoids. That provenance is why this file outlived
the Python fill it was written against - the numbers came from somewhere else
entirely, and they hold the C renderer to the same answers.

Two divergences the oracle itself always had, preserved as recorded facts
rather than live comparisons:

* ``RotateMidfile`` - a program that changes its XY rotation after motion has
  begun. ``unrotate_preview`` un-rotated the whole program by the *final*
  rotation; the renderer uses each move's own. ``PER_MOVE_ZERO_RXY`` is the renderer's
  (still asserted live); ``ORACLE_ZERO_RXY`` is what the retired oracle used
  to say, kept only as a documented contrast.
* The rotation-removed pairs are recorded to 12 decimal places rather than
  exactly, for the same reason in miniature: ``M2`` re-emits the g5x offset,
  and the value that came back could differ from the one the moves were laid
  down under in the last bit or two.

Needs the built ``gcode`` extension.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from headless_canon import parse                        # noqa: E402

#: (fixture, geometry). Every one is a program whose XY rotation and g5x
#: origin are constant; ``rotate_midfile.ngc`` is deliberately absent and has
#: its own case, below.
FIXTURES = [
    ("order_mixed.ngc", "XYZ"),
    ("dwell_m1xx.ngc", "XYZ"),
    ("rotary_abc.ngc", "XYZ"),
    ("rotated_xy.ngc", "XYZ"),
    ("hide_jump.ngc", "XYZ"),
    ("lathe_xz.ngc", "XZ"),
    ("foam_xyuv.ngc", "XY;UV"),
]

#: The eight vectors ``calc_extents`` populates, as (min, max) pairs, named in
#: the order the properties dialog reads them.
PAIRS = [
    ("min_extents", "max_extents"),
    ("min_extents_notool", "max_extents_notool"),
    ("min_extents_zero_rxy", "max_extents_zero_rxy"),
    ("min_extents_notool_zero_rxy", "max_extents_notool_zero_rxy"),
]

#: Captured from ``gcode.calc_extents(canon.arcfeed, canon.feed,
#: canon.traverse)`` (plain/notool pairs) and ``gcode.calc_extents(fresh.
#: unrotate_preview() and .preview_zero_rxy)`` (zero_rxy pairs, rounded to 12
#: places) - the still-working oracle, on the ``FIXTURES`` corpus above. See
#: the module docstring: not read off the code under test.
RECORDED = {
    'order_mixed.ngc': {
        'min_extents': [-2.220446049250313e-16, -1.0, 0.0],
        'max_extents': [2.0, 2.000000000000001, 0.5],
        'min_extents_notool': [-2.220446049250313e-16, -1.0, 0.0],
        'max_extents_notool': [2.0, 2.000000000000001, 0.5],
        'min_extents_zero_rxy': [-0.0, -1.0, 0.0],
        'max_extents_zero_rxy': [2.0, 2.0, 0.5],
        'min_extents_notool_zero_rxy': [-0.0, -1.0, 0.0],
        'max_extents_notool_zero_rxy': [2.0, 2.0, 0.5],
    },
    'dwell_m1xx.ngc': {
        'min_extents': [0.0, 0.0, 0.0],
        'max_extents': [3.0, 0.0, 0.5],
        'min_extents_notool': [0.0, 0.0, 0.0],
        'max_extents_notool': [3.0, 0.0, 0.5],
        'min_extents_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_zero_rxy': [3.0, 0.0, 0.5],
        'min_extents_notool_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_notool_zero_rxy': [3.0, 0.0, 0.5],
    },
    'rotary_abc.ngc': {
        'min_extents': [0.0, 0.0, 0.0],
        'max_extents': [2.0, 1.0, 0.5],
        'min_extents_notool': [0.0, 0.0, 0.0],
        'max_extents_notool': [2.0, 1.0, 0.5],
        'min_extents_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_zero_rxy': [2.0, 1.0, 0.5],
        'min_extents_notool_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_notool_zero_rxy': [2.0, 1.0, 0.5],
    },
    'rotated_xy.ngc': {
        'min_extents': [-0.24999999999999994, 0.75, 0.0],
        'max_extents': [1.1160254037844388, 2.1160254037844384, 0.5],
        'min_extents_notool': [-0.24999999999999994, 0.75, 0.0],
        'max_extents_notool': [1.1160254037844388, 2.1160254037844384, 0.5],
        'min_extents_zero_rxy': [0.25, 0.75, 0.0],
        'max_extents_zero_rxy': [1.25, 1.75, 0.5],
        'min_extents_notool_zero_rxy': [0.25, 0.75, 0.0],
        'max_extents_notool_zero_rxy': [1.25, 1.75, 0.5],
    },
    'hide_jump.ngc': {
        'min_extents': [0.0, 0.0, -0.5],
        'max_extents': [2.0, 2.0, 0.5],
        'min_extents_notool': [0.0, 0.0, 0.0],
        'max_extents_notool': [2.0, 2.0, 0.5],
        'min_extents_zero_rxy': [0.0, 0.0, -0.5],
        'max_extents_zero_rxy': [2.0, 2.0, 0.5],
        'min_extents_notool_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_notool_zero_rxy': [2.0, 2.0, 0.5],
    },
    'lathe_xz.ngc': {
        'min_extents': [0.3, 0.0, -1.5],
        'max_extents': [0.6, 0.0, 0.1],
        'min_extents_notool': [0.3, 0.0, -1.5],
        'max_extents_notool': [0.6, 0.0, 0.1],
        'min_extents_zero_rxy': [0.3, 0.0, -1.5],
        'max_extents_zero_rxy': [0.6, 0.0, 0.1],
        'min_extents_notool_zero_rxy': [0.3, 0.0, -1.5],
        'max_extents_notool_zero_rxy': [0.6, 0.0, 0.1],
    },
    'foam_xyuv.ngc': {
        'min_extents': [0.0, 0.0, 0.0],
        'max_extents': [1.0, 1.0, 0.0],
        'min_extents_notool': [0.0, 0.0, 0.0],
        'max_extents_notool': [1.0, 1.0, 0.0],
        'min_extents_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_zero_rxy': [1.0, 1.0, 0.0],
        'min_extents_notool_zero_rxy': [0.0, 0.0, 0.0],
        'max_extents_notool_zero_rxy': [1.0, 1.0, 0.0],
    },
}

#: ``foam_xyuv.ngc`` parsed *without* ``is_foam=1`` above (matching the
#: FIXTURES list, which never sets it) - ``calc_extents`` skips the Z-pair
#: override in that case, so the plain pair is asserted like any other.


class ExtentsMatchRecordedValues(unittest.TestCase):
    """``calc_extents()`` == the values the retired oracle used to produce."""

    def test_plain_and_notool_pairs(self):
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                canon.calc_extents()
                want = RECORDED[fixture]
                self.assertEqual(list(canon.min_extents), want['min_extents'])
                self.assertEqual(list(canon.max_extents), want['max_extents'])
                self.assertEqual(list(canon.min_extents_notool),
                                 want['min_extents_notool'])
                self.assertEqual(list(canon.max_extents_notool),
                                 want['max_extents_notool'])

    def test_zero_rxy_pairs(self):
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                canon.calc_extents()
                want = RECORDED[fixture]
                for attr in ('min_extents_zero_rxy', 'max_extents_zero_rxy',
                            'min_extents_notool_zero_rxy',
                            'max_extents_notool_zero_rxy'):
                    got = getattr(canon, attr)
                    for axis in range(3):
                        self.assertAlmostEqual(got[axis], want[attr][axis], 12)


class ExtentsIgnoreSubdividedRotaryPoints(unittest.TestCase):
    """The extents cover move endpoints, never the interpolated rotary points.

    The retired oracle was given move tuples and never saw a subdivided
    point, so this was true for free; asserted here against the recorded box
    because the renderer computes both from the same pass and could easily
    accumulate on the wrong side of the subdivision.
    """

    def test_endpoints_bound_the_box(self):
        canon = parse("rotary_abc.ngc")
        canon.calc_extents()
        want = RECORDED["rotary_abc.ngc"]
        self.assertEqual(list(canon.min_extents), want['min_extents'])
        self.assertEqual(list(canon.max_extents), want['max_extents'])

    def test_the_program_really_does_subdivide(self):
        """Without this the test above would pass on a fixture with no rotary
        motion at all, i.e. would assert nothing.

        No per-move traversal: a move contributes more than one vertex only
        when it subdivides, so more vertices than moves (beyond the one
        record vertex every program starts with) means some move did.
        """
        canon = parse("rotary_abc.ngc")
        geometry = canon.program_geometry
        self.assertGreater(len(geometry) - geometry.n_moves, 1,
                           "fixture has no rotary change")


class EmptyProgram(unittest.TestCase):
    """A program with no motion reports zeroes, not the 9e99 sentinels."""

    def test_all_eight_vectors_are_zero(self):
        canon = parse("blank_m2.ngc")
        canon.calc_extents()
        for pair in PAIRS:
            for name in pair:
                self.assertEqual(list(getattr(canon, name)), [0.0, 0.0, 0.0])

    def test_the_fixture_produces_no_moves(self):
        canon = parse("blank_m2.ngc")
        self.assertTrue(canon.program_geometry.is_empty)


class FoamZ(unittest.TestCase):
    """A foam program's Z pair is replaced by the two plane heights.

    Not a property of the move data at all - it is a rule living in
    ``calc_extents``' body, and one of the two that has to survive the move to
    a delegation.
    """

    #: Captured from ``gcode.calc_extents(canon.arcfeed, canon.feed,
    #: canon.traverse)`` with ``is_foam=1`` (a different canon construction
    #: than ``RECORDED['foam_xyuv.ngc']`` above, which does not set it).
    XY_PAIR = {"min": [0.0, 0.0], "max": [1.0, 1.0]}

    def setUp(self):
        self.canon = parse("foam_xyuv.ngc", "XY;UV",
                           is_foam=1, foam_z=0.25, foam_w=1.75)
        self.canon.calc_extents()

    def test_z_pair_is_the_plane_heights(self):
        self.assertEqual(self.canon.min_extents[2], 0.25)
        self.assertEqual(self.canon.max_extents[2], 1.75)
        self.assertEqual(self.canon.min_extents_notool[2], 0.25)
        self.assertEqual(self.canon.max_extents_notool[2], 1.75)

    def test_xy_pair_is_untouched_by_the_override(self):
        self.assertEqual(list(self.canon.min_extents[:2]), self.XY_PAIR["min"])
        self.assertEqual(list(self.canon.max_extents[:2]), self.XY_PAIR["max"])

    def test_the_zero_rxy_pairs_are_not_overridden(self):
        """Only the two non-rotated pairs get the foam Z; the rotation-removed
        pairs keep the Z the moves had. Asymmetric, and deliberate - it is what
        the code does today, so it is what the delegation must keep doing."""
        self.assertNotEqual(self.canon.min_extents_zero_rxy[2], 0.25)


class RotateMidfile(unittest.TestCase):
    """The one fixture where the renderer is expected to differ from the retired
    oracle.

    ``unrotate_preview`` read ``self.rotation_xy`` and ``self.g5x_offset_*``
    once, at the end of the parse, and applied that single rotation to every
    move in the program - including the moves emitted before a mid-file
    ``G10 L2 R`` changed it. A point laid down under R0 was therefore
    un-rotated by 40 degrees, and the resulting box was of a point set that
    never existed. Accumulating during the parse uses the rotation in effect
    for each move, which is the coherent answer.
    """

    #: The retired oracle's values, to 9 decimal places, captured before it
    #: was removed. Kept only as a documented contrast - regenerate
    #: deliberately, by hand, if the fixture changes, never by pasting what
    #: the renderer prints.
    ORACLE_ZERO_RXY = {
        "min": [0.0, -0.64278761, 0.0],
        "max": [2.0, 2.0, 0.5],
    }

    #: What the renderer's per-move accumulation gives instead: the first two
    #: moves were emitted under R0 and stay where they were laid down, the
    #: last three were emitted under R40 and un-rotate back onto the
    #: coordinates the program asked for. Every corner is then a point the
    #: machine visits.
    PER_MOVE_ZERO_RXY = {
        "min": [0.0, 0.0, 0.0],
        "max": [2.0, 2.0, 0.5],
    }

    def setUp(self):
        self.canon = parse("rotate_midfile.ngc")
        self.canon.calc_extents()

    def test_the_fixture_really_rotates_mid_program(self):
        """Otherwise this whole case is vacuous: motion must really have
        happened before the rotation changed, not just at the end.

        Vertex 0 is the record at the program's start; vertex 1 is the first
        real move's end point - (1, 0), laid down under R0, before the
        mid-file ``G10 L2 R40``.
        """
        self.assertEqual(self.canon.rotation_xy, 40)
        first = self.canon.program_geometry.positions()[1]
        self.assertAlmostEqual(float(first[0]), 1.0, 5)
        self.assertAlmostEqual(float(first[1]), 0.0, 5)

    def test_the_new_value_is_the_per_move_one(self):
        """The asserted answer is the renderer's, not the retired oracle's."""
        for key, attr in (("min", "min_extents_zero_rxy"),
                          ("max", "max_extents_zero_rxy")):
            for got, want in zip(getattr(self.canon, attr),
                                 self.PER_MOVE_ZERO_RXY[key]):
                self.assertAlmostEqual(got, want, 9)

    def test_the_two_answers_differ(self):
        """The divergence, stated as the two boxes rather than as prose."""
        self.assertNotEqual(self.ORACLE_ZERO_RXY, self.PER_MOVE_ZERO_RXY)


if __name__ == "__main__":
    unittest.main()
