#!/usr/bin/env python3
"""``GLCanon.highlight()``'s array-mask rewrite, recorded values.

When this test was written, ``highlight()`` was cross-checked live against
the old per-list loops (both-endpoints-per-segment weighting) for every line
in every fixture - see the group-2 task in ``retire-canon-move-lists``'
tasks.md, and this file's git history for that differential form. All lines
in all fixtures agreed to 5 decimal places (float32 position storage).
``retire-canon-move-lists`` then removed the lists that cross-check read, so -
per the same "replace the oracle before the lists go" ordering used for
``test_extents_oracle.py`` - this pins the values the equality already
proved.

Needs the built ``gcode`` extension.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from headless_canon import parse                            # noqa: E402

FIXTURES = [
    ("order_mixed.ngc", "XYZ", {}),
    ("dwell_m1xx.ngc", "XYZ", {}),
    ("rotary_abc.ngc", "XYZ", {}),
    ("rotated_xy.ngc", "XYZ", {}),
    ("hide_jump.ngc", "XYZ", {}),
    ("lathe_xz.ngc", "XZ", {}),
    ("alternating_dwells.ngc", "XYZ", {}),
    ("foam_xyuv.ngc", "XY;UV", {"is_foam": 1}),
]

#: Captured from ``canon.highlight(lineno, geometry)`` while it was still
#: cross-checked live against the old per-list, both-endpoints weighting (see
#: module docstring) - not read off code this test no longer exercises live.
RECORDED = {
    'order_mixed.ngc': {
        8: (0.0, 0.0, 0.25), 9: (0.5, 0.0, 0.0),
        10: (0.787836, 1.212164, 0.0), 11: (2.0, 1.5, 0.0),
        12: (1.0, 2.0, 0.0), 13: (0.0, 1.0, 0.0),
        14: (0.363508, -0.636492, 0.0), 15: (1.5, -1.0, 0.0),
        16: (2.0, -1.0, 0.25), 17: (2.0, -1.0, 0.5),
    },
    'dwell_m1xx.ngc': {
        11: (0.0, 0.0, 0.25), 12: (0.5, 0.0, 0.0), 13: (1.0, 0.0, 0.0),
        14: (1.5, 0.0, 0.0), 15: (2.0, 0.0, 0.0), 16: (2.5, 0.0, 0.0),
        17: (3.0, 0.0, 0.0),
    },
    'rotary_abc.ngc': {
        9: (0.5, 0.0, 0.0), 10: (1.0, 0.0, 0.0), 11: (1.5, 0.0, 0.0),
        12: (2.0, 0.0, 0.0), 13: (2.0, 0.5, 0.0), 14: (2.0, 1.0, 0.0),
        15: (1.0, 0.5, 0.0), 16: (0.0, 0.0, 0.0), 17: (0.0, 0.0, 0.25),
    },
    'rotated_xy.ngc': {
        11: (0.683013, 1.0, 0.0), 12: (0.866025, 1.683013, 0.0),
        13: (0.183013, 1.866025, 0.0), 14: (0.0, 1.183013, 0.0),
        15: (0.25, 0.75, 0.25),
    },
    'hide_jump.ngc': {
        14: (0.5, 0.0, 0.0), 15: (1.0, 0.5, 0.0), 22: (1.5, 1.0, 0.0),
        24: (2.0, 1.5, -0.5), 25: (1.0, 2.0, -0.5), 27: (0.0, 1.0, 0.0),
        28: (0.0, 0.0, 0.25),
    },
    'lathe_xz.ngc': {
        8: (0.6, 0.0, 0.05), 9: (0.55, 0.0, -0.1), 10: (0.5, 0.0, -0.2),
        11: (0.5, 0.0, -0.5), 12: (0.45, 0.0, -0.9), 13: (0.4, 0.0, -1.0),
        14: (0.336351, 0.0, -1.036351), 15: (0.3, 0.0, -1.3),
        16: (0.3, 0.0, -1.5), 17: (0.45, 0.0, -1.5), 18: (0.6, 0.0, -0.7),
    },
    'alternating_dwells.ngc': {
        10: (0.5, 0.0, 0.0), 11: (1.0, 0.0, 0.0), 12: (1.5, 0.0, 0.0),
        13: (2.0, 0.0, 0.0), 14: (2.0, 0.5, 0.0), 15: (2.0, 1.0, 0.0),
        16: (2.0, 1.5, 0.0), 17: (2.0, 2.0, 0.0), 18: (1.5, 2.0, 0.0),
        19: (1.0, 2.0, 0.0), 20: (0.5, 2.0, 0.0), 21: (0.0, 2.0, 0.25),
    },
    'foam_xyuv.ngc': {
        10: (0.5, 0.0, 0.0), 11: (1.0, 0.0, 0.0), 12: (1.0, 0.5, 0.0),
        13: (1.0, 1.0, 0.0), 14: (0.5, 1.0, 0.0), 15: (0.0, 0.5, 0.0),
        16: (0.0, 0.0, 0.0),
    },
}

#: rotary_abc.ngc line 15 (``G1 X0 Y0 A30 B30 C30``) subdivides on all three
#: rotary axes at once - up to 36 interpolated points on one source line.
MANY_SEGMENTS_FIXTURE = ("rotary_abc.ngc", "XYZ", 15)

#: alternating_dwells.ngc line 11 is a bare ``G4``, between two moves on
#: different lines - no drawn geometry of its own.
DWELL_ONLY_FIXTURE = ("alternating_dwells.ngc", "XYZ", 11)


class HighlightMatchesRecordedValues(unittest.TestCase):

    def test_every_line_in_every_fixture(self):
        for fixture, geometry, kw in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry, **kw)
                for lineno, want in RECORDED[fixture].items():
                    with self.subTest(lineno=lineno):
                        got = canon.highlight(lineno, geometry)
                        for g, w in zip(got, want):
                            self.assertAlmostEqual(g, w, 5)

    def test_a_line_with_many_arc_segments(self):
        fixture, geometry, lineno = MANY_SEGMENTS_FIXTURE
        canon = parse(fixture, geometry)
        got = canon.highlight(lineno, geometry)
        for g, w in zip(got, RECORDED[fixture][lineno]):
            self.assertAlmostEqual(g, w, 5)

    def test_a_line_carrying_only_a_dwell(self):
        fixture, geometry, lineno = DWELL_ONLY_FIXTURE
        canon = parse(fixture, geometry)
        got = canon.highlight(lineno, geometry)
        for g, w in zip(got, RECORDED[fixture][lineno]):
            self.assertAlmostEqual(g, w, 5)

    def test_an_unknown_line_falls_back_to_extents_centre(self):
        canon = parse("order_mixed.ngc")
        canon.calc_extents()
        x, y, z = canon.highlight(999999, "XYZ")
        self.assertAlmostEqual(x, (canon.min_extents[0] + canon.max_extents[0]) / 2)
        self.assertAlmostEqual(y, (canon.min_extents[1] + canon.max_extents[1]) / 2)
        self.assertAlmostEqual(z, (canon.min_extents[2] + canon.max_extents[2]) / 2)


if __name__ == "__main__":
    unittest.main()
