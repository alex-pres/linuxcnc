#!/usr/bin/env python3
"""The accumulated path lengths and cutting time, recorded values.

When this test was written, ``canon.g0_length``/``g1_length``/``run_time()``
were cross-checked live against the summation the properties dialogs used to
run over ``canon.traverse``/``feed``/``arcfeed`` - see the group-1 task in
``retire-canon-move-lists``' tasks.md, and the git history of this file for
that differential form. All fixtures passed at 1e-9. ``retire-canon-move-
lists`` then removed the lists that cross-check read, so - per the same
"replace the oracle before the lists go" ordering used for
``test_extents_oracle.py`` - this pins the values the equality already proved,
rather than losing the coverage.

Needs the built ``gcode`` extension.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from headless_canon import parse                            # noqa: E402

FIXTURES = [
    ("order_mixed.ngc", "XYZ"),
    ("dwell_m1xx.ngc", "XYZ"),
    ("rotary_abc.ngc", "XYZ"),
    ("rotated_xy.ngc", "XYZ"),
    ("hide_jump.ngc", "XYZ"),
    ("lathe_xz.ngc", "XZ"),
    ("alternating_dwells.ngc", "XYZ"),
    ("foam_xyuv.ngc", "XY;UV"),
]

#: Below and above every rate any fixture here commands.
LOW, HIGH = 1.0, 1.0e6

#: Captured from ``canon.g0_length``/``g1_length``/``run_time(LOW/HIGH)``
#: while they were still cross-checked live against the list summation (see
#: module docstring) - not read off code this test no longer exercises live.
RECORDED = {
    'order_mixed.ngc': {'g0': 2.5, 'g1': 11.782554501865539,
                        'rt_low': 37.94766350559662, 'rt_high': 35.44766600559662},
    'dwell_m1xx.ngc': {'g0': 0.0, 'g1': 3.5,
                       'rt_low': 10.8, 'rt_high': 10.8},
    'rotary_abc.ngc': {'g0': 0.5, 'g1': 5.23606797749979,
                       'rt_low': 16.20820393249937, 'rt_high': 15.70820443249937},
    'rotated_xy.ngc': {'g0': 0.5, 'g1': 4.0,
                       'rt_low': 12.5, 'rt_high': 12.0000005},
    'hide_jump.ngc': {'g0': 0.5, 'g1': 8.0,
                      'rt_low': 24.5, 'rt_high': 24.0000005},
    'lathe_xz.ngc': {'g0': 1.6, 'g1': 2.0042774580465967,
                     'rt_low': 21.74277458046597, 'rt_high': 20.14277618046597},
    'alternating_dwells.ngc': {'g0': 0.5, 'g1': 6.0,
                              'rt_low': 18.53, 'rt_high': 18.0300005},
    'foam_xyuv.ngc': {'g0': 0.0, 'g1': 4.0,
                      'rt_low': 12.1, 'rt_high': 12.1},
}


class AccumulatedValuesMatchRecorded(unittest.TestCase):

    def test_g0_length(self):
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                self.assertAlmostEqual(canon.g0_length,
                                      RECORDED[fixture]['g0'], 9)

    def test_g1_length(self):
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                self.assertAlmostEqual(canon.g1_length,
                                      RECORDED[fixture]['g1'], 9)

    def test_run_time_below_and_above_every_commanded_rate(self):
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                self.assertAlmostEqual(canon.run_time(LOW),
                                      RECORDED[fixture]['rt_low'], 9)
                self.assertAlmostEqual(canon.run_time(HIGH),
                                      RECORDED[fixture]['rt_high'], 9)

    def test_run_time_is_monotonic_in_max_feed_rate(self):
        """A lower ceiling can only add time, never remove it."""
        for fixture, geometry in FIXTURES:
            with self.subTest(fixture=fixture):
                canon = parse(fixture, geometry)
                self.assertGreaterEqual(canon.run_time(LOW),
                                        canon.run_time(HIGH))


if __name__ == "__main__":
    unittest.main()
