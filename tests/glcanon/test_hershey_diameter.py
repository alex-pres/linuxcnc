#!/usr/bin/env python3
"""The Hershey diameter glyph the lathe extents labels are marked with.

``string_polylines`` looks a character up as ``self.hershey[translate[c]]``
with no fallback, so an unmapped character is a KeyError inside a draw call -
a dead frame, not a missing mark. These tests pin the two things that would
make that happen again: the glyph exists for both diameter codepoints, and it
is laid out with the ordinary full advance so nothing around it reflows.

GL-free; runs anywhere:
    python3 tests/glcanon/test_hershey_diameter.py
"""
import importlib.util
import math
import os
import sys
import unittest

_spec = importlib.util.spec_from_file_location(
    "hershey", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "lib", "python", "hershey.py"))
hershey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hershey)

DIA = "Ø"          # what the renderer emits
DIA_ALIAS = "⌀"    # DIAMETER SIGN, accepted but never emitted


class TestDiameterGlyph(unittest.TestCase):
    def setUp(self):
        self.h = hershey.Hershey()

    def test_both_codepoints_draw_the_same_two_strokes(self):
        plain = self.h.string_polylines("1.25")
        for mark in (DIA, DIA_ALIAS):
            marked = self.h.string_polylines(mark + "1.25")
            # circle + slash, on top of the four the digits already draw
            self.assertEqual(len(marked), len(plain) + 2, mark)
        self.assertEqual(self.h.string_polylines(DIA),
                         self.h.string_polylines(DIA_ALIAS))

    def test_mark_takes_the_default_advance(self):
        # 400/440 - the same advance as a digit, which is what lets the label
        # keep its layout, frac centring and bbox with the mark in front.
        self.assertAlmostEqual(self.h.string_len(DIA + "1.25")
                               - self.h.string_len("1.25"), 400.0/440.0)

    def test_circle_closes_and_the_slash_crosses_it(self):
        # Asserted on the raw 440-space strokes: the slash overshoots the ring
        # along its own diagonal, which a bounding box would not show.
        circle, slash = self.h.hershey[hershey.translate[DIA]]
        for a, b in zip(circle[0], circle[-1]):
            self.assertAlmostEqual(a, b)   # closes on itself
        self.assertEqual(len(slash), 2)
        cx = sum(p[0] for p in circle[:-1]) / (len(circle) - 1)
        cy = sum(p[1] for p in circle[:-1]) / (len(circle) - 1)
        r = math.hypot(circle[0][0] - cx, circle[0][1] - cy)
        for end in slash:
            self.assertGreater(math.hypot(end[0] - cx, end[1] - cy), r)

    def test_mark_sits_within_digit_height(self):
        digit = self.h.string_polylines("0")
        dys = [p[1] for stroke in digit for p in stroke]
        mys = [p[1] for stroke in self.h.string_polylines(DIA) for p in stroke]
        self.assertGreaterEqual(min(mys), min(dys))
        self.assertLessEqual(max(mys), max(dys))


if __name__ == "__main__":
    unittest.main(verbosity=2)
