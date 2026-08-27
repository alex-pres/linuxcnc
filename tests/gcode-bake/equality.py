#!/usr/bin/env python3
"""What a parsed canon is, reduced to comparable numbers - and how to compare.

:func:`snapshot` reads everything a preview parse produces that anything
downstream can see: the vertex arrays, the four machine-frame extent pairs,
the path lengths, the dwell and tool-change tables, and the canon state a
reader looks at once the parse is over. Two snapshots compare with
:meth:`EqualityMixin.assertSnapshotsEqual`, and a snapshot saves to and loads
from a JSON text file, which is what a baked expectation is.

The tolerances live here, in one place, with their causes:

* **coordinates** - a few ULPs, not exact. The renderer's arithmetic is
  compiled, and a compiler may contract ``x*cos - y*sin`` into one rounding
  where the source says two; the same expression contracts differently on
  different architectures, so a number baked on one machine is compared on
  another. Sized for one float32 vertex ULP (~1.2e-7), which is far above
  what any fixture actually shows and far below a dropped offset, a missed
  rotation or a mis-chained move.
* **accumulated lengths** - summed a move at a time, so a running total drifts
  with move count: ~20000 ULPs over 200k moves against the exact answer, 4e-12
  relative, nanometres on a metre of tool path. A move whose length was
  dropped or mis-scaled is millions of ULPs out and still fails.

GL-free; needs numpy only.
"""
import json
import sys

import numpy as np


def _floats(values):
    """A JSON-shaped list, so a fresh snapshot and a loaded one are the same
    object graph and compare with ``assertEqual`` rather than by coincidence."""
    return [float(v) for v in values]


def snapshot(canon):
    """Everything about a parsed canon that a reader downstream can see."""
    geometry = canon.program_geometry
    canon.calc_extents()
    planes = [np.array(geometry.positions(i), dtype=np.float32, copy=True)
              for i in range(len(geometry.planes))]
    return {
        "planes": planes,
        "lines": np.array(geometry.lines, dtype=np.uint32, copy=True),
        "kinds": np.array(geometry.kinds, dtype=np.uint32, copy=True),
        "tools": np.array(geometry.tools, dtype=np.uint32, copy=True),
        "extents": np.array([geometry.extents, geometry.extents_notool,
                             geometry.extents_zero_rxy,
                             geometry.extents_notool_zero_rxy],
                            dtype=np.float64),
        "meta": {
            "n_vertices": int(len(geometry)),
            "n_moves": int(geometry.n_moves),
            "rapid_length": float(geometry.rapid_length),
            "cutting_length": float(geometry.cutting_length),
            "cutting_time_100": float(geometry.cutting_time(100.)),
            "tool_numbers": list(geometry.tool_numbers),
            "toolchanges": [[int(lineno),
                             None if tool is None else int(tool),
                             [_floats(p) for p in points]]
                            for lineno, tool, points in geometry.toolchanges],
            "record_dwells": [[int(lineno), _floats(rgba), int(plane),
                               [_floats(p) for p in points]]
                              for lineno, rgba, plane, points
                              in geometry.dwells],
            "canon_dwells": [[int(row[0]), _floats(row[1]),
                              _floats(row[2:5]), int(row[5])]
                             for row in canon.dwells],
            "dwell_time": float(canon.dwell_time),
            "tool_list": [int(t) for t in canon.tool_list],
            "lo": _floats(canon.lo),
            "first_move": bool(canon.first_move),
            "tool_offset": _floats((canon.xo, canon.yo, canon.zo)),
            "min_extents": _floats(canon.min_extents),
            "max_extents": _floats(canon.max_extents),
            "g0_length": float(canon.g0_length),
            "g1_length": float(canon.g1_length),
        },
    }


def _f32(a):
    """Shortest text for each float32 that reads back as the same float32."""
    return a.astype(str)


def _rows(text_rows, indent=""):
    """One row per line, so a re-bake diffs a line at a time."""
    rows = ["%s[%s]" % (indent, ",".join(r)) for r in text_rows]
    return "[\n%s\n%s]" % (",\n".join(rows), indent) if rows else "[]"


def _flat(values, per_line=16, indent=""):
    """A flat array wrapped at ``per_line``, for the same reason."""
    values = [str(v) for v in values]
    if not values:
        return "[]"
    lines = [indent + ",".join(values[i:i + per_line])
             for i in range(0, len(values), per_line)]
    return "[\n%s\n%s]" % (",\n".join(lines), indent)


def _meta(meta):
    """One key per line; a table of rows gets one row per line, and every
    other value stays on its key's line."""
    lines = []
    for key in sorted(meta):
        value = meta[key]
        if isinstance(value, list) and value and isinstance(value[0], list):
            rows = ",\n".join("    " + json.dumps(row) for row in value)
            text = "[\n%s\n  ]" % rows
        else:
            text = json.dumps(value)
        lines.append('  "%s": %s' % (key, text))
    return "{\n%s\n}" % ",\n".join(lines)


def dumps(snap):
    """A snapshot as JSON text: readable, diffable, and exact.

    Hand-emitted rather than ``json.dumps``-ed so the layout carries meaning -
    one vertex per line, integer arrays wrapped - which is what makes a
    re-bake reviewable. Every float is written at its shortest round-tripping
    length, so what is read back is bit-for-bit what was saved.
    """
    meta = _meta(snap["meta"])
    extents = _rows(np.asarray(snap["extents"], dtype=np.float64)
                    .reshape(-1, 3).astype(str), "  ")
    planes = ",\n".join("  " + _rows(_f32(p.reshape(-1, 3)), "   ")
                        for p in snap["planes"])
    return ("{\n"
            '"meta": %s,\n'
            '"extents": %s,\n'
            '"lines": %s,\n'
            '"kinds": %s,\n'
            '"tools": %s,\n'
            '"planes": [\n%s\n]\n'
            "}\n") % (meta, extents,
                      _flat(snap["lines"], indent="  "),
                      _flat(snap["kinds"], indent="  "),
                      _flat(snap["tools"], indent="  "),
                      planes)


def save(path, snap):
    """Write a snapshot as JSON text."""
    with open(path, "w") as f:
        f.write(dumps(snap))


def load(path):
    """Read back what :func:`save` wrote."""
    with open(path) as f:
        data = json.load(f)
    return {
        "planes": [np.array(p, dtype=np.float32).reshape(-1, 3)
                   for p in data["planes"]],
        "lines": np.array(data["lines"], dtype=np.uint32),
        "kinds": np.array(data["kinds"], dtype=np.uint32),
        "tools": np.array(data["tools"], dtype=np.uint32),
        "extents": np.array(data["extents"],
                            dtype=np.float64).reshape(-1, 2, 3),
        "meta": data["meta"],
    }


#: ``meta`` keys holding a coordinate, compared with the point tolerance.
_POINT_KEYS = ("lo", "tool_offset", "min_extents", "max_extents")
#: ``meta`` keys holding an accumulated sum, compared in ULPs.
_SUM_KEYS = ("rapid_length", "cutting_length", "cutting_time_100",
             "g0_length", "g1_length")


class EqualityMixin:
    """Compares two snapshots, or two canons, to the tolerances above."""

    #: Allowance on every coordinate compared - vertex positions, the extents,
    #: the dwell positions, the final chain point. See the module docstring
    #: for why it is not zero.
    POINT_RTOL = 1e-6
    POINT_ATOL = 1e-9

    #: How far apart an accumulated length may land, in ULPs.
    SUM_ULPS = 100000.

    def assertPointsEqual(self, want, got, name):
        want = np.asarray(want)
        got = np.asarray(got)
        if not self.POINT_RTOL and not self.POINT_ATOL:
            np.testing.assert_array_equal(want, got, name)
        else:
            np.testing.assert_allclose(got, want, rtol=self.POINT_RTOL,
                                       atol=self.POINT_ATOL, err_msg=name)

    def assertSameSum(self, want, got, name):
        """Equal to the last few ULPs, which is as equal as a sum gets here."""
        if want == got:
            return
        biggest = max(abs(want), abs(got))
        if not biggest:
            self.fail("%s: %r != %r" % (name, want, got))
        ulps = abs(want - got) / biggest / sys.float_info.epsilon
        self.assertLess(ulps, self.SUM_ULPS,
                        "%s: %r != %r (%.1f ulps apart - too far to be "
                        "summation order)" % (name, want, got, ulps))

    def assertSnapshotsEqual(self, want, got):
        self.assertEqual(len(want["planes"]), len(got["planes"]),
                         "drawn plane count")
        self.assertEqual(want["meta"]["n_vertices"], got["meta"]["n_vertices"],
                         "vertex count")
        for i, (a, b) in enumerate(zip(want["planes"], got["planes"])):
            self.assertPointsEqual(a, b, "positions on plane %d" % i)
        for name in ("lines", "kinds", "tools"):
            np.testing.assert_array_equal(want[name], got[name], name)
        self.assertPointsEqual(want["extents"], got["extents"],
                               "the four extent pairs")

        a, b = want["meta"], got["meta"]
        self.assertEqual(a["n_moves"], b["n_moves"], "move count")
        for key in _SUM_KEYS:
            self.assertSameSum(a[key], b[key], key)
        for key in _POINT_KEYS:
            self.assertPointsEqual(a[key], b[key], key)
        for key in ("tool_numbers", "tool_list", "first_move", "dwell_time"):
            self.assertEqual(a[key], b[key], key)

        self.assertEqual(len(a["toolchanges"]), len(b["toolchanges"]),
                         "tool change count")
        for i, (x, y) in enumerate(zip(a["toolchanges"], b["toolchanges"])):
            self.assertEqual(x[:2], y[:2], "tool change %d line/number" % i)
            self.assertPointsEqual(x[2], y[2], "tool change %d position" % i)

        self.assertEqual(len(a["record_dwells"]), len(b["record_dwells"]),
                         "dwell marker count")
        for i, (x, y) in enumerate(zip(a["record_dwells"],
                                       b["record_dwells"])):
            self.assertEqual(x[:3], y[:3],
                             "dwell marker %d line/colour/plane" % i)
            self.assertPointsEqual(x[3], y[3], "dwell marker %d position" % i)

        self.assertEqual(len(a["canon_dwells"]), len(b["canon_dwells"]),
                         "canon.dwells")
        for i, (x, y) in enumerate(zip(a["canon_dwells"], b["canon_dwells"])):
            self.assertEqual((x[0], x[1], x[3]), (y[0], y[1], y[3]),
                             "canon.dwells[%d] line/colour/plane" % i)
            self.assertPointsEqual(x[2], y[2], "canon.dwells[%d] position" % i)

    def assertCanonsEqual(self, want, got):
        self.assertSnapshotsEqual(snapshot(want), snapshot(got))
