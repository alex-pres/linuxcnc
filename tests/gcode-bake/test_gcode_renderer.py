#!/usr/bin/env python3
#!/usr/bin/env python3
"""The C renderer, against what it produced when it was last validated.

``GCodeRenderer`` (``src/emc/rs274ngc/gcode_renderer.{hh,cc}``) builds the whole
preview during ``gcode.parse`` - the g92/rotation/g5x transform, the chain
point, the arcs, the rigid-tap pair, the ``first_move`` drop, suppression, the
vertices, the extents, the lengths and the event records - and hands it over
once. It is the only way a preview gets built, so there is nothing in the tree
left to compare it against. What this file holds instead is:

  * a **baked corpus** - every fixture below, snapshotted whole into
    ``baked/``. The numbers were captured from this renderer after it had been
    validated move for move against the per-move Python canon it replaced;
    that validation is their provenance, and a re-bake is a deliberate act
    with a written cause, never a way to make a test pass;
  * **invariants** that hold whatever is baked, which is what catches the
    class of change a re-bake would otherwise wave through;
  * the **protocol**: the bool-only opt-in, a missing or raising consumer, a
    re-entered parse, the partial program a failed or stopped parse leaves,
    the progress cadence, and the feed-rate forwarding rules.

Cross-version truth - "has the preview drifted from the LinuxCNC we are
replacing" - is not an in-tree question and is not asked here: it is a
comparison between two builds, and belongs to whatever rig builds both.

The fixtures are generated inline so that what each one exercises is readable
next to the assertions: hidden spans that open and close mid-program, offsets
and rotations that change between moves, taps that do not advance the chain,
arcs in every plane, and a random stream that mixes all of it.

Needs the built ``gcode`` extension.
"""
import os
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import gcode                                              # noqa: E402
import rs274.glcanon_bake as bake                         # noqa: E402
from headless_canon import FIXTURES, HeadlessCanon        # noqa: E402
import equality                                           # noqa: E402
from equality import EqualityMixin                        # noqa: E402
from fake_preview import FakePreview                   # noqa: E402
from streams import (bench_feed, mixed, truncated_mixed,  # noqa: E402
                     write)


class RendererHeadlessCanon(HeadlessCanon):
    """The canon under test: the headless one, counting what it is handed."""

    def __init__(self, *args, **kw):
        HeadlessCanon.__init__(self, *args, **kw)
        self.progress_lines = []
        self.adopted = 0

    def renderer_progress(self, lineno):
        self.progress_lines.append(lineno)

    def adopt_geometry(self, pg):
        # Counted so a test can tell a rendered parse from one that never
        # rendered and would pass for the wrong reason.
        self.adopted += 1
        HeadlessCanon.adopt_geometry(self, pg)


def parse(cls, path, **kw):
    canon = cls("XYZ", **kw)
    with tempfile.NamedTemporaryFile(suffix=".var") as var:
        canon.parameter_file = var.name
        result = gcode.parse(path, canon, "", "")
    return canon, result


# -- fixtures ---------------------------------------------------------------

def hidden_spans():
    """``(AXIS,hide)`` opening and closing between moves, and nested.

    The depth is the renderer's own, counted off the comment text after the
    canon has had it; a word it fails to read draws moves that must not be
    drawn, and a depth it fails to nest closes a span one word too early.
    """
    return """(hidden_spans)
G20 G17 G90
G0 X0 Y0 Z0
G1 F10 X1 Y0
(AXIS,hide)
G1 X2 Y0
G1 X2 Y1
(AXIS,show)
G1 X3 Y1
(AXIS,hide)
(AXIS,hide)
G1 X4 Y1
(AXIS,show)
G1 X5 Y1
(AXIS,show)
G1 X6 Y1
(AXIS,hide)
G0 X7 Y7
G4 P0.1
M100 P1 Q2
G92 X0.5
(AXIS,show)
G1 X8 Y8
G92.1
G1 X9 Y9
M2
"""


def comment_vocabulary():
    """The comment words, in the spellings and shapes the parser must survive.

    ``(PREVIEW,hide)`` is the other spelling of the same word and must count
    the same depth. A word with nothing after it, a word that is a prefix of
    ``hide``, one that merely starts with it, and an ``(AXIS,...)`` word the
    renderer has no business reading must all leave the depth alone. The file
    ends inside an open span, which is legal and hides everything after it -
    including the moves between the last ``hide`` and ``M2``.
    """
    return """(comment_vocabulary)
G20 G17 G90
G0 X0 Y0 Z0
G1 F10 X1 Y0
(PREVIEW,hide)
G1 X2 Y0
(PREVIEW,show)
G1 X3 Y0
(AXIS,hide)
G1 X4 Y0
(PREVIEW,show)
G1 X5 Y0
(AXIS,)
(AXIS,hid)
(AXIS,hidden)
(AXIS,notify,still visible)
(a plain comment, not ours)
(AXISX,hide)
G1 X6 Y0
(AXIS,hide,with a trailing field)
G1 X7 Y0
(AXIS,show,and one here too)
G1 X8 Y0
(AXIS,hide)
G0 X9 Y9
G1 X10 Y10
M2
"""


def stopped_inside_hidden():
    """``(AXIS,stop)`` while a span is open.

    The forward raises ``KeyboardInterrupt`` out of the canon's ``comment``,
    so the parse ends on that line and the renderer never reads the word -
    which is exactly the order that keeps a ``(AXIS,stop)`` from being
    swallowed by a hide that came after it in the same comment.
    """
    return """(stopped_inside_hidden)
G20 G17 G90
G0 X0 Y0 Z0
G1 F10 X1 Y0
(AXIS,hide)
G1 X2 Y0
(AXIS,stop)
G1 X3 Y0
(AXIS,show)
G1 X4 Y0
M2
"""


def tool_changes():
    """Repeated changes, one of them inside a hidden span.

    A tool change is a record whether or not the moves around it are drawn -
    the legacy append was unconditional, and the renderer writes it outside
    the suppression gate for the same reason: the properties dialog's tool
    list is what the program *uses*, not what the preview happens to show.

    Every change here lands on T0: the standalone ``gcode`` module has no tool
    table, so a ``T<n> M6`` walks the interpreter off the end of one that is
    not there. What T numbers other than zero do to the list is covered
    against a hand-built handover in ToolList below.
    """
    return """(tool_changes)
G20 G17 G90
G0 X0 Y0 Z0
G1 F10 X1
M6
G1 X2
M6
(AXIS,hide)
M6
G1 X3
(AXIS,show)
G1 X4
M2
"""


def moving_transform():
    """g92, g5x and the XY rotation all changing between moves."""
    out = ["(moving_transform)", "G20 G17 G90", "G0 X0 Y0 Z0", "F20"]
    for i in range(40):
        out.append("G1 X%.3f Y%.3f" % (i * 0.1, (i % 5) * 0.2))
        if i % 7 == 3:
            out.append("G92 X%.3f Y%.3f" % (i * 0.01, -i * 0.01))
        if i % 11 == 5:
            out.append("G10 L2 P1 X%.3f Y%.3f R%d" % (i * 0.02, i * 0.03, i))
            out.append("G54")
        if i % 13 == 9:
            out.append("G92.1")
    out.append("M2")
    return "\n".join(out) + "\n"


def taps_and_traverses():
    """Rigid taps back to back, and leading traverses before every cut.

    A tap draws two segments and leaves the chain point where it was, so
    consecutive taps all hang off the same point; a traverse before the first
    cut moves the tool without drawing.
    """
    return """(taps_and_traverses)
G20 G17 G90
G0 X0 Y0 Z1
G0 X0.5 Y0.5
G0 X1 Y1
G1 F10 X1 Y1.5
S500 M3
G33.1 Z-0.1 K0.05
G33.1 Z-0.2 K0.05
G33.1 Z-0.3 K0.05
G1 X2 Y2
G43.1 Z0.25
G0 X3 Y3
G1 X3.5 Y3.5
G49
G1 X4 Y4
M2
"""


def arcs():
    """Arcs the renderer segments itself: every plane, helical, multi-turn.

    The C core is the one gcode.arc_to_segments has always used, but the
    renderer feeds it its own chain point and offsets rather than the canon's
    attributes, and consumes the segments without a Python call.
    """
    return """(arcs)
G20 G17 G90
G0 X0 Y0 Z0.5
G1 F20 X1 Y0
G2 X2 Y0 I0.5 J0
G3 X1 Y0 I-0.5 J0
G2 I0.5 J0 P3
G1 Z0.4
G2 X2 Y0 Z0.1 I0.5 J0
G18
G2 X3 Z0.5 I0.5 K0
G3 X2 Z0.1 I-0.5 K0
G19
G2 Y1 Z0.5 J0.5 K0
G17
G92 X0.3 Y0.4
G10 L2 P1 X0.2 Y0.1 R20
G54
G2 X3 Y1 I0.5 J0
(AXIS,hide)
G3 X2 Y1 I-0.5 J0
(AXIS,show)
G2 X3 Y1 I0.5 J0
G92.1
M2
"""


def feed_modes():
    """Inverse-time (G93) and units-per-revolution (G95) feed, then back.

    Neither mode changes what a move *is*, but both change the number in the
    F word by orders of magnitude, and that number is what the renderer files
    the move's length under. A program that switches modes therefore lands its
    cutting length in three different rows of the per-rate table, which is
    what the properties dialog's run time is summed from.
    """
    return """(feed_modes)
G20 G17 G90
G0 X0 Y0 Z0.1
G94 F10
G1 X1
G93 F2
G1 X2
G1 Y1
G95 F0.01
S600 M3
G1 X3
G94 F25
G1 X4 Y2
G4 P0.2
G1 X5
M2
"""


def random_stream(seed, lines=600):
    """A random mix of everything the renderer's state machine branches on."""
    rng = random.Random(seed)
    out = ["(random %d)" % seed, "G20 G17 G90", "G0 X0 Y0 Z0.5", "F15"]
    hidden = 0
    x = y = z = 0.0
    for _ in range(lines):
        pick = rng.random()
        x += rng.uniform(-0.4, 0.4)
        y += rng.uniform(-0.4, 0.4)
        z += rng.uniform(-0.05, 0.05)
        if pick < 0.45:
            out.append("G1 X%.4f Y%.4f Z%.4f" % (x, y, z))
        elif pick < 0.60:
            out.append("G0 X%.4f Y%.4f Z%.4f" % (x, y, z))
        elif pick < 0.66:
            out.append("F%d" % rng.randint(5, 60))
        elif pick < 0.71:
            out.append("G4 P%.2f" % rng.uniform(0.01, 0.4))
        elif pick < 0.75:
            out.append("M100 P%d Q%d" % (rng.randint(0, 3), rng.randint(0, 3)))
        elif pick < 0.79:
            out.append("G33.1 Z%.4f K0.05" % (z - 0.2))
        elif pick < 0.82:
            out.append("G38.2 Z%.4f F5" % (z - 0.1))
        elif pick < 0.85:
            out.append("G92 X%.3f Y%.3f" % (rng.uniform(-1, 1),
                                            rng.uniform(-1, 1)))
        elif pick < 0.87:
            out.append("G92.1")
        elif pick < 0.90:
            out.append("G10 L2 P1 X%.3f Y%.3f R%d"
                       % (rng.uniform(-1, 1), rng.uniform(-1, 1),
                          rng.randint(0, 90)))
            out.append("G54")
        elif pick < 0.92:
            out.append("G43.1 Z%.3f" % rng.uniform(0, 0.5))
        elif pick < 0.94:
            out.append("G49")
        elif pick < 0.96:
            out.append("M6")
        elif pick < 0.98 and not hidden:
            hidden = 1
            out.append("(AXIS,hide)")
        elif hidden:
            hidden = 0
            out.append("(AXIS,show)")
        else:
            out.append("G2 X%.4f Y%.4f I%.4f J%.4f"
                       % (x + 0.2, y, 0.1, 0.1))
    if hidden:
        out.append("(AXIS,show)")
    out.append("M2")
    return "\n".join(out) + "\n"


#: Checked-in fixture, GEOMETRY string, rotation offsets, canon kwargs. The
#: corpus the renderer is exercised over: every transform shape the preview draws -
#: mill, lathe, negated and reordered axes, a rotary letter that turns the
#: points, the foam pair of planes - against a program that actually uses it.
RO_ROTARY = bake.RotationOffsets(respect_offsets=True, coords="XYZABC")
RO_OFFSET = bake.RotationOffsets(respect_offsets=True, coords="XYZABC",
                                 x=0.3, y=-0.7, z=1.1)

#: name, fixture, GEOMETRY string, rotation offsets, canon kwargs. The name is
#: written out rather than derived: it is the baked expectation's file name,
#: and two entries can differ only in their rotation offsets, which no
#: sanitised GEOMETRY string can say.
CORPUS = [
    ("order_mixed__XYZ", "order_mixed.ngc", "XYZ", None, {}),
    ("order_mixed__negXYZ", "order_mixed.ngc", "-XYZ", None, {}),
    ("order_mixed__XnegYZ", "order_mixed.ngc", "X-YZ", None, {}),
    ("order_mixed__bangXYZ", "order_mixed.ngc", "!XYZ", None, {}),
    ("order_mixed__XYZUVW", "order_mixed.ngc", "XYZUVW", None, {}),
    ("dwell_m1xx__XYZ", "dwell_m1xx.ngc", "XYZ", None, {}),
    ("alternating_dwells__XYZ", "alternating_dwells.ngc", "XYZ", None, {}),
    ("hide_jump__XYZ", "hide_jump.ngc", "XYZ", None, {}),
    ("rotated_xy__XYZ", "rotated_xy.ngc", "XYZ", None, {}),
    ("rotate_midfile__XYZ", "rotate_midfile.ngc", "XYZ", None, {}),
    ("lathe_xz__XZ", "lathe_xz.ngc", "XZ", None, {}),
    ("blank_m2__XYZ", "blank_m2.ngc", "XYZ", None, {}),
    ("rotary_abc__XYZ", "rotary_abc.ngc", "XYZ", None, {}),
    ("rotary_abc__XYZA", "rotary_abc.ngc", "XYZA", RO_ROTARY, {}),
    ("rotary_abc__XYZB", "rotary_abc.ngc", "XYZB", RO_ROTARY, {}),
    ("rotary_abc__XYZC", "rotary_abc.ngc", "XYZC", RO_ROTARY, {}),
    ("rotary_abc__XYZABC", "rotary_abc.ngc", "XYZABC", RO_ROTARY, {}),
    ("rotary_abc__XYZnegAB", "rotary_abc.ngc", "XYZ-AB", RO_ROTARY, {}),
    ("rotary_abc__bangCXYZ", "rotary_abc.ngc", "!CXYZ", RO_ROTARY, {}),
    ("rotary_abc__XYZABC_offset", "rotary_abc.ngc", "XYZABC", RO_OFFSET, {}),
    ("rotary_abc__XYZABC_norotate", "rotary_abc.ngc", "XYZABC",
     bake.RotationOffsets(), {}),
    ("foam_xyuv__XY_UV", "foam_xyuv.ngc", "XY;UV", None, {"is_foam": 1}),
]


def parse_fixture(cls, name, geometry, ro, kw):
    canon = cls(geometry, **kw)
    if ro is not None:
        canon.configure_program_geometry(geometry, ro, bool(kw.get("is_foam")))
    with tempfile.NamedTemporaryFile(suffix=".var") as var:
        canon.parameter_file = var.name
        gcode.parse(os.path.join(FIXTURES, name), canon, "", "")
    return canon


# -- the baked corpus -------------------------------------------------------
#
# Every case above, parsed once by the renderer and its whole snapshot written
# to ``baked/<name>.json``. The numbers came from the renderer as phases 0-6
# left it, validated move for move against the per-move canon it replaced -
# that validation is the provenance of every literal in these files, and the
# reason a self-referential expectation is worth anything at all.
#
# Re-baking is a deliberate act, never a way to make a test pass: this file
# rewrites the whole corpus when run with ``--bake``, and the commit that does
# it says which behaviour changed and why.

BAKED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baked")


def _program_case(name, text, **kw):
    """One case built from generated G-code, error paths included.

    A parse that fails or is stopped still leaves a program - that is the
    whole point of the partial-parse handover - so the exception is caught
    here rather than excluding the case.
    """
    def build():
        path = write(text)
        canon = RendererHeadlessCanon("XYZ", **kw)
        try:
            with tempfile.NamedTemporaryFile(suffix=".var") as var:
                canon.parameter_file = var.name
                try:
                    gcode.parse(path, canon, "", "")
                except Exception:
                    pass
        finally:
            os.unlink(path)
        return canon
    return name, build


def _fixture_case(name, fixture, geometry, ro, kw):
    def build():
        return parse_fixture(RendererHeadlessCanon, fixture, geometry, ro, kw)
    return name, build


def _stopped_program():
    lines = bench_feed(400).splitlines()
    lines.insert(200, "(AXIS,stop)")
    return "\n".join(lines) + "\n"


def baked_cases():
    """``(name, build)`` for every case with a baked expectation."""
    cases = [
        _program_case("bench_feed", bench_feed()),
        _program_case("mixed", mixed()),
        _program_case("truncated_mixed", truncated_mixed()),
        _program_case("stopped", _stopped_program()),
        _program_case("hidden_spans", hidden_spans()),
        _program_case("comment_vocabulary", comment_vocabulary()),
        _program_case("stopped_inside_hidden", stopped_inside_hidden()),
        _program_case("moving_transform", moving_transform()),
        _program_case("taps_and_traverses", taps_and_traverses()),
        _program_case("tool_changes", tool_changes()),
        _program_case("arcs", arcs()),
        _program_case("feed_modes", feed_modes()),
    ]
    cases += [_program_case("random_%d" % seed, random_stream(seed))
              for seed in range(8)]
    cases += [_fixture_case(name, fixture, geometry, ro, kw)
              for name, fixture, geometry, ro, kw in CORPUS]
    return cases


BAKED_CASES = baked_cases()


def rebake():
    """Rewrite every baked expectation from this build of the renderer."""
    os.makedirs(BAKED, exist_ok=True)
    for name, build in BAKED_CASES:
        equality.save(os.path.join(BAKED, name + ".json"),
                      equality.snapshot(build()))
        print("baked %s" % name)


# -- the tests --------------------------------------------------------------

class PartialPrograms(unittest.TestCase):
    """A parse that ends early still hands over what it rendered.

    The arrays those two cases produce are baked like any other; what is
    asserted here is the thing a snapshot cannot say - that the parse really
    did fail, and that the failure did not cost the preview.
    """

    def test_a_syntax_error_leaves_a_partial_program(self):
        path = write(truncated_mixed())
        try:
            canon, (result, _seq) = parse(RendererHeadlessCanon, path)
        finally:
            os.unlink(path)
        self.assertGreater(result, gcode.MIN_ERROR, "the fixture must fail")
        self.assertEqual(canon.adopted, 1, "the parse did not render")
        self.assertTrue(len(canon.program_geometry),
                        "the fixture must draw something before it fails")

    def test_a_stopped_parse_keeps_what_it_drew(self):
        """(AXIS,stop) raises out of the comment callback, mid-program."""
        lines = bench_feed(400).splitlines()
        lines.insert(200, "(AXIS,stop)")
        path = write("\n".join(lines) + "\n")
        canon = RendererHeadlessCanon("XYZ")
        try:
            with tempfile.NamedTemporaryFile(suffix=".var") as var:
                canon.parameter_file = var.name
                with self.assertRaises(Exception):
                    gcode.parse(path, canon, "", "")
        finally:
            os.unlink(path)
        self.assertEqual(canon.adopted, 1, "the parse did not render")
        self.assertTrue(len(canon.program_geometry),
                        "the fixture must draw something before it stops")


class Protocol(unittest.TestCase):
    def test_progress_is_reported(self):
        path = write(bench_feed(20000))
        try:
            canon, _ = parse(RendererHeadlessCanon, path)
        finally:
            os.unlink(path)
        self.assertTrue(canon.progress_lines)
        self.assertEqual(sorted(canon.progress_lines), canon.progress_lines,
                         "progress line numbers must not go backwards")

    def test_the_flag_without_a_consumer_is_rejected(self):
        class NoConsumer(HeadlessCanon):
            use_gcode_renderer = True
            adopt_geometry = None

        path = write("G1 X1\nM2\n")
        try:
            with self.assertRaises(TypeError):
                parse(NoConsumer, path)
        finally:
            os.unlink(path)

    def test_only_the_bool_opts_in(self):
        """A merely truthy flag is not an opt-in.

        The rule exists for the partial canons that answer every unknown
        attribute with a stub - ``def __getattr__(self, name): return lambda
        *a: None`` - which would otherwise hand back a callable for both the
        flag and the consumer and be opted in without ever asking. Here it
        shows up as the parse falling through to the per-move callbacks, which
        this canon does not implement: loudly, rather than as an empty
        preview.
        """
        class Truthy(HeadlessCanon):
            use_gcode_renderer = 1

        path = write("G0 X0\nG1 F10 X1\nM2\n")
        try:
            with self.assertRaises(Exception):
                parse(Truthy, path)
        finally:
            os.unlink(path)

    def test_a_consumer_that_raises_fails_the_parse(self):
        class Raising(RendererHeadlessCanon):
            def adopt_geometry(self, pg):
                raise ValueError("no")

        path = write(bench_feed(200))
        try:
            with self.assertRaises(Exception):
                parse(Raising, path)
        finally:
            os.unlink(path)


class Baked(unittest.TestCase, EqualityMixin):
    """The corpus against the expectations checked in beside it.

    This is what replaces the differential-against-legacy assertion once the
    legacy fill is gone: there is one preview code path in the tree, so its
    in-tree test compares it against what it produced when it was last
    validated, and the cross-version comparison - being a comparison between
    two builds - is made outside the tree instead.
    """

    def test_every_case_matches_its_baked_expectation(self):
        for name, build in BAKED_CASES:
            with self.subTest(case=name):
                path = os.path.join(BAKED, name + ".json")
                self.assertTrue(os.path.exists(path),
                                "no baked expectation for %s - re-bake "
                                "this file with --bake" % name)
                self.assertSnapshotsEqual(equality.load(path),
                                          equality.snapshot(build()))

    def test_the_bake_covers_every_file_in_the_directory(self):
        """A stale expectation file is a case that stopped being run without anyone
        noticing, which is the one way a baked corpus rots quietly."""
        on_disk = {name[:-len(".json")]
                   for name in os.listdir(BAKED) if name.endswith(".json")}
        self.assertEqual(sorted(on_disk),
                         sorted(name for name, _ in BAKED_CASES))


class Suppression(unittest.TestCase):
    """Which lines a hidden span drops, asserted rather than baked.

    The depth is the renderer's own now, read out of the comment text after
    the canon has had it, so what it does with the words is worth stating in
    full rather than only snapshotting: a bake would carry a mis-read word
    forward as happily as a right one.
    """

    def drawn_lines(self, text):
        _name, build = _program_case("suppression", text)
        geom = build().program_geometry
        return sorted({int(line) for line in geom.lines})

    def test_only_the_words_that_are_ours_move_the_depth(self):
        """Line by line, against ``comment_vocabulary`` above.

        4 opens; 6/10 sit inside spans that ``PREVIEW,show`` closes; 13-18 are
        words the parser must not read (empty, a prefix of ``hide``, a string
        starting with it, another command, a foreign prefix, a plain comment),
        so 19 still draws; 20's trailing field does not stop it being a hide,
        so 21 does not; and the span 24 opens is never closed, so 25 and 26
        are gone with the file's end.
        """
        self.assertEqual(self.drawn_lines(comment_vocabulary()),
                         [4, 8, 12, 19, 23])

    def test_a_nested_span_closes_one_level_at_a_time(self):
        """``hide hide show`` is still hidden; the second ``show`` reopens it."""
        self.assertEqual(self.drawn_lines(
            "(nested)\nG20 G17 G90\nG0 X0 Y0 Z0\nG1 F10 X1\n"
            "(AXIS,hide)\n(AXIS,hide)\nG1 X2\n(AXIS,show)\nG1 X3\n"
            "(AXIS,show)\nG1 X4\nM2\n"), [4, 11])

    def test_a_hidden_move_does_not_move_the_chain_point(self):
        """The move after a span continues from before it, not from inside.

        The reason a hidden block looks like a jump and is not: the renderer
        returns before the chain point is touched, so the drawn path runs
        straight from the last visible point to the next one.
        """
        _name, build = _program_case(
            "chain", "(chain)\nG20 G17 G90\nG0 X0 Y0 Z0\nG1 F10 X1 Y0\n"
                     "(AXIS,hide)\nG1 X5 Y5\n(AXIS,show)\nG1 X2 Y0\nM2\n")
        geom = build().program_geometry
        pos = [tuple(round(float(v), 6) for v in p) for p in geom.positions()]
        self.assertEqual(pos, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                               (2.0, 0.0, 0.0)])

    def test_stop_wins_over_a_hide_that_never_ran(self):
        """``(AXIS,stop)`` ends the parse inside an open span.

        The canon raises out of the forward, so the renderer is not called for
        that comment at all - which is the ordering that keeps a ``stop`` from
        being read as anything else.
        """
        self.assertEqual(self.drawn_lines(stopped_inside_hidden()), [4])


class TransformOwnership(unittest.TestCase, EqualityMixin):
    """The offsets and the rotation are the renderer's, taken from the calls.

    They used to be read back off the canon's ``g5x_offset_*`` /
    ``g92_offset_*`` / ``rotation_xy`` attributes once per change, which made
    a canon able to steer the fill by writing to them. It is not any more:
    the three callbacks are still delivered, in full and in order, but what
    the canon does with them afterwards cannot move a single vertex.
    """

    PROGRAM = moving_transform()

    def build(self, cls):
        path = write(self.PROGRAM)
        try:
            canon, _ = parse(cls, path)
        finally:
            os.unlink(path)
        return canon

    def test_a_canon_that_clobbers_the_attributes_draws_the_same_program(self):
        class Clobbering(RendererHeadlessCanon):
            """Takes every call, then writes nonsense where it used to land."""

            def _wreck(self):
                for axis in "xyzabcuvw":
                    setattr(self, "g5x_offset_" + axis, 1e6)
                    setattr(self, "g92_offset_" + axis, -1e6)
                self.rotation_xy = 137.0
                self.rotation_cos = 0.0
                self.rotation_sin = 0.0

            def set_g5x_offset(self, *args):
                super().set_g5x_offset(*args)
                self._wreck()

            def set_g92_offset(self, *args):
                super().set_g92_offset(*args)
                self._wreck()

            def set_xy_rotation(self, theta):
                super().set_xy_rotation(theta)
                self._wreck()

        clean = self.build(RendererHeadlessCanon)
        wrecked = self.build(Clobbering)
        # The clobber has to have happened, or this proves nothing at all.
        self.assertEqual(wrecked.g5x_offset_x, 1e6)
        self.assertEqual(wrecked.g92_offset_x, -1e6)
        self.assertEqual(wrecked.rotation_xy, 137.0)
        self.assertNotEqual(clean.g5x_offset_x, wrecked.g5x_offset_x)
        self.assertSnapshotsEqual(equality.snapshot(clean),
                                  equality.snapshot(wrecked))

    def test_the_three_callbacks_are_still_delivered_in_full(self):
        """What the canon may no longer steer, it must still be told."""
        seen = []

        class Watching(RendererHeadlessCanon):
            def set_g5x_offset(self, *args):
                seen.append(("g5x",) + args)
                super().set_g5x_offset(*args)

            def set_g92_offset(self, *args):
                seen.append(("g92",) + args)
                super().set_g92_offset(*args)

            def set_xy_rotation(self, theta):
                seen.append(("rot", theta))
                super().set_xy_rotation(theta)

        self.build(Watching)
        kinds = [row[0] for row in seen]
        self.assertIn("g5x", kinds)
        self.assertIn("g92", kinds)
        self.assertGreater(len({row[1] for row in seen if row[0] == "rot"}), 1,
                           "the fixture must turn the program more than once")


class ToolList(unittest.TestCase):
    """``canon.tool_list`` is rebuilt from the record, not appended to.

    ``GLCanon.change_tool`` used to grow the list one event at a time, beside
    the record the renderer was writing for the same event.  Now
    ``adopt_geometry`` reads it off the record in one pass. The list is the
    same list - emission order, repeats and T0 included, a change inside a
    hidden span included - and the properties dialog, its only reader in the
    tree, sees no difference. What changed is when it appears: at the end of
    the parse, rather than during it.
    """

    def parse(self, text):
        _name, build = _program_case("tools", text)
        return build()

    def test_every_change_lands_in_the_list_in_order(self):
        canon = self.parse(tool_changes())
        self.assertEqual(canon.tool_list, [0, 0, 0])
        self.assertEqual([lineno for lineno, _t, _p
                          in canon.program_geometry.toolchanges],
                         [5, 7, 9], "including the one at line 9, hidden")

    def test_a_program_with_no_changes_leaves_it_empty(self):
        canon = self.parse("(none)\nG20 G17 G90\nG0 X0 Y0 Z0\n"
                           "G1 F10 X1\nM2\n")
        self.assertEqual(canon.tool_list, [])

    def test_a_partial_parse_hands_over_the_partial_list(self):
        """An aborted parse keeps the changes it got to, as it always did."""
        canon = self.parse("(stopped)\nG20 G17 G90\nG0 X0 Y0 Z0\n"
                           "G1 F10 X1\nM6\nG1 X2\n(AXIS,stop)\n"
                           "M6\nG1 X3\nM2\n")
        self.assertEqual(canon.tool_list, [0])

    def test_the_numbers_are_the_ones_commanded(self):
        """The half a headless parse cannot reach: T numbers that are not 0.

        Built as a handover rather than parsed, so the production
        ``adopt_geometry`` is what runs. Repeats, a T0 among them and the
        order are what a properties dialog reads out.
        """
        changes = [(10, 3, ((0.0, 0.0, 0.0),)),
                   (20, 3, ((1.0, 0.0, 0.0),)),
                   (30, 0, ((2.0, 0.0, 0.0),)),
                   (40, 99, ((3.0, 0.0, 0.0),))]
        canon = HeadlessCanon("XYZ")
        canon.adopt_geometry(FakePreview(
            [np.zeros((2, 3), dtype=np.float32)], [1, 1],
            [bake.KIND_FEED, bake.KIND_FEED], toolchanges=changes,
            tool_numbers=[None, 3, 3, 0, 99]))
        self.assertEqual(canon.tool_list, [3, 3, 0, 99])


class Invariants(unittest.TestCase):
    """Facts about any program the renderer builds, whatever the numbers.

    These hold no matter what is baked, so they catch the class of change a
    re-bake would otherwise wave through.
    """

    def geometry(self, text, **kw):
        _name, build = _program_case("invariant", text, **kw)
        return build().program_geometry

    def test_every_drawn_vertex_lies_inside_the_drawn_extents(self):
        geom = self.geometry(mixed())
        low, high = geom.drawn_extents
        pos = geom.positions()
        self.assertTrue((pos >= np.float32(low)).all(), "below the box")
        self.assertTrue((pos <= np.float32(high)).all(), "above the box")

    def test_the_record_vertices_match_the_record_tables(self):
        geom = self.geometry(mixed())
        kinds = geom.kinds
        self.assertEqual(int((kinds == bake.KIND_DWELL).sum()),
                         len(geom.dwells), "dwell markers")
        self.assertEqual(int((kinds == bake.KIND_TOOLCHANGE).sum()),
                         len(geom.toolchanges), "tool-change markers")

    def test_the_spare_bits_of_the_kind_tool_word_stay_zero(self):
        geom = self.geometry(mixed())
        self.assertEqual(int((geom.kindtool & bake.SPARE_MASK).max()), 0)

    def test_lengths_are_non_negative_and_scale_with_the_program(self):
        one = self.geometry(bench_feed(500))
        two = self.geometry(bench_feed(1000))
        for geom in (one, two):
            self.assertGreaterEqual(geom.rapid_length, 0.0)
            self.assertGreaterEqual(geom.cutting_length, 0.0)
        self.assertGreater(two.cutting_length, one.cutting_length)

    def test_a_contiguous_program_draws_one_vertex_per_move_plus_the_start(self):
        """No rotary motion, no jump, no record: the strip is the moves."""
        geom = self.geometry("G20 G17 G90\nG0 X0 Y0 Z0\nF10\n"
                             + "".join("G1 X%.3f\n" % (i * 0.01)
                                       for i in range(1, 51))
                             + "M2\n")
        self.assertEqual(len(geom), geom.n_moves + 1)


class CallbackCanon:
    """A canon on the per-move *callback* protocol, which this change leaves
    exactly as it was.

    Not a preview: it is the shape ``rs274.interpret``'s ``PrintCanon``, the
    interpreter tests and out-of-tree users of ``gcode.parse`` have - a
    catch-all that answers every canon call with a no-op. The explicit
    ``use_gcode_renderer = False`` is what that catch-all makes necessary: it
    would otherwise answer the opt-in probe with a callable.
    """

    use_gcode_renderer = False

    def __init__(self, *args, **kw):
        self.rates = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args: None

    def set_feed_rate(self, rate):
        self.rates.append(rate)

    def get_external_length_units(self): return 1.0
    def get_external_angular_units(self): return 1.0
    def get_axis_mask(self): return 0x1ff
    def get_block_delete(self): return False
    def get_tool(self, pocket): return (-1,) + (0.0,) * 12 + (0,)


class FeedRateForwarding(unittest.TestCase, EqualityMixin):
    """An F word that changes nothing costs nothing in renderer mode.

    ``interp_execute.cc`` calls ``SET_FEED_RATE`` for every block carrying an F
    word and never compares it to the rate already in force, so CAM output that
    repeats one rate on every line - which is most of it - would otherwise be
    one forwarded Python call per move in a protocol whose whole point is not
    having those. A canon on the per-move *callback* protocol must keep
    receiving every one of them.
    """

    #: The same rate on every line, then a real change, then the same again.
    REPEATED = ("G20 G17 G90\nG0 X0 Y0 Z0.1\n"
                + "".join("G1 F600 X%.3f\n" % (i * 0.01) for i in range(10))
                + "".join("G1 F900 X%.3f\n" % (1.0 + i * 0.01)
                          for i in range(10))
                + "M2\n")

    @staticmethod
    def rates(cls, path):
        seen = []
        real = cls.set_feed_rate

        class Counting(cls):
            def set_feed_rate(self, arg):
                seen.append(arg)
                return real(self, arg)

        canon, _ = parse(Counting, path)
        return canon, seen

    def setUp(self):
        self.path = write(self.REPEATED)
        self.addCleanup(os.unlink, self.path)

    def test_the_renderer_forwards_only_the_changes(self):
        _, rates = self.rates(RendererHeadlessCanon, self.path)
        # The trailing 0.0 is the interpreter resetting the rate at M2 - a real
        # change, so it is forwarded like any other.
        self.assertEqual(rates, [600.0, 900.0, 0.0])

    def test_a_callback_canon_still_sees_every_f_word(self):
        canon, rates = self.rates(CallbackCanon, self.path)
        self.assertEqual(rates, [600.0] * 10 + [900.0] * 10 + [0.0])
        self.assertEqual(canon.rates, rates, "the canon saw them too")

    def test_the_first_f_word_is_always_forwarded(self):
        """Even at 60.0, which is what the C-side tracker starts at.

        Otherwise a consumer's own starting feed rate would be right only by
        the coincidence of matching that initial value.
        """
        path = write("G20 G17 G90\nG1 F60 X1\nM2\n")
        self.addCleanup(os.unlink, path)
        _, rates = self.rates(RendererHeadlessCanon, path)
        self.assertEqual(rates, [60.0, 0.0])

    def test_the_suppressed_calls_do_not_cost_the_rate_itself(self):
        rendered, _ = self.rates(RendererHeadlessCanon, self.path)
        self.assertGreater(rendered.program_geometry.cutting_length, 0.0)
        self.assertEqual(sorted(rendered.program_geometry
                                ._cut_length_by_feed), [10.0, 15.0])


if __name__ == "__main__":
    if "--bake" in sys.argv:
        rebake()
    else:
        unittest.main()
