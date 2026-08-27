#!/usr/bin/env python3
"""G-code fixtures generated inline, shared by the renderer tests and benches.

Generated rather than checked in so that what each one exercises is readable
next to the assertions that use it. The checked-in fixtures under
``fixtures/`` are the other half of the corpus: those stand in for real
configs (a lathe, a foam cutter, a rotary program), these stand in for
program *shapes* (a long run of feeds, one of nearly everything, a file that
stops mid-word).

GL-free, but the canons that consume them need the built ``gcode``
extension, so their tests run against a built tree.
"""
import os
import tempfile


def write(text):
    """``text`` in a temporary ``.ngc`` file; the caller unlinks it."""
    fd, path = tempfile.mkstemp(suffix=".ngc", prefix="gcode-bake-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def bench_feed(moves=2000):
    """N short G1 moves drifting in x, y and z, with the feed rate changing.

    Long enough at the default to span more than one progress report, short
    enough to parse in well under a second.
    """
    out = ["(bench_feed)", "G20 G17 G90", "G0 X0 Y0 Z0.1", "F10"]
    for i in range(moves):
        if i % 97 == 0:
            out.append("F%d" % (10 + i % 40))
        out.append("G1 X%.4f Y%.4f Z%.4f"
                   % (i * 0.001, (i % 31) * 0.002, -((i % 7) * 0.001)))
    out.append("M2")
    return "\n".join(out) + "\n"


def mixed():
    """One of nearly everything, in an order that stresses the state machine.

    Deliberately not a tidy program. What each group is here for:

      * leading G0s before any cutting move - the ``first_move`` drop;
      * F and S words between moves - the feed rate a move is recorded at, and
        the fact that an S word must not disturb anything;
      * G2/G3 - arcs, segmented by the renderer itself, which have to
        interleave with the straight moves in the right order;
      * G4 and M1xx between moves - events between moves, and the dwell plane
        after a plane change;
      * G92, G10 L2 with an R rotation, G54/G55 - the transform changing
        between moves;
      * M6 and G43.1/G49 - the tool change and tool offset, one pair of them
        inside a hidden span, where they apply despite the suppression;
      * G38.2 and G33.1 - the probe and rigid-tap kinds;
      * G20/G21 - the metric conversion, which happens C-side;
      * G5.2/G5.3 - NURBS, which feed through STRAIGHT_FEED internally;
      * G81/G82 - a canned cycle, whose per-hole dwells must not disturb the
        moves around them.

    ``T`` words are absent on purpose: the standalone ``gcode`` module has no
    tool-data table and a ``T`` word segfaults the interpreter, which is a
    limitation of parsing without a running LinuxCNC. ``M6`` alone still
    changes the tool, and ``G43.1`` still sets an offset, so both are covered
    end to end.
    """
    return """(mixed)
G20 G17 G90
S800 M3
G0 X0 Y0 Z1
G0 X0.1 Y0.1
G0 X0.2 Y0.2
G1 F12 X1 Y0 Z0
G1 X1 Y1
F30
G2 X0 Y1 I-0.5 J0
G3 X0 Y0 I0 J-0.5
G1 X0.5 Y0.5
G4 P0.25
G1 X0.6 Y0.6
M100 P1 Q2
G1 X0.7 Y0.7
G18
G4 P0.5
G1 X0.8 Y0.8
G17
G4 P0.1
S900
G1 X0.9 Y0.9
G92 X0.1 Y0.2
G1 X2 Y2
G10 L2 P1 X0.05 Y0.05 R15
G54
G1 X2.5 Y2.5
G55
G1 X2.6 Y2.6
G54
G92.1
(MSG, a message is still forwarded)
(AXIS,hide)
G1 X3 Y3
M6
G43.1 Z0.3
G1 X3.2 Y3.2
(AXIS,show)
G1 X3.5 Y3.5
G38.2 Z-0.2 F5
S1200 M3
G33.1 Z-0.3 K0.05
G33.1 Z-0.35 K0.05
G1 X3.6 Y3.6
G21
G1 X92 Y92
G20
G81 X4 Y4 Z-0.1 R0.2 L3
G82 X5 Y5 Z-0.1 R0.2 P0.3
G80
G49
G1 X5.5 Y5.5
G5.2 X6 Y6 P1
X6.5 Y6.8 P1
X7 Y6 P1
G5.3
S1500
G1 X7.5 Y7.5
M2
"""


def truncated_mixed():
    """``mixed`` cut short by a syntax error, for the partial-parse case."""
    return mixed().replace("G1 X3.5 Y3.5", "G1 X3.5 Y3.5\nG1 X(bogus")
