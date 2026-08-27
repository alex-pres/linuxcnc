#!/bin/sh
# Unit tests for the C++ G-code preview renderer.
#
# What gcode.parse hands over - the program record GCodeRenderer builds - and
# what rs274.glcanon_bake makes of it. Nothing here is a checked-in
# expectation: every program is generated into a tempfile for the length of
# one parse, and every expected value is either arithmetic written out in the
# test, an answer from the independent line9 reference, or a property that
# holds whatever the numbers are.
#
# test_reference.py and test_bake.py are GL-free and need numpy alone, so they
# run on a tree with no built extension; the rest drive a real parse and need
# the built gcode extension, which runtests makes importable by sourcing
# rip-environment.
set -e

# The independent reference first: everything below leans on it, so if it has
# drifted from the C it is pinned against, nothing else means anything.
python3 test_reference.py >&2

# The renderer: its transform against that reference, then its behaviour and
# the shape of the record it hands over.
python3 test_transform.py >&2
python3 test_renderer.py >&2
python3 test_record.py >&2

# What readers downstream ask a finished program, and what the bake makes of
# one on its way to the GPU.
python3 test_queries.py >&2
python3 test_bake.py >&2

echo ok
