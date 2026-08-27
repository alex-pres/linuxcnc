#!/bin/sh
# Unit tests for the C++ G-code preview renderer.
#
# What gcode.parse hands over - the program record built by GCodeRenderer -
# and what rs274.glcanon_bake builds from it. Two of them are GL-free and need
# only numpy; the rest drive a real parse, so they need the built gcode
# extension, which runtests makes importable by sourcing rip-environment.
set -e

# The independent references first: if these are wrong, nothing below means
# anything.
python3 test_line9_bake_reference.py >&2
python3 test_bake_vs_reference.py >&2

# The renderer itself, against its baked corpus and its protocol.
python3 test_gcode_renderer.py >&2
python3 test_program_record.py >&2
python3 test_canon_order.py >&2

# What readers downstream ask a finished program.
python3 test_extents_oracle.py >&2
python3 test_highlight_centroid.py >&2
python3 test_length_time.py >&2

echo ok
