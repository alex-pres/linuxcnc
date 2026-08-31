#!/bin/sh
# Unit tests for the OpenGL 3.3 core / GLES 3.1 preview renderer shared by
# AXIS, the GTK screens (Gremlin/gmoccapy/gscreen/hal_gremlin) and QtVCP.
#
# All are GL-free: they exercise the backplot palette, the explicit camera
# matrices and the Hershey label font as plain Python, so they need neither a
# GPU nor an X display.
set -e

python3 test_backplot_palette.py >&2
python3 test_camera_matrices.py >&2
python3 test_hershey_diameter.py >&2

echo ok
