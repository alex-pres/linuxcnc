#!/usr/bin/env python3
"""A hand-written stand-in for the ``gcode.PreviewGeometry`` C hands over.

The program record is built in C++ and adopted whole; a test that wants a
*particular* record - four vertices with known kinds, eleven dwells in eleven
colours - cannot get one out of a parse, and there is no Python fill left to
build one with. So it builds the handover instead: this duck-types what
:meth:`rs274.glcanon_bake.ProgramGeometry.adopt` reads, which means the
production adopt path is what runs, not a test-only shortcut past it.

Deliberately *not* a way to test the renderer: nothing here says what a
program should contain, only what shape a handover has. What the renderer
produces is pinned by the baked corpus in ``test_move_renderer.py`` and by
the parse-driven cases in ``test_program_record.py``.

GL-free; needs numpy only.
"""
import numpy as np

PLANE_POS = np.dtype(np.float32)


class FakePreview:
    """Everything ``ProgramGeometry.adopt`` asks a handover for."""

    def __init__(self, planes, lines, kinds, tools=None, moves=None,
                 rapid_length=0.0, cut_lengths=None, tool_numbers=None,
                 dwells=(), toolchanges=(), dwell_time=0.0, extents=None):
        self._planes = [np.ascontiguousarray(p, dtype=np.float32)
                        for p in planes]
        lines = np.asarray(lines, dtype=np.uint32)
        kinds = np.asarray(kinds, dtype=np.uint32)
        tools = (np.zeros(len(lines), dtype=np.uint32) if tools is None
                 else np.asarray(tools, dtype=np.uint32))
        self._attrs = np.empty((len(lines), 2), dtype=np.uint32)
        self._attrs[:, 0] = lines
        self._attrs[:, 1] = kinds | (tools << np.uint32(8))
        self.n_vertices = len(lines)
        self.n_planes = len(self._planes)
        self.n_moves = len(lines) - 1 if moves is None else moves
        self.rapid_length = float(rapid_length)
        self.dwell_time = float(dwell_time)
        self._cut_lengths = dict(cut_lengths or {})
        self._tool_numbers = list(tool_numbers or [None])
        self._dwells = list(dwells)
        self._toolchanges = list(toolchanges)
        if extents is None:
            box = self._box()
            extents = [box] * 4
        self._extents = extents

    def _box(self):
        stacked = np.concatenate(self._planes)
        return (tuple(stacked.min(axis=0).tolist()),
                tuple(stacked.max(axis=0).tolist()))

    def positions(self, plane=0):
        return self._planes[plane]

    def attrs(self):
        return self._attrs

    def extents(self):
        return self._extents

    def drawn_extents(self):
        return self._box()

    def cut_lengths(self):
        return dict(self._cut_lengths)

    def tool_numbers(self):
        return list(self._tool_numbers)

    def dwells(self):
        return list(self._dwells)

    def toolchanges(self):
        return list(self._toolchanges)
