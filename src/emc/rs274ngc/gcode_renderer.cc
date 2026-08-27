//    This is a component of AXIS, a front-end for emc
//    Copyright 2004, 2005, 2006 Jeff Epler <jepler@unpythonic.net> and
//    Chris Radek <chris@timeguy.com>
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program; if not, write to the Free Software
//    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// The G-code renderer: the whole preview pipeline, run during the parse. Only
// the one entry point a canon function reaches per move stays inline in
// gcode_renderer.hh, where the call sites can see it.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nml_intf/canon.hh"
#include "posemath.h"

#include "gcode_renderer.hh"

void GCodeRenderer::progress() {
    if(!active()) return;
    // Never call into Python with an exception pending: the canon would be
    // handed a broken interpreter state, and the error we already have is the
    // one worth reporting.
    if(interp_error) return;
    if(!owned()) return;
    instance_->report_progress();
}

void GCodeRenderer::comment(const char *text) {
    if(!active()) return;
    // The canon has already had this comment. If it raised - `(AXIS,stop)`
    // does - the parse is over, and the word is not read for a hide either.
    if(interp_error) return;
    if(!owned()) return;
    instance_->note_comment(text);
}

// The transform. Each of the three arrives in inches, already converted, at
// the moment its callback forwards - so there is nothing to read back off the
// canon and nothing that can go stale between the call and the move it
// applies to.
void GCodeRenderer::set_g5x(const double offsets[9]) {
    if(!active() || interp_error || !owned()) return;
    memcpy(instance_->g5x_, offsets, sizeof instance_->g5x_);
}

void GCodeRenderer::set_g92(const double offsets[9]) {
    if(!active() || interp_error || !owned()) return;
    memcpy(instance_->g92_, offsets, sizeof instance_->g92_);
}

void GCodeRenderer::set_rotation_xy(double degrees) {
    if(!active() || interp_error || !owned()) return;
    GCodeRenderer *r = instance_;
    r->rotation_xy_ = degrees;
    // `M_PI / 180.0` folded first, which is what `math.radians` multiplies by:
    // the fill this replaced took its sin and cos from the canon, so keeping
    // the argument bit-identical keeps the baked expectations so too.
    double rad = degrees * (M_PI / 180.0);
    r->rotation_cos_ = cos(rad);
    r->rotation_sin_ = sin(rad);
    // The angle back, as it has always been computed here.
    double back = -degrees * M_PI / 180.0;
    r->unrot_cos_ = cos(back);
    r->unrot_sin_ = sin(back);
}
void GCodeRenderer::finish() { if(active()) instance_->hand_over(); }
void GCodeRenderer::note_plane(int plane) {
    if(active()) instance_->plane_ = plane;
}

void GCodeRenderer::arc(int line_number, double first_end, double second_end,
                        double first_axis, double second_axis, int rotation,
                        double axis_end_point, double a, double b, double c,
                        double u, double v, double w) {
    if(!owned()) return;
    last_sequence_number = line_number;
    instance_->render_arc(line_number, first_end, second_end, first_axis,
                          second_axis, rotation, axis_end_point, a, b, c,
                          u, v, w, rate_);
}

// ---------------------------------------------------------------------------
// One parse's renderer
// ---------------------------------------------------------------------------

// Read one float attribute. False with an exception set, as the caller stops.
static bool get_double(PyObject *o, const char *name, double *out) {
    PyObject *v = PyObject_GetAttrString(o, name);
    if(!v) return false;
    double d = PyFloat_AsDouble(v);
    Py_DECREF(v);
    if(d == -1.0 && PyErr_Occurred()) return false;
    *out = d;
    return true;
}

static const char AXES[9] = {'x','y','z','a','b','c','u','v','w'};

// ---------------------------------------------------------------------------
// PreviewData: storage, growth, and the Python object it is handed over in
// ---------------------------------------------------------------------------

PreviewData::~PreviewData() {
    for(int i = 0; i < 2; i++) free(pos[i]);
    free(attrs);
}

bool PreviewData::reserve(size_t extra) {
    size_t need = n + extra;
    if(need <= cap) return true;
    size_t want = cap * 2;
    if(want < 1024) want = 1024;
    while(want < need) want *= 2;
    float *grown[2] = {nullptr, nullptr};
    for(int i = 0; i < nplanes; i++) {
        grown[i] = (float*)realloc(pos[i], want * 3 * sizeof(float));
        if(!grown[i]) {
            // Whatever did grow is still valid and still holds the program;
            // only `cap` decides what may be written, so leaving it alone is
            // what makes this safe to fail.
            for(int j = 0; j < i; j++) pos[j] = grown[j];
            return false;
        }
    }
    uint32_t *grown_attrs =
        (uint32_t*)realloc(attrs, want * 2 * sizeof(uint32_t));
    if(!grown_attrs) {
        for(int i = 0; i < nplanes; i++) pos[i] = grown[i];
        return false;
    }
    for(int i = 0; i < nplanes; i++) pos[i] = grown[i];
    attrs = grown_attrs;
    cap = want;
    return true;
}

// Give back the doubling slack once the program is complete - up to half the
// array, and this is the copy every reader keeps.
void PreviewData::shrink() {
    if(cap == n) return;
    size_t want = n ? n : 1;
    for(int i = 0; i < nplanes; i++) {
        float *fit = (float*)realloc(pos[i], want * 3 * sizeof(float));
        if(fit) pos[i] = fit;           // a refused shrink keeps the slack
    }
    uint32_t *fit = (uint32_t*)realloc(attrs, want * 2 * sizeof(uint32_t));
    if(fit) attrs = fit;
    cap = want;
}

typedef struct {
    PyObject_HEAD
    PreviewData *data;
} PreviewGeometry;

// A read-only buffer over part of a PreviewGeometry, keeping it alive. What
// numpy wraps, so the program reaches Python without a copy.
typedef struct {
    PyObject_HEAD
    PyObject *owner;
    void *ptr;
    Py_ssize_t nbytes, itemsize;
    const char *format;
} ArrayView;

static void ArrayView_dealloc(ArrayView *self) {
    Py_XDECREF(self->owner);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int ArrayView_getbuffer(ArrayView *self, Py_buffer *view, int flags) {
    int bad = PyBuffer_FillInfo(view, (PyObject*)self, self->ptr,
                                self->nbytes, 1, flags);
    if(bad) return bad;
    view->itemsize = self->itemsize;
    view->format = (flags & PyBUF_FORMAT) ? (char*)self->format : nullptr;
    return 0;
}

static PyBufferProcs ArrayView_as_buffer = {
    (getbufferproc)ArrayView_getbuffer,
    nullptr,
};

// Both types are filled in by preview_geometry_ready(): a designated
// initializer cannot follow PyVarObject_HEAD_INIT in C++. The remaining
// fields are value-initialized to zero, which is what a static PyTypeObject
// wants; -Wmissing-field-initializers has nothing real to say about it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyTypeObject ArrayViewType = { PyVarObject_HEAD_INIT(nullptr, 0) };
#pragma GCC diagnostic pop

static PyObject *array_view(PyObject *owner, void *ptr, Py_ssize_t nbytes,
                            Py_ssize_t itemsize, const char *format) {
    ArrayView *v = PyObject_New(ArrayView, &ArrayViewType);
    if(!v) return nullptr;
    Py_INCREF(owner);
    v->owner = owner;
    v->ptr = ptr;
    v->nbytes = nbytes;
    v->itemsize = itemsize;
    v->format = format;
    return (PyObject*)v;
}

static void PreviewGeometry_dealloc(PreviewGeometry *self) {
    delete self->data;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *pg_positions(PreviewGeometry *self, PyObject *args) {
    int plane = 0;
    if(!PyArg_ParseTuple(args, "|i:positions", &plane)) return nullptr;
    if(plane < 0 || plane >= self->data->nplanes) {
        PyErr_SetString(PyExc_IndexError, "no such drawn plane");
        return nullptr;
    }
    return array_view((PyObject*)self, self->data->pos[plane],
                      (Py_ssize_t)self->data->n * 3 * sizeof(float),
                      sizeof(float), "f");
}

static PyObject *pg_attrs(PreviewGeometry *self, PyObject *) {
    return array_view((PyObject*)self, self->data->attrs,
                      (Py_ssize_t)self->data->n * 2 * sizeof(uint32_t),
                      sizeof(uint32_t), "I");
}

static PyObject *triple(const double *v) {
    return Py_BuildValue("ddd", v[0], v[1], v[2]);
}

static PyObject *pg_extents(PreviewGeometry *self, PyObject *) {
    PyObject *out = PyTuple_New(4);
    if(!out) return nullptr;
    for(int i = 0; i < 4; i++)
        PyTuple_SET_ITEM(out, i, Py_BuildValue("NN",
                    triple(self->data->extents[i][0]),
                    triple(self->data->extents[i][1])));
    return out;
}

static PyObject *pg_drawn_extents(PreviewGeometry *self, PyObject *) {
    return Py_BuildValue("NN", triple(self->data->drawn[0]),
                         triple(self->data->drawn[1]));
}

static PyObject *pg_cut_lengths(PreviewGeometry *self, PyObject *) {
    PyObject *out = PyDict_New();
    if(!out) return nullptr;
    for(auto &entry : self->data->cut_length) {
        PyObject *k = PyFloat_FromDouble(entry.first);
        PyObject *v = PyFloat_FromDouble(entry.second);
        if(!k || !v || PyDict_SetItem(out, k, v) < 0) {
            Py_XDECREF(k); Py_XDECREF(v); Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(k); Py_DECREF(v);
    }
    return out;
}

static PyObject *pg_tool_numbers(PreviewGeometry *self, PyObject *) {
    size_t n = self->data->tool_numbers.size();
    PyObject *out = PyList_New(n);
    if(!out) return nullptr;
    // Ordinal 0 is the state before any tool change: not stated, not T0.
    Py_INCREF(Py_None);
    PyList_SET_ITEM(out, 0, Py_None);
    for(size_t i = 1; i < n; i++)
        PyList_SET_ITEM(out, i, PyLong_FromLong(self->data->tool_numbers[i]));
    return out;
}

static PyObject *points_tuple(const double pts[2][3], int nplanes) {
    PyObject *out = PyTuple_New(nplanes);
    if(!out) return nullptr;
    for(int i = 0; i < nplanes; i++) PyTuple_SET_ITEM(out, i, triple(pts[i]));
    return out;
}

static PyObject *pg_dwells(PreviewGeometry *self, PyObject *) {
    PyObject *out = PyList_New(self->data->dwells.size());
    if(!out) return nullptr;
    Py_ssize_t at = 0;
    for(const DwellRecord &d : self->data->dwells) {
        PyObject *row = Py_BuildValue("iiONN", d.lineno, d.plane,
                d.m1xx ? Py_True : Py_False, triple(d.raw),
                points_tuple(d.pts, self->data->nplanes));
        if(!row) { Py_DECREF(out); return nullptr; }
        PyList_SET_ITEM(out, at ++, row);
    }
    return out;
}

static PyObject *pg_toolchanges(PreviewGeometry *self, PyObject *) {
    PyObject *out = PyList_New(self->data->toolchanges.size());
    if(!out) return nullptr;
    Py_ssize_t at = 0;
    for(const ToolChangeRecord &c : self->data->toolchanges) {
        PyObject *row = Py_BuildValue("iiN", c.lineno, c.tool,
                points_tuple(c.pts, self->data->nplanes));
        if(!row) { Py_DECREF(out); return nullptr; }
        PyList_SET_ITEM(out, at ++, row);
    }
    return out;
}

static PyMethodDef PreviewGeometry_methods[] = {
    {"positions", (PyCFunction)pg_positions, METH_VARARGS,
        "Read-only float32 xyz view of one drawn plane"},
    {"attrs", (PyCFunction)pg_attrs, METH_NOARGS,
        "Read-only uint32 (line, kind|tool) view"},
    {"extents", (PyCFunction)pg_extents, METH_NOARGS,
        "The four machine-frame (min, max) pairs"},
    {"drawn_extents", (PyCFunction)pg_drawn_extents, METH_NOARGS,
        "(min, max) over the transformed points in the array"},
    {"cut_lengths", (PyCFunction)pg_cut_lengths, METH_NOARGS,
        "{commanded rate: cutting length at it}"},
    {"tool_numbers", (PyCFunction)pg_tool_numbers, METH_NOARGS,
        "Ordinal -> T number, entry 0 None"},
    {"dwells", (PyCFunction)pg_dwells, METH_NOARGS,
        "(lineno, plane, is_m1xx, raw xyz, points per plane) per dwell"},
    {"toolchanges", (PyCFunction)pg_toolchanges, METH_NOARGS,
        "(lineno, tool number, points per plane) per tool change"},
    {},
};

static PyObject *pg_get_n(PreviewGeometry *self, void *) {
    return PyLong_FromSize_t(self->data->n);
}
static PyObject *pg_get_moves(PreviewGeometry *self, void *) {
    return PyLong_FromSize_t(self->data->moves);
}
static PyObject *pg_get_planes(PreviewGeometry *self, void *) {
    return PyLong_FromLong(self->data->nplanes);
}
static PyObject *pg_get_rapid(PreviewGeometry *self, void *) {
    return PyFloat_FromDouble(self->data->rapid_length);
}
static PyObject *pg_get_dwell_time(PreviewGeometry *self, void *) {
    return PyFloat_FromDouble(self->data->dwell_time);
}

static PyGetSetDef PreviewGeometry_getset[] = {
    {(char*)"n_vertices", (getter)pg_get_n, nullptr, nullptr, nullptr},
    {(char*)"n_moves", (getter)pg_get_moves, nullptr, nullptr, nullptr},
    {(char*)"n_planes", (getter)pg_get_planes, nullptr, nullptr, nullptr},
    {(char*)"rapid_length", (getter)pg_get_rapid, nullptr, nullptr, nullptr},
    {(char*)"dwell_time", (getter)pg_get_dwell_time, nullptr, nullptr, nullptr},
    {},
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyTypeObject PreviewGeometryType = { PyVarObject_HEAD_INIT(nullptr, 0) };
#pragma GCC diagnostic pop

bool preview_geometry_ready() {
    ArrayViewType.tp_name = "gcode.arrayview";
    ArrayViewType.tp_basicsize = sizeof(ArrayView);
    ArrayViewType.tp_dealloc = (destructor)ArrayView_dealloc;
    ArrayViewType.tp_as_buffer = &ArrayView_as_buffer;
    ArrayViewType.tp_flags = Py_TPFLAGS_DEFAULT;
    ArrayViewType.tp_doc = "Read-only view of a PreviewGeometry array";

    PreviewGeometryType.tp_name = "gcode.PreviewGeometry";
    PreviewGeometryType.tp_basicsize = sizeof(PreviewGeometry);
    PreviewGeometryType.tp_dealloc = (destructor)PreviewGeometry_dealloc;
    PreviewGeometryType.tp_flags = Py_TPFLAGS_DEFAULT;
    PreviewGeometryType.tp_doc =
        "A parsed program: vertex arrays, extents, lengths and records";
    PreviewGeometryType.tp_methods = PreviewGeometry_methods;
    PreviewGeometryType.tp_getset = PreviewGeometry_getset;

    return PyType_Ready(&PreviewGeometryType) >= 0
        && PyType_Ready(&ArrayViewType) >= 0;
}

PyObject *preview_geometry_new(PreviewData *data) {
    PreviewGeometry *pg = PyObject_New(PreviewGeometry, &PreviewGeometryType);
    if(!pg) { delete data; return nullptr; }
    pg->data = data;
    return (PyObject*)pg;
}

// Compile one GEOMETRY string into the steps transform_points walks, and read
// the rotation offsets beside it. Both come from the canon's ProgramGeometry,
// which is where the widget put them just before the parse.
bool GCodeRenderer::read_planes() {
    PyObject *pg = PyObject_GetAttrString(canon_, "program_geometry");
    if(!pg) return false;
    PyObject *planes = PyObject_GetAttrString(pg, "planes");
    PyObject *ro = PyObject_GetAttrString(pg, "ro");
    Py_DECREF(pg);
    if(!planes || !ro) { Py_XDECREF(planes); Py_XDECREF(ro); return false; }

    double rox = 0, roy = 0, roz = 0;
    long mask = 0;
    PyObject *respect = PyObject_GetAttrString(ro, "respect_offsets");
    PyObject *m = PyObject_GetAttrString(ro, "axis_mask");
    if(!respect || !m
    || !get_double(ro, "x", &rox) || !get_double(ro, "y", &roy)
    || !get_double(ro, "z", &roz)) {
        Py_XDECREF(respect); Py_XDECREF(m);
        Py_DECREF(planes); Py_DECREF(ro);
        return false;
    }
    respect_offsets_ = PyObject_IsTrue(respect);
    mask = PyLong_AsLong(m);
    Py_DECREF(respect); Py_DECREF(m); Py_DECREF(ro);
    data_->respect_offsets = respect_offsets_;

    Py_ssize_t n = PySequence_Size(planes);
    if(n < 1 || n > 2) {
        Py_DECREF(planes);
        PyErr_SetString(PyExc_ValueError,
                "parse: the renderer draws one or two planes");
        return false;
    }
    data_->nplanes = (int)n;
    for(Py_ssize_t i = 0; i < n; i++) {
        PyObject *s = PySequence_GetItem(planes, i);
        const char *geom = s ? PyUnicode_AsUTF8(s) : nullptr;
        if(!geom) { Py_XDECREF(s); Py_DECREF(planes); return false; }
        double sign = 1.0;
        for(const char *ch = geom; *ch; ch++) {
            GeomOp op = {};
            op.sign = sign;
            switch(*ch) {
            case '-': sign = -1.0; continue;
            case 'X': op.col = 0; op.a = 0; break;
            case 'Y': op.col = 1; op.a = 1; break;
            case 'Z': op.col = 2; op.a = 2; break;
            case 'U': op.col = 6; op.a = 0; break;
            case 'V': op.col = 7; op.a = 1; break;
            case 'W': op.col = 8; op.a = 2; break;
            case 'A': case 'B': case 'C': {
                // A rotary letter turns a component pair - but only when the
                // config asked for it, which is what the mask says.
                int bit = *ch == 'A' ? 0x08 : *ch == 'B' ? 0x10 : 0x20;
                sign = 1.0;
                if(!(mask & bit)) continue;
                op.rotate = true;
                if(*ch == 'A') {
                    op.col = 3; op.a = 1; op.b = 2; op.offa = roy; op.offb = roz;
                } else if(*ch == 'B') {
                    op.col = 4; op.a = 0; op.b = 2; op.offa = rox; op.offb = roz;
                } else {
                    op.col = 5; op.a = 0; op.b = 1; op.offa = rox; op.offb = roy;
                }
                data_->ops[i].push_back(op);
                continue;
            }
            default: continue;          // '!', ';' and friends, sign preserved
            }
            sign = 1.0;
            data_->ops[i].push_back(op);
        }
        Py_DECREF(s);
    }
    Py_DECREF(planes);
    return true;
}

bool GCodeRenderer::arm(PyObject *canon) {
    owner_ = nullptr;
    rate_ = 60.0;
    rate_seen_ = false;
    speed_ = 0.0;
    delete instance_;
    instance_ = nullptr;

    PyObject *flag = PyObject_GetAttrString(canon, "use_gcode_renderer");
    if(!flag) {
        if(!PyErr_ExceptionMatches(PyExc_AttributeError)) return false;
        PyErr_Clear();              // no attribute: the callback protocol
        return true;
    }
    // Anything that is not a bool is not an opt-in - see the note on
    // catch-all `__getattr__` in the header. Callback protocol, no complaint:
    // a canon that never mentions the flag must not be made to fail.
    bool opted_in = PyBool_Check(flag) && flag == Py_True;
    Py_DECREF(flag);
    if(!opted_in) return true;

    // Fail fast rather than fall back: a canon that asked for a preview and
    // silently got per-move callbacks would look like it worked and quietly
    // build nothing at all.
    PyObject *consumer = PyObject_GetAttrString(canon, "adopt_geometry");
    bool usable = consumer && PyCallable_Check(consumer);
    Py_XDECREF(consumer);
    if(!usable) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError,
                "parse: canon sets use_gcode_renderer but has no callable "
                "adopt_geometry");
        return false;
    }

    GCodeRenderer *r = new GCodeRenderer(canon);
    r->data_ = new PreviewData();
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 3; j++) {
            r->data_->extents[i][0][j] = 9e99;
            r->data_->extents[i][1][j] = -9e99;
        }
    for(int j = 0; j < 3; j++) {
        r->data_->drawn[0][j] = 9e99;
        r->data_->drawn[1][j] = -9e99;
    }
    r->data_->tool_numbers.push_back(0);         // ordinal 0 is None
    if(!r->read_planes()) {
        delete r;
        return false;
    }
    r->progress_ = PyObject_GetAttrString(canon, "renderer_progress");
    if(r->progress_ && !PyCallable_Check(r->progress_)) {
        Py_CLEAR(r->progress_);
    }
    PyErr_Clear();                      // a canon without one wants no progress

    // The chain point and the leading-traverse flag are the canon's own, not
    // assumed: a canon may be handed to a second parse mid-program. The
    // transform is not read: it starts at zero, and the interpreter re-issues
    // the offsets and the rotation from the parameter file during init(),
    // which runs after this.
    r->sync_in();
    PyObject *div = PyObject_GetAttrString(canon, "arcdivision");
    if(div) {
        long n = PyLong_AsLong(div);
        Py_DECREF(div);
        if(n > 0) r->arcdivision_ = (int)n;
    }
    PyErr_Clear();                      // a canon without one keeps the default
    char name[4];
    for(int i = 0; i < 9; i++) {                // xo, yo, zo, ao .. wo
        snprintf(name, sizeof name, "%co", AXES[i]);
        if(!get_double(canon, name, r->tool_ + i)) break;
    }
    if(PyErr_Occurred()) {
        delete r;
        return false;
    }
    instance_ = r;
    owner_ = canon;
    return true;
}

GCodeRenderer::~GCodeRenderer() {
    delete data_;
    Py_XDECREF(progress_);
}

// `(AXIS,hide)` / `(AXIS,show)`: the two words of the comment vocabulary the
// fill depends on, counted as a depth so nested spans close in order. The rest
// of the vocabulary - `stop`, `notify`, the foam Z levels - is the canon's own
// and reached it through the forward that precedes this call.
void GCodeRenderer::note_comment(const char *text) {
    const char *rest;
    if(!strncmp(text, "AXIS,", 5)) rest = text + 5;
    else if(!strncmp(text, "PREVIEW,", 8)) rest = text + 8;
    else return;
    // The word up to the next comma, as `arg.split(",")[1]` took it.
    size_t n = strcspn(rest, ",");
    if(n != 4) return;
    if(!strncmp(rest, "hide", 4)) suppress_ ++;
    else if(!strncmp(rest, "show", 4)) suppress_ --;
}

void GCodeRenderer::sync_out(bool with_line) {
    if(interp_error) return;
    PyObject *lo = PyTuple_New(9);
    if(!lo) { interp_error ++; return; }
    for(int i = 0; i < 9; i++)
        PyTuple_SET_ITEM(lo, i, PyFloat_FromDouble(lo_[i]));
    int bad = PyObject_SetAttrString(canon_, "lo", lo);
    Py_DECREF(lo);
    bad |= PyObject_SetAttrString(canon_, "first_move",
                                  first_move_ ? Py_True : Py_False);
    if(with_line) {
        PyObject *n = PyLong_FromLong(last_line_);
        if(!n) { interp_error ++; return; }
        bad |= PyObject_SetAttrString(canon_, "lineno", n);
        Py_DECREF(n);
    }
    if(bad) interp_error ++;
}

void GCodeRenderer::sync_in() {
    if(interp_error) return;
    PyObject *lo = PyObject_GetAttrString(canon_, "lo");
    if(!lo) { interp_error ++; return; }
    PyObject *seq = PySequence_Fast(lo, "canon.lo is not a sequence");
    Py_DECREF(lo);
    if(!seq || PySequence_Fast_GET_SIZE(seq) != 9) {
        Py_XDECREF(seq);
        if(!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "canon.lo is not nine numbers");
        interp_error ++;
        return;
    }
    for(int i = 0; i < 9; i++)
        lo_[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    Py_DECREF(seq);
    if(PyErr_Occurred()) { interp_error ++; return; }
    PyObject *fm = PyObject_GetAttrString(canon_, "first_move");
    if(!fm) { interp_error ++; return; }
    first_move_ = PyObject_IsTrue(fm);
    Py_DECREF(fm);
}

void GCodeRenderer::transform(const double *in, double *out) const {
    for(int i = 0; i < 9; i++) out[i] = in[i] + g92_[i];
    if(rotation_xy_ != 0.0) {
        double rotx = out[0] * rotation_cos_ - out[1] * rotation_sin_;
        out[1] = out[0] * rotation_sin_ + out[1] * rotation_cos_;
        out[0] = rotx;
    }
    for(int i = 0; i < 9; i++) out[i] += g5x_[i];
}

// The GEOMETRY-string transform (the C vertex9), for one point through one
// compiled plane.
static void plane_point(const std::vector<GeomOp> &ops, bool respect,
                        const double *pts9, double *out) {
    out[0] = out[1] = out[2] = 0.0;
    for(const GeomOp &op : ops) {
        if(!op.rotate) {
            out[op.a] += pts9[op.col] * op.sign;
            continue;
        }
        double theta = pts9[op.col] * op.sign * (M_PI / 180.0);
        double c = cos(theta), s = sin(theta);
        double a = out[op.a], b = out[op.b];
        if(respect) { a -= op.offa; b -= op.offb; }
        out[op.a] = a * c - b * s;
        out[op.b] = a * s + b * c;
    }
}

void GCodeRenderer::write_vertex(const double *pts9, int line_number,
                                unsigned char kind, double points[2][3]) {
    if(!data_->reserve(1)) {
        if(!interp_error) PyErr_NoMemory();
        interp_error ++;
        return;
    }
    size_t at = data_->n;
    for(int i = 0; i < data_->nplanes; i++) {
        double p[3];
        plane_point(data_->ops[i], data_->respect_offsets, pts9, p);
        for(int j = 0; j < 3; j++) {
            if(p[j] < data_->drawn[0][j]) data_->drawn[0][j] = p[j];
            if(p[j] > data_->drawn[1][j]) data_->drawn[1][j] = p[j];
            data_->pos[i][at * 3 + j] = (float)p[j];
            if(points) points[i][j] = p[j];
        }
    }
    data_->attrs[at * 2] = (uint32_t)line_number;
    data_->attrs[at * 2 + 1] = (uint32_t)kind | (data_->tool << 8);
    data_->n = at + 1;
}

void GCodeRenderer::mark(int line_number, const double *at, unsigned char kind,
                        double points[2][3]) {
    write_vertex(at, line_number, kind, points);
}

// The four machine-frame extent pairs, from one move's raw endpoints.
void GCodeRenderer::accumulate_extents(const double *p1, const double *p2) {
    double box[2][3];
    for(int j = 0; j < 3; j++) {
        box[0][j] = p1[j] < p2[j] ? p1[j] : p2[j];
        box[1][j] = p1[j] > p2[j] ? p1[j] : p2[j];
    }
    // The tool-corrected box is the raw box shifted: adding a constant is
    // monotonic, so this is the same box, not an approximation of it.
    double shift[3] = {tool_[0], tool_[1], tool_[2]};
    double rot[2][3];
    if(rotation_xy_ != 0.0) {
        double u1[3], u2[3];
        unrotate_xy(p1, u1);
        unrotate_xy(p2, u2);
        for(int j = 0; j < 3; j++) {
            rot[0][j] = u1[j] < u2[j] ? u1[j] : u2[j];
            rot[1][j] = u1[j] > u2[j] ? u1[j] : u2[j];
        }
    }
    // raw, notool, zero_rxy, notool_zero_rxy - and with no rotation to remove,
    // the last two are the first two.
    bool rotated = rotation_xy_ != 0.0;
    for(int i = 0; i < 4; i++) {
        bool unrotated = (i >= 2) && rotated;
        const double *lo = unrotated ? rot[0] : box[0];
        const double *hi = unrotated ? rot[1] : box[1];
        bool notool = (i == 1 || i == 3);
        for(int j = 0; j < 3; j++) {
            double a = notool ? lo[j] + shift[j] : lo[j];
            double b = notool ? hi[j] + shift[j] : hi[j];
            if(a < data_->extents[i][0][j]) data_->extents[i][0][j] = a;
            if(b > data_->extents[i][1][j]) data_->extents[i][1][j] = b;
        }
    }
}

// The g5x XY rotation taken back out about the g5x origin, for the
// rotation-removed extents. Z is left alone.
void GCodeRenderer::unrotate_xy(const double *p, double *out) const {
    double tx = p[0] - g5x_[0];
    double ty = p[1] - g5x_[1];
    out[0] = tx * unrot_cos_ - ty * unrot_sin_ + g5x_[0];
    out[1] = tx * unrot_sin_ + ty * unrot_cos_ + g5x_[1];
    out[2] = p[2];
}

void GCodeRenderer::fill(int line_number, const double *p1, const double *p2,
                        double feedrate, unsigned char cat) {
    data_->moves ++;
    accumulate_extents(p1, p2);

    double dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
    double len = sqrt(dx * dx + dy * dy + dz * dz);
    if(cat == CAT_TRAVERSE) data_->rapid_length += len;
    else data_->cut_length[feedrate] += len;

    // A move that does not start where the last one ended gets a record vertex
    // at its start; the shaders discard the segment into it.
    bool jump = !data_->has_cur;
    if(!jump) {
        for(int i = 0; i < 9; i++)
            if(p1[i] != data_->cur9[i]) { jump = true; break; }
    }
    // Rotary subdivision: a move that turns A, B or C is drawn as a polyline,
    // since the tool's path through the machine's frame is not a straight line.
    long steps = 1;
    bool turning = false;
    double dc = 0.0;
    for(int i = 3; i < 6; i++) {
        if(p1[i] != p2[i]) turning = true;
        double d = fabs(p2[i] - p1[i]);
        if(d > dc) dc = d;
    }
    if(turning) {
        double want = dc / 10.0;
        steps = (long)ceil(want > 10.0 ? want : 10.0);
    }
    long count = steps + (jump ? 1 : 0);
    for(long i = 0; i < count; i++) {
        long sub = i - (jump ? 1 : 0) + 1;
        double pt[9];
        if(steps == 1 && !jump) {
            memcpy(pt, p2, 9 * sizeof(double));
        } else {
            double t = (double)sub / (double)steps;
            for(int k = 0; k < 9; k++) pt[k] = t * p2[k] + (1.0 - t) * p1[k];
        }
        write_vertex(pt, line_number, sub == 0 ? KIND_NOOP : cat, nullptr);
    }
    memcpy(data_->cur9, p2, 9 * sizeof(double));
    data_->has_cur = true;
}

void GCodeRenderer::move(Kind kind, int line_number,
                          double x, double y, double z,
                          double a, double b, double c,
                          double u, double v, double w, double rate) {
    last_line_ = line_number;
    consumed_ = true;
    const double in[9] = {x, y, z, a, b, c, u, v, w};
    if(kind >= Dwell) { event(kind, line_number, in); return; }
    // A hidden move touches nothing at all, not even the chain point.
    if(suppress_ > 0) return;

    double p[9];
    transform(in, p);
    if(kind == RigidTap) {
        // Down and back up the way it came, joined to the chain point's
        // rotary and UVW components, and the chain point does not move.
        double end[9];
        end[0] = p[0]; end[1] = p[1]; end[2] = p[2];
        for(int i = 3; i < 9; i++) end[i] = lo_[i];
        first_move_ = false;
        fill(line_number, lo_, end, rate / 60., CAT_FEED);
        fill(line_number, end, lo_, rate / 60., CAT_FEED);
        return;
    }
    if(first_move_) {
        // A leading traverse moves the tool without drawing.
        if(kind == Traverse) { memcpy(lo_, p, sizeof p); return; }
        first_move_ = false;
    }
    if(kind == Traverse) fill(line_number, lo_, p, 0.0, CAT_TRAVERSE);
    else fill(line_number, lo_, p, rate / 60., CAT_FEED);
    memcpy(lo_, p, sizeof p);
}

// CANON_PLANE to the 0/1/2 code a dwell record carries: XY/UV -> 0,
// XZ/UW -> 1, YZ/VW -> 2.
static int plane_code(int plane) {
    switch(plane) {
    case 2: case 5: return 2;
    case 3: case 6: return 1;
    default: return 0;
    }
}

void GCodeRenderer::event(Kind kind, int line_number,
                         const double *axes) {
    switch(kind) {
    case ToolOffset:
        // Not forwarded: it moved only the chain point and the offset triple,
        // and both live here now.
        first_move_ = true;
        for(int i = 0; i < 9; i++) {
            lo_[i] = lo_[i] - axes[i] + tool_[i];
            tool_[i] = axes[i];
        }
        return;
    case Dwell:
    case M1xx: {
        // Both are markers at the current position; a hidden one is dropped,
        // as the canon methods they replace drop it.
        if(suppress_ > 0) return;
        if(kind == Dwell) data_->dwell_time += axes[0];
        DwellRecord rec = {};
        rec.lineno = line_number;
        rec.plane = plane_code(plane_);
        rec.m1xx = (kind == M1xx);
        rec.raw[0] = lo_[0]; rec.raw[1] = lo_[1]; rec.raw[2] = lo_[2];
        mark(line_number, lo_, KIND_DWELL, rec.pts);
        data_->dwells.push_back(rec);
        return;
    }
    case ChangeTool: {
        int tool = (int)axes[0];
        // The record vertex carries the *new* ordinal: it marks where the new
        // tool's work begins. 65535 changes in one program reuse the last
        // ordinal rather than wrap onto another tool's entry.
        if(data_->tool_numbers.size() > 0xFFFF) {
            data_->tool = 0xFFFF;
        } else {
            data_->tool = (uint32_t)data_->tool_numbers.size();
            data_->tool_numbers.push_back(tool);
        }
        ToolChangeRecord rec = {};
        rec.lineno = line_number;
        // The T number as commanded, not the ordinal's entry: past 65535
        // changes the ordinal stops advancing and would hand the record the
        // previous tool's number.
        rec.tool = tool;
        mark(line_number, lo_, KIND_TOOLCHANGE, rec.pts);
        data_->toolchanges.push_back(rec);
        first_move_ = true;
        // Still forwarded, and not for the record: the interpreter reads the
        // canon's tool table for a G43 after this, and a GUI's change_tool
        // override is what moves the simulated spindle slot it reads.
        sync_out(true);
        if(interp_error) return;
        PyObject *result = callmethod(canon_, "change_tool", "i", tool);
        if(!result) { interp_error ++; return; }
        Py_DECREF(result);
        return;
    }
    default:
        PyErr_Format(PyExc_RuntimeError,
                "gcode renderer: unknown event kind %d", (int)kind);
        interp_error ++;
    }
}

void GCodeRenderer::report_progress() {
    if(interp_error) return;
    if(!consumed_) return;
    consumed_ = false;
    if(!progress_) return;
    PyObject *result = PyObject_CallFunction(progress_, "i", last_line_);
    if(!result) { interp_error ++; return; }
    Py_DECREF(result);
}

void GCodeRenderer::hand_over() {
    if(handed_over_) return;
    handed_over_ = true;
    // The parse may be ending *because* something raised - an abort, a syntax
    // error, a canon callback. Put that aside for the handover and put it
    // back: what was rendered before the failure is still a preview.
    PyObject *type, *value, *tb;
    PyErr_Fetch(&type, &value, &tb);
    int errors = interp_error;
    interp_error = 0;
    sync_out(false);
    char name[4];
    for(int i = 0; i < 9; i++) {
        snprintf(name, sizeof name, "%co", AXES[i]);
        PyObject *v = PyFloat_FromDouble(tool_[i]);
        if(!v || PyObject_SetAttrString(canon_, name, v) < 0) interp_error ++;
        Py_XDECREF(v);
    }
    data_->shrink();
    PyObject *pg = preview_geometry_new(data_);
    data_ = nullptr;                    // the Python object owns it now
    if(pg) {
        PyObject *result = callmethod(canon_, "adopt_geometry", "O", pg);
        Py_DECREF(pg);
        Py_XDECREF(result);
    }
    // A failure here loses the geometry, but never the reason the parse ended:
    // the first exception wins, and the parse stays failed either way.
    if(!type && PyErr_Occurred()) PyErr_Fetch(&type, &value, &tb);
    PyErr_Clear();
    if(type) {
        PyErr_Restore(type, value, tb);
        if(!errors) errors = 1;
    }
    interp_error = errors;
}

void GCodeRenderer::render_arc(int line_number, double first_end, double second_end,
                       double first_axis, double second_axis, int rotation,
                       double axis_end_point, double a, double b, double c,
                       double u, double v, double w, double rate) {
    last_line_ = line_number;
    consumed_ = true;
    if(suppress_ > 0) return;
    int steps = arc_segments(lo_, plane_, rotation_cos_, rotation_sin_,
                             g5x_, g92_, first_end, second_end,
                             first_axis, second_axis, rotation,
                             axis_end_point, a, b, c, u, v, w,
                             arcdivision_, segs_);
    // The segments arrive transformed, so no transform here - and an arc is
    // drawn whether or not it is the program's first move, as the per-move
    // canon draws it.
    first_move_ = false;
    for(int i = 0; i < steps; i++) {
        const double *p = &segs_[(size_t)i * 9];
        fill(line_number, lo_, p, rate / 60., CAT_ARC);
        memcpy(lo_, p, 9 * sizeof(double));
    }
}

// ---------------------------------------------------------------------------
// Arc segmentation
// ---------------------------------------------------------------------------
//
// Shared by gcode.arc_to_segments (the canon-driven Python entry point) and
// the renderer, which segments arcs itself rather than asking Python to.

static void unrotate(double &x, double &y, double c, double s) {
    double tx = x * c + y * s;
    y = -x * s + y * c;
    x = tx;
}

static void rotate(double &x, double &y, double c, double s) {
    double tx = x * c - y * s;
    y = x * s + y * c;
    x = tx;
}

int arc_segments(const double lo[9], int plane,
                 double rotation_cos, double rotation_sin,
                 const double g5xoffset[9], const double g92offset[9],
                 double x1, double y1, double cx, double cy, int rot,
                 double z1, double a, double b, double c,
                 double u, double v, double w,
                 int max_segments, std::vector<double> &out) {
    double o[9], n[9];
    int X, Y, Z;
    memcpy(o, lo, 9 * sizeof(double));

    if(plane == 1) {
        X=0; Y=1; Z=2;
    } else if(plane == 3) {
        X=2; Y=0; Z=1;
    } else {
        X=1; Y=2; Z=0;
    }
    n[X] = x1;
    n[Y] = y1;
    n[Z] = z1;
    n[3] = a;
    n[4] = b;
    n[5] = c;
    n[6] = u;
    n[7] = v;
    n[8] = w;
    for(int ax=0; ax<9; ax++) o[ax] -= g5xoffset[ax];
    unrotate(o[0], o[1], rotation_cos, rotation_sin);
    for(int ax=0; ax<9; ax++) o[ax] -= g92offset[ax];

    double theta1 = atan2(o[Y]-cy, o[X]-cx);
    double theta2 = atan2(n[Y]-cy, n[X]-cx);
    /* Issue #1528 1/2/22 andypugh */
    /*_posemath checks for small arcs too, but uses config units */
    double len = hypot(o[X]-n[X], o[Y]-n[Y]) * (25.4 * GET_EXTERNAL_LENGTH_UNITS());
    /* If the signs of the angles differ, make them the same to allow monotonic progress through the arc */
    /* If start and end points are nearly identical, then interpret as a full turn */
    if(rot < 0) { // CW G2
        if (theta1 < theta2) theta2 -= 2*M_PI;
        if (len < CART_FUZZ) theta2 -= 2*M_PI;
    } else { // CCW G3
        if (theta1 > theta2) theta2 += 2*M_PI;
        if (len < CART_FUZZ) theta2 += 2*M_PI;
    }

    // if multi-turn, add the right number of full circles
    if(rot < -1) theta2 += 2*M_PI*(rot+1);
    if(rot > 1) theta2 += 2*M_PI*(rot-1);

    int steps = std::max(3, int(max_segments * fabs(theta1 - theta2) / M_PI));
    double rsteps = 1. / steps;
    out.resize((size_t)steps * 9);

    double dtheta = theta2 - theta1;
    double d[9] = {0, 0, 0, n[3]-o[3], n[4]-o[4], n[5]-o[5], n[6]-o[6], n[7]-o[7], n[8]-o[8]};
    d[Z] = n[Z] - o[Z];

    double tx = o[X] - cx, ty = o[Y] - cy, dc = cos(dtheta*rsteps), ds = sin(dtheta*rsteps);
    for(int i=0; i<steps-1; i++) {
        double f = (i+1) * rsteps;
        double *p = &out[(size_t)i * 9];
        rotate(tx, ty, dc, ds);
        p[X] = tx + cx;
        p[Y] = ty + cy;
        p[Z] = o[Z] + d[Z] * f;
        p[3] = o[3] + d[3] * f;
        p[4] = o[4] + d[4] * f;
        p[5] = o[5] + d[5] * f;
        p[6] = o[6] + d[6] * f;
        p[7] = o[7] + d[7] * f;
        p[8] = o[8] + d[8] * f;
        for(int ax=0; ax<9; ax++) p[ax] += g92offset[ax];
        rotate(p[0], p[1], rotation_cos, rotation_sin);
        for(int ax=0; ax<9; ax++) p[ax] += g5xoffset[ax];
    }
    for(int ax=0; ax<9; ax++) n[ax] += g92offset[ax];
    rotate(n[0], n[1], rotation_cos, rotation_sin);
    for(int ax=0; ax<9; ax++) n[ax] += g5xoffset[ax];
    memcpy(&out[(size_t)(steps-1) * 9], n, 9 * sizeof(double));
    return steps;
}

