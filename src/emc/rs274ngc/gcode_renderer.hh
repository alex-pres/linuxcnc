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

#ifndef GCODE_RENDERER_HH
#define GCODE_RENDERER_HH

#include <Python.h>
#include <pybind11/pybind11.h>

#include <array>
#include <map>
#include <math.h>
#include <stdint.h>
#include <vector>

// State of the parse in flight, defined in gcodemodule.cc. The renderer is a
// separate translation unit but not a separate parse: it delivers to the canon
// `callback` points at, reports failures by bumping `interp_error`, and
// advances `last_sequence_number` for the moves that deliver no next_line.
extern PyObject *callback;
extern int interp_error;
extern int last_sequence_number;

// Run one canon-side step that touches Python. A failure becomes
// `interp_error++` with the Python error left set, which is the protocol
// above; nothing may leave as a C++ exception, because these run from inside
// Interp::execute() and would unwind past state it owns. Every use of
// pybind11 on the canon side goes through this or through forward().
template <typename F>
static inline void canon_guard(F &&body) {
    if(interp_error) return;
    try {
        body();
    } catch(pybind11::error_already_set &e) {
        e.restore();                    // the error stays set, as it was
        interp_error ++;
    } catch(pybind11::builtin_exception &e) {
        e.set_error();                  // py::type_error and friends, by kind
        interp_error ++;
    } catch(const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        interp_error ++;
    }
}

// The common case: call a canon method and discard what it returned.
template <typename... A>
static inline void forward(const char *method, A &&...args) {
    canon_guard([&]{
        pybind11::handle(callback).attr(method)(std::forward<A>(args)...);
    });
}

// ---------------------------------------------------------------------------
// The program the renderer builds: the C++ side of rs274.glcanon_bake's
// ProgramGeometry, filled during the parse and handed to Python at the end of
// it. Layouts match that module's PLANE_DTYPE and ATTR_DTYPE exactly, so the
// arrays reach numpy as views rather than copies.
// ---------------------------------------------------------------------------

// A point in the interpreter's nine axes, in the order the canon functions
// take them: x y z a b c u v w. A value, so the chain point and the offsets
// are copied by assignment rather than by a memcpy whose length is spelled at
// every call site - `sizeof p` was right only while `p` was an array in scope.
// Trivially copyable and 72 bytes, so the generated code is what memcpy gave.
using Point9 = std::array<double, 9>;

// One step of a GEOMETRY string, compiled once per parse. Translate adds an
// axis of the 9-DOF point to an output component; rotate turns a component
// pair by a rotary letter's value (the C vertex9's rotate_x/y/z).
struct GeomOp {
    bool rotate;
    int col;                            // 9-DOF column the step reads
    int a, b;                           // output components (b: rotate only)
    double sign;
    double offa, offb;                  // rotation offsets, when respected
};

struct DwellRecord {
    int lineno;
    int plane;                          // 0/1/2, as GLCanon._record_dwell
    bool m1xx;                          // which colour Python attaches
    double raw[3];                      // machine coords, for canon.dwells
    double pts[2][3];                   // transformed, per drawn plane
};

struct ToolChangeRecord {
    int lineno;
    int tool;                           // as commanded, not the ordinal's entry
    double pts[2][3];
};

struct PreviewData {
    ~PreviewData();
    // False when the arrays could not grow: the caller must stop writing, as
    // the old buffers are still their old size.
    bool reserve(size_t extra);
    void shrink();

    int nplanes = 1;
    std::vector<GeomOp> ops[2];         // one compiled transform per plane
    bool respect_offsets = false;
    float *pos[2] = {nullptr, nullptr}; // 3 floats per vertex, per plane
    uint32_t *attrs = nullptr;          // line, kind|tool per vertex
    size_t n = 0, cap = 0;

    double extents[4][2][3];            // raw, notool, zero_rxy, notool_zero_rxy
    double drawn[2][3];
    // Summed a move at a time, so a running total drifts with move count -
    // about 4e-12 relative over a million moves, nanometres on a metre of tool
    // path. The baked expectations allow for it; no reader of a path length
    // can see it.
    double rapid_length = 0.0;
    std::map<double, double> cut_length; // commanded rate -> cutting length
    size_t moves = 0;

    std::vector<int> tool_numbers;      // entry 0 is the None before any change
    uint32_t tool = 0;
    double dwell_time = 0.0;
    std::vector<DwellRecord> dwells;
    std::vector<ToolChangeRecord> toolchanges;

    Point9 cur9 = {};                   // where the trajectory is
    bool has_cur = false;
};

// The finished program as `gcode.PreviewGeometry`; takes ownership of `data`.
pybind11::object preview_geometry_new(PreviewData *data);
// Register PreviewGeometry and its array views on the module.
void preview_geometry_register(pybind11::module_ &m);

// One arc as up to `max_segments`-ish 9-DOF points, transformed the way a move
// is. Shared by gcode.arc_to_segments and the renderer.
int arc_segments(const Point9 &lo, int plane,
                 double rotation_cos, double rotation_sin,
                 const Point9 &g5xoffset, const Point9 &g92offset,
                 double x1, double y1, double cx, double cy, int rot,
                 double z1, double a, double b, double c,
                 double u, double v, double w,
                 int max_segments, std::vector<Point9> &out);


// ---------------------------------------------------------------------------
// The G-code renderer protocol
// ---------------------------------------------------------------------------
//
// An *opt-in* alternative to the per-event callback protocol, for a canon that
// wants a finished preview rather than a million Python calls. A canon opts in
// by setting `use_gcode_renderer = True` and providing a callable
// `adopt_geometry`; both are read once, in parse_file, before any
// interpretation. Everything below is inert when the flag is absent, and the
// per-event callback sequence is then byte-for-byte what it always was - which
// is what canons that are not previews (rs274.interpret's PrintCanon, the
// interpreter tests, out-of-tree users of gcode.parse) are built on.
//
// The flag must be a *bool*, not merely truthy: a canon that answers every
// unknown attribute with a stub - `def __getattr__(self, name): return lambda
// *a: None`, a common idiom for partial canons, and what
// tests/interp_initcode's does - would otherwise hand back a callable for both
// `use_gcode_renderer` and `adopt_geometry` and be opted in without ever
// asking, silently dropping the whole program into the stub. A canon that sets
// the flag *and* has no callable consumer is a TypeError rather than a silent
// fall back: a preview that quietly came out empty looks like a program with
// nothing in it.
//
// In renderer mode the canon functions listed under `GCodeRenderer::Kind` do
// not call Python at all. Instead the renderer runs the whole preview
// pipeline - the g92 -> XY rotation -> g5x transform, the chain point,
// arc segmentation, the two segments a rigid tap draws, the leading traverses
// `first_move` drops, suppression, rotary subdivision, the GEOMETRY-string
// transform per drawn plane, the extents, the path lengths, and the dwell and
// tool-change records - into a `PreviewData`, and hands the finished program to
// `adopt_geometry` once, at the end of the parse, as a `gcode.PreviewGeometry`.
//
// What still crosses back from Python, and why:
//
//   * `comment` is still forwarded, because the rest of the `(AXIS,...)`
//     vocabulary - `stop`, `notify`, the foam Z levels - is the canon's. The
//     suppression depth those comments used to carry back is read here
//     instead, out of the same text, after the canon has had it.
//   * the g5x/g92 offsets, the XY rotation, the plane and the feed rate are
//     still forwarded, so a canon that watches them still sees them - but as
//     pure observations: the renderer keeps its own copy, taken from the same
//     call, and never reads what Python did with it.
//   * `change_tool` is still forwarded - not for the record, which is written
//     here, but because the interpreter reads the canon's tool table for a G43
//     after it, and a GUI's override is what moves the simulated spindle slot.
//     The canon keeps no list of its own: `GLCanon.adopt_geometry` rebuilds
//     `tool_list` out of the records at the end of the parse.
//     `tool_offset` is *not* forwarded: it moved only geometry state the
//     renderer now owns.
//
// Ordering falls out: a move is rendered where it happens, under exactly the
// offsets, rotation, plane and suppression in force at that point, because
// nothing is held back, and every one of those is captured here as the call
// that changes it goes past. Nothing flows Python to C mid-parse.
//
// A parse therefore starts from a zero transform rather than from whatever the
// canon was holding: the interpreter re-issues the offsets and the rotation
// out of the parameter file during `init()`, which happens after arming, so
// the renderer receives them as canon calls like any others.
//
// Progress is reported through the canon's optional `renderer_progress`: a
// rendered move delivers no `next_line`, so that is what a GUI's progress bar
// counts instead. It fires before each still-forwarded callback, before the
// periodic check_abort(), and at end of parse - which bounds how stale a
// progress bar can get without costing a Python call per move. SET_FEED_RATE
// is the deliberate exception: an F word tells a progress bar nothing, and CAM
// output with adaptive feed emits one every few moves (see feed_rate below).
//
// Lifetime: the program's arrays are owned by the `gcode.PreviewGeometry` the
// handover creates and are never handed out before the parse ends, so no reader
// can hold a view over a buffer that is still growing.
//
// Ownership: like every other piece of canon state in gcodemodule.cc
// (`callback`, `pinterp`, `metric`, `_pos_*`), the renderer is per-process,
// not per-parse -
// there is exactly one, since this translation unit is linked into
// lib/python/gcode.so and nothing else. `interp_from_shlib` swaps the
// *interpreter*, not the canon, so an alternate interpreter still renders here.
// A second parse_file entered while one is in flight (gcode.parse is not
// reentrant, and AXIS's check_abort pumps the Tk event loop) would re-arm the
// renderer for a different canon, and the outer parse's moves would then be
// drawn into the inner parse's program. Hence `owner_`: the canon the renderer
// was armed for, compared - never dereferenced - on every entry point, so a
// mismatch raises instead of misdrawing.

// Hidden visibility, as pybind11 asks of anything holding a py::object: its
// own types are hidden, and a default-visibility class with one as a member
// is an ODR hazard across shared objects (and a -Wattributes warning here).
class __attribute__((visibility("hidden"))) GCodeRenderer {
public:
    // What a canon function reports. Kinds 0-3 are moves; 4-7 are the events
    // between them, carrying their payload in the axis arguments:
    //
    //     dwell (4)        seconds in x
    //     m1xx  (5)        function index, P, Q in x, y, z
    //     change_tool (6)  tool number in x
    //     tool_offset (7)  the nine offsets in x..w
    enum Kind : int {
        Traverse = 0,
        Feed = 1,
        Probe = 2,
        RigidTap = 3,
        Dwell = 4,
        M1xx = 5,
        ChangeTool = 6,
        ToolOffset = 7,
    };

    // Vertex kinds, matching rs274.glcanon_bake. The three drawn ones are also
    // the categories a move carries; the rest are records the shaders discard.
    static constexpr unsigned char CAT_TRAVERSE = 0;
    static constexpr unsigned char CAT_FEED = 1;
    static constexpr unsigned char CAT_ARC = 2;
    static constexpr unsigned char KIND_NOOP = 3;
    static constexpr unsigned char KIND_DWELL = 4;
    static constexpr unsigned char KIND_TOOLCHANGE = 5;

    // Read the opt-in off `canon` and, if it is set, make a renderer ready for
    // it. Returns false with a Python exception set; true means the parse may
    // proceed, in whichever protocol active() now reports.
    static bool arm(PyObject *canon);

    static bool active() { return instance_ != nullptr; }

    // The canon functions' entry points. Defined below the class, which the
    // hot one needs complete; still inline, where the call sites can see it.
    static void append(Kind kind, int line_number,
                       double x, double y, double z,
                       double a, double b, double c,
                       double u, double v, double w);
    static void arc(int line_number, double first_end, double second_end,
                    double first_axis, double second_axis, int rotation,
                    double axis_end_point, double a, double b, double c,
                    double u, double v, double w);
    // Report progress for everything rendered since the last call.
    static void progress();
    // A comment, after the canon has had it: the `(AXIS,hide)`/`(AXIS,show)`
    // depth is the renderer's own. Called from COMMENT only when the forward
    // succeeded, so `(AXIS,stop)` still stops the parse before the word after
    // it could open a hidden span.
    static void comment(const char *text);
    // The transform, from the three canon calls that carry it. Each is still
    // forwarded, so a canon that watches offsets and rotation sees every one;
    // these are called in the same place, after a successful forward, and are
    // where the fill's own copy comes from.
    static void set_g5x(const Point9 &offsets);
    static void set_g92(const Point9 &offsets);
    static void set_rotation_xy(double degrees);
    // End of parse: hand the program over and give the canon back the state the
    // renderer took over, so a reader of canon.lo/first_move/xo..wo sees what
    // it always saw.
    static void finish();
    static void note_plane(int plane);

    // Record the rate the following moves are made at, and answer whether it
    // actually moved. The interpreter reports an F word whether or not it
    // changes anything - interp_execute.cc branches on `block->f_flag` alone,
    // with no comparison against settings->feed_rate - so CAM output that
    // repeats the same `F600` on every line calls in here once per move. In
    // renderer mode there is nothing to say about a rate that did not change:
    // the value already reached the program in every move's own length table.
    // The first call of a parse always counts as a change, so a canon's own
    // starting feed rate is set from the file even when the file opens with the
    // same 60.0 this starts at.
    static bool feed_rate(double rate) {
        if(rate == rate_ && rate_seen_) return false;
        rate_ = rate;
        rate_seen_ = true;
        return true;
    }

    // Tracked whether or not a renderer is armed - one store, and there is no
    // callback to suppress, since SET_SPINDLE_SPEED has never forwarded one.
    // Nothing reads it yet: the program record has no per-move spindle speed,
    // and the G95 (units per revolution) feed mode is what will need one.
    static void spindle_speed(double rpm) { speed_ = rpm; }

private:
    explicit GCodeRenderer(pybind11::handle canon) : canon_(canon) {}
    GCodeRenderer(const GCodeRenderer &) = delete;
    GCodeRenderer &operator=(const GCodeRenderer &) = delete;
    ~GCodeRenderer();

    // Is the renderer still the one armed for the canon being parsed into?
    // Only a re-entered parse_file can make this false; say so rather than draw
    // one canon's moves into another's program.
    static bool owned() {
        if(owner_ == callback) return true;
        PyErr_SetString(PyExc_RuntimeError,
                "gcode.parse: the renderer belongs to a different canon - "
                "gcode.parse was re-entered");
        interp_error ++;
        return false;
    }

    // -- the instance: one parse's pipeline and its program ----------------
    void move(Kind kind, int line_number,
              double x, double y, double z,
              double a, double b, double c,
              double u, double v, double w, double rate);
    void render_arc(int line_number, double first_end, double second_end,
                    double first_axis, double second_axis, int rotation,
                    double axis_end_point, double a, double b, double c,
                    double u, double v, double w, double rate);
    void report_progress();
    void note_comment(const char *text);
    void hand_over();
    void sync_out(bool with_line);
    void sync_in();
    // One move into the geometry: extents, length, then its vertices.
    void fill(int line_number, const Point9 &p1, const Point9 &p2,
              double feedrate, unsigned char cat);
    // One record vertex at `at`, writing its per-plane position to `points`.
    void mark(int line_number, const Point9 &at, unsigned char kind,
              double points[2][3]);
    void write_vertex(const Point9 &pts9, int line_number, unsigned char kind,
                      double points[2][3]);
    void accumulate_extents(const Point9 &p1, const Point9 &p2);
    bool read_planes();
    void unrotate_xy(const Point9 &p, double out[3]) const;
    // g92 -> XY rotation -> g5x, the operations and the order
    // `rs274.interpret.Translated.rotate_and_translate` applies - which is
    // where this came from, though that method no longer runs on a rendered
    // parse. Not bit-identical to it by construction: the compiler is free to
    // contract the rotation's multiply-add, so the tests allow a few ULPs.
    void transform(const Point9 &in, Point9 &out) const;
    void event(Kind kind, int line_number, const Point9 &axes);

    pybind11::handle canon_;            // borrowed, as owner_ is
    pybind11::object progress_;         // canon.renderer_progress, or empty
    PreviewData *data_ = nullptr;       // the program being built, owned
    bool handed_over_ = false;

    // The transform, zero until the interpreter's startup re-issues it.
    Point9 g92_ = {};
    Point9 g5x_ = {};
    double rotation_xy_ = 0.0;
    double rotation_cos_ = 1.0;
    double rotation_sin_ = 0.0;
    double unrot_cos_ = 1.0;            // the same rotation, negated, for the
    double unrot_sin_ = 0.0;            // rotation-removed extents
    Point9 lo_ = {};                    // chain point
    Point9 tool_ = {};                  // xo..wo
    bool first_move_ = true;
    // The `(AXIS,hide)` depth, counted here from the comments themselves.
    // A parse starts at zero: a canon that set `suppress` before the parse
    // was never a supported idiom, and there is no attribute to read now.
    long suppress_ = 0;
    bool respect_offsets_ = false;
    int plane_ = 1;                     // CANON_PLANE, for arc segmentation
    int arcdivision_ = 64;              // the canon's, read once at arm time
    std::vector<Point9> segs_;          // reused by render_arc()

    // Progress is reported once per delivery that consumed anything, moves or
    // not: a hidden stretch still costs parse time.
    int last_line_ = -1;
    bool consumed_ = false;

    // The renderer for the parse in flight, or null when no canon asked for
    // one. arm() owns it.
    static inline GCodeRenderer *instance_ = nullptr;
    static inline PyObject *owner_ = nullptr;   // compared, never dereferenced
    static inline double rate_ = 60.0;
    static inline bool rate_seen_ = false;
    static inline double speed_ = 0.0;
};

inline void GCodeRenderer::append(Kind kind, int line_number,
                                  double x, double y, double z,
                                  double a, double b, double c,
                                  double u, double v, double w) {
    if(!owned()) return;
    // No next_line is delivered for a rendered move, but the error line the
    // parse reports must still advance with it.
    last_sequence_number = line_number;
    instance_->move(kind, line_number, x, y, z, a, b, c, u, v, w, rate_);
}

#endif  // GCODE_RENDERER_HH
