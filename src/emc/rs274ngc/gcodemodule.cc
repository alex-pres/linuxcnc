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

/*

  Notes:

  NURBS
  -----
  The code in this file for nurbs calculations is from University of Palermo.
  The publications can be found at: http://wiki.linuxcnc.org/cgi-bin/wiki.pl?NURBS
  AMST08_art837759.pdf and ECME14.pdf

  1: 
  M. Leto, R. Licari, E. Lo Valvo1 , M. Piacentini:
  CAD/CAM INTEGRATION FOR NURBS PATH INTERPOLATION ON PC BASED REAL-TIME NUMERICAL CONTROL
  Proceedings of AMST 2008 Conference, 2008, pp. 223-233

  2:
  ERNESTO LO VALVO, STEFANO DRAGO:
  An Efficient NURBS Path Generator for a Open Source CNC
  Recent Advances in Mechanical Engineering (pp.173-180). WSEAS Press

  The code from University of Palermo is modified to work on planes xy, yz and zx by Joachim Franek
  */


#include <Python.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <memory>

#include "rs274ngc.hh"
#include "rs274ngc_interp.hh"
#include "nml_intf/interp_return.hh"
#include "nml_intf/canon.hh"

#include "gcode_renderer.hh"

int _task = 0; // control preview behaviour when remapping

char _parameter_file_name[LINELEN];

extern "C" PyObject* PyInit_interpreter(void);
extern "C" PyObject* PyInit_emccanon(void);
extern "C" struct _inittab builtin_modules[];
struct _inittab builtin_modules[] = {
    { "interpreter", PyInit_interpreter },
    { "emccanon", PyInit_emccanon },
    { NULL, NULL }
};


namespace py = pybind11;

// What `next_line` hands the canon: the interpreter's active settings and its
// active G and M codes for one source line. Plain data - the read-only
// properties below are its whole Python surface, and `sequence_number`
// overlays gcodes[0] exactly as the old PyMemberDef offsets did.
struct LineCode {
    double settings[ACTIVE_SETTINGS];
    int gcodes[ACTIVE_G_CODES];
    int mcodes[ACTIVE_M_CODES];
};

static py::tuple int_array(const int *arr, int sz) {
    py::tuple res(sz);
    for(int i = 0; i < sz; i++) res[i] = arr[i];
    return res;
}

static void linecode_register(py::module_ &m) {
    py::class_<LineCode> c(m, "linecode");
#define LC(name, slot) \
    c.def_property_readonly(name, [](const LineCode &l) { return l.slot; })
    LC("sequence_number", gcodes[0]);

    LC("feed_rate", settings[1]);
    LC("speed", settings[2]);
    LC("motion_mode", gcodes[1]);
    LC("block", gcodes[2]);
    LC("plane", gcodes[3]);
    LC("cutter_side", gcodes[4]);
    LC("units", gcodes[5]);
    LC("distance_mode", gcodes[6]);
    LC("feed_mode", gcodes[7]);
    LC("origin", gcodes[8]);
    LC("tool_length_offset", gcodes[9]);
    LC("retract_mode", gcodes[10]);
    LC("path_mode", gcodes[11]);

    LC("stopping", mcodes[1]);
    LC("spindle", mcodes[2]);
    LC("toolchange", mcodes[3]);
    LC("mist", mcodes[4]);
    LC("flood", mcodes[5]);
    LC("overrides", mcodes[6]);
#undef LC
    c.def_property_readonly("gcodes", [](const LineCode &l) {
            return int_array(l.gcodes, ACTIVE_G_CODES); });
    c.def_property_readonly("mcodes", [](const LineCode &l) {
            return int_array(l.mcodes, ACTIVE_M_CODES); });
}

PyObject *callback;
int interp_error;
int last_sequence_number;
static int selected_tool = 0;
static bool metric;
static double _pos_x, _pos_y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w;
EmcPose tool_offset;

static InterpBase *pinterp;

// The `next_line` delivery guard, split off from last_sequence_number: a
// rendered move advances that one (parse_file returns it, and error reporting
// reads it) without a next_line having been delivered, so a still-forwarded
// callback later on the same line must not be mistaken for a repeat. The two
// move in lockstep on the callback protocol, where nothing but delivery
// touches either.
static int last_delivered_sequence_number;

static void maybe_new_line(int sequence_number);
static void maybe_new_line();

// The line number for a rendered event whose canon function is not given one.
static int renderer_line_number() {
    return pinterp ? pinterp->sequence_number() : last_sequence_number;
}

// Deliver next_line to the canon, at most once per source line.
static void deliver_new_line(int sequence_number) {
    if(!pinterp) return;
    if(interp_error) return;
    if(sequence_number == last_delivered_sequence_number)
        return;
    auto line = std::make_unique<LineCode>();
    pinterp->active_settings(line->settings);
    pinterp->active_g_codes(line->gcodes);
    pinterp->active_m_codes(line->mcodes);
    line->gcodes[0] = sequence_number;
    last_sequence_number = sequence_number;
    last_delivered_sequence_number = sequence_number;
    // The cast is inside the guard too: it is the one step here that can raise.
    canon_guard([&]{
        py::handle(callback).attr("next_line")(py::cast(std::move(line)));
    });
}

static void maybe_new_line(int sequence_number) {
    // Every canon function that forwards a Python callback calls
    // maybe_new_line() first, so this is where the renderer's progress report
    // lands: once per source line that produced anything, which is what a GUI
    // counts now that a rendered move delivers no next_line. In renderer mode
    // very little still forwards, so the periodic report in the parse loop is
    // what actually bounds how stale a progress bar gets.
    GCodeRenderer::progress();
    deliver_new_line(sequence_number);
}

static void maybe_new_line() {
    if(!pinterp) return;
    maybe_new_line(pinterp->sequence_number());
}

//das ist für die Vorschau
/* G_5_2/G_5_3*/
void NURBS_G5_FEED(int line_number, const std::vector<NURBS_CONTROL_POINT>& nurbs_control_points, unsigned int nurbs_order, CANON_PLANE plane)
    {
    double u = 0.0;
    unsigned int n = nurbs_control_points.size() - 1;
    double umax = n - nurbs_order + 2;
    unsigned int div = nurbs_control_points.size()*15;
    std::vector<unsigned int> knot_vector = nurbs_G5_knot_vector_creator(n, nurbs_order);	
    NURBS_PLANE_POINT P1;
    while (u+umax/div < umax) {
        NURBS_PLANE_POINT P1 = nurbs_G5_point(u+umax/div,nurbs_order,nurbs_control_points,knot_vector);
        //printf("P1 X: %8.4f Y: %8.4f pos_x: %8.4f pos_y: %8.4f pos_z: %8.4f (F: %s L: %d)\n",P1.NURBS_X,P1.NURBS_Y,_pos_x,_pos_y,_pos_z,__FILE__,__LINE__);

        //STRAIGHT_FEED(line_number, P1.X,P1.Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
        if(plane==CANON_PLANE::XY) {
            //printf("XY (F: %s L: %d)\n",__FILE__,__LINE__);
            STRAIGHT_FEED(line_number, P1.NURBS_X, P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
            }
        if(plane==CANON_PLANE::YZ) {
            //printf("YZ (F: %s L: %d)\n",__FILE__,__LINE__);
            STRAIGHT_FEED(line_number, _pos_x, P1.NURBS_X, P1.NURBS_Y, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
            }
        if(plane==CANON_PLANE::XZ) {
            //printf("XZ (F: %s L: %d)\n",__FILE__,__LINE__);
            STRAIGHT_FEED(line_number, P1.NURBS_Y, _pos_y, P1.NURBS_X, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
            }
        u = u + umax/div;
        } 
    P1.NURBS_X = nurbs_control_points[n].NURBS_X;
    P1.NURBS_Y = nurbs_control_points[n].NURBS_Y;
    //printf("Pn X: %8.4f Y: %8.4f pos_x: %8.4f pos_y: %8.4f pos_z: %8.4f (F: %s L: %d)\n",P1.X,P1.Y,_pos_x,_pos_y,_pos_z,__FILE__,__LINE__);
    //STRAIGHT_FEED(line_number, P1.X,P1.Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    if(plane==CANON_PLANE::XY) {
        STRAIGHT_FEED(line_number, P1.NURBS_X, P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
        }
    if(plane==CANON_PLANE::YZ) {
        STRAIGHT_FEED(line_number, _pos_x, P1.NURBS_X, P1.NURBS_Y, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
        }
    if(plane==CANON_PLANE::XZ) {
        STRAIGHT_FEED(line_number, P1.NURBS_Y, _pos_y, P1.NURBS_X, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w); //
        }
    knot_vector.clear();
}

/* G_6_2  L_option is unused */
//-----------------------------------------------------------------------------------------------------------------------------------------
void NURBS_G6_FEED(int line_number, const std::vector<NURBS_G6_CONTROL_POINT>& nurbs_control_points, unsigned int k, double /*feedrate*/, int /*L_option*/, CANON_PLANE plane) { // (L_option: NICU, NICL, NICC see publication from Lo Valvo and Drago)
    double u = 0.0;
    unsigned int n = nurbs_control_points.size() - 1-k;
    double umax = nurbs_control_points[n+k].NURBS_K;
    unsigned int div = (nurbs_control_points.size()-k)*15;
    std::vector<double> knot_vector = nurbs_g6_knot_vector_creator(n, k, nurbs_control_points);

    //printf("gcodemodule NURBS_G6_FEED cps: %ld k: %d L: %d fr: %f (F: %s L: %d)\n",nurbs_control_points.size(), k, L_option, feedrate, __FILE__, __LINE__);
    NURBS_PLANE_POINT P1x, P1;
    std::vector< std::vector<double> > A6;
    A6 = nurbs_G6_Nmix_creator(u+umax/div, k, n+1, knot_vector);
    P1 = nurbs_G6_pointx(knot_vector[0],k,nurbs_control_points,knot_vector,A6);	
    //printf("%.3d P1  X: %8.4f Y: %8.4f pos_x: %8.4f pos_y: %8.4f pos_z: %8.4f (F: %s L: %d)\n",line_number,P1.NURBS_X,P1.NURBS_Y,_pos_x,_pos_y,_pos_z,__FILE__,__LINE__);
    //STRAIGHT_FEED(line_number, P1.NURBS_X,P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    if(plane==CANON_PLANE::XY) {
		    STRAIGHT_FEED(line_number, P1.NURBS_X, P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
        }
    if(plane==CANON_PLANE::YZ) {
		    STRAIGHT_FEED(line_number, _pos_x, P1.NURBS_X, P1.NURBS_Y, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
        }
    if(plane==CANON_PLANE::XZ) {
		    STRAIGHT_FEED(line_number, P1.NURBS_Y, _pos_y, P1.NURBS_X, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
        }
    u=0.1;
    while (u+umax/div < umax) {
        P1x = nurbs_G6_point_x(u+umax/div,k,nurbs_control_points,knot_vector);
        //printf("%.3d P1x X: %8.4f Y: %8.4f pos_x: %8.4f pos_y: %8.4f pos_z: %8.4f (F: %s L: %d)\n",line_number,P1x.NURBS_X,P1x.NURBS_Y,_pos_x,_pos_y,_pos_z,__FILE__,__LINE__);
        //STRAIGHT_FEED(line_number, P1x.NURBS_X,P1x.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
		if(plane==CANON_PLANE::XY) {
			    STRAIGHT_FEED(line_number, P1x.NURBS_X, P1x.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
			}
		if(plane==CANON_PLANE::YZ) {
			STRAIGHT_FEED(line_number, _pos_x, P1x.NURBS_X, P1x.NURBS_Y, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
			}
		if(plane==CANON_PLANE::XZ) {
			STRAIGHT_FEED(line_number, P1x.NURBS_Y, _pos_y, P1x.NURBS_X, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
			}
		u = u + umax/div;
    } 
    A6 = nurbs_G6_Nmix_creator (umax,  k, n+1, knot_vector);
    P1 = nurbs_G6_pointx(umax,k,nurbs_control_points,knot_vector,A6);	
    //printf("%.3d P1  X: %8.4f Y: %8.4f pos_x: %8.4f pos_y: %8.4f pos_z: %8.4f (F: %s L: %d)\n",line_number,P1.NURBS_X,P1.NURBS_Y,_pos_x,_pos_y,_pos_z,__FILE__,__LINE__);
    //STRAIGHT_FEED(line_number, P1.NURBS_X,P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    if(plane==CANON_PLANE::XY) {
        STRAIGHT_FEED(line_number, P1.NURBS_X, P1.NURBS_Y, _pos_z, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    	}
    if(plane==CANON_PLANE::YZ) {
		STRAIGHT_FEED(line_number, _pos_x, P1.NURBS_X, P1.NURBS_Y, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    	}
    if(plane==CANON_PLANE::XZ) {
		STRAIGHT_FEED(line_number, P1.NURBS_Y, _pos_y, P1.NURBS_X, _pos_a, _pos_b, _pos_c, _pos_u, _pos_v, _pos_w);
    	}
    knot_vector.clear();
	}

//-----------------------------------------------------------------------------------------------------------------------------------------
void ARC_FEED(int line_number,
              double first_end, double second_end, double first_axis,
              double second_axis, int rotation, double axis_end_point,
              double a_position, double b_position, double c_position,
              double u_position, double v_position, double w_position) {
    // XXX: set _pos_*
    if(metric) {
        first_end /= 25.4;
        second_end /= 25.4;
        first_axis /= 25.4;
        second_axis /= 25.4;
        axis_end_point /= 25.4;
        u_position /= 25.4;
        v_position /= 25.4;
        w_position /= 25.4;
    }
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        GCodeRenderer::arc(line_number, first_end, second_end, first_axis,
                           second_axis, rotation, axis_end_point,
                           a_position, b_position, c_position,
                           u_position, v_position, w_position);
        return;
    }
    maybe_new_line(line_number);
    forward("arc_feed", first_end, second_end, first_axis, second_axis,
            rotation, axis_end_point,
            a_position, b_position, c_position,
            u_position, v_position, w_position);
}

void STRAIGHT_FEED(int line_number,
                   double x, double y, double z,
                   double a, double b, double c,
                   double u, double v, double w) {
    _pos_x=x; _pos_y=y; _pos_z=z; 
    _pos_a=a; _pos_b=b; _pos_c=c;
    _pos_u=u; _pos_v=v; _pos_w=w;
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; u /= 25.4; v /= 25.4; w /= 25.4; }
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        GCodeRenderer::append(GCodeRenderer::Feed, line_number,
                              x, y, z, a, b, c, u, v, w);
        return;
    }
    maybe_new_line(line_number);
    forward("straight_feed", x, y, z, a, b, c, u, v, w);
}

void STRAIGHT_TRAVERSE(int line_number,
                       double x, double y, double z,
                       double a, double b, double c,
                       double u, double v, double w) {
    _pos_x=x; _pos_y=y; _pos_z=z; 
    _pos_a=a; _pos_b=b; _pos_c=c;
    _pos_u=u; _pos_v=v; _pos_w=w;
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; u /= 25.4; v /= 25.4; w /= 25.4; }
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        GCodeRenderer::append(GCodeRenderer::Traverse, line_number,
                              x, y, z, a, b, c, u, v, w);
        return;
    }
    maybe_new_line(line_number);
    forward("straight_traverse", x, y, z, a, b, c, u, v, w);
}

void SET_G5X_OFFSET(int g5x_index,
                    double x, double y, double z,
                    double a, double b, double c,
                    double u, double v, double w) {
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; u /= 25.4; v /= 25.4; w /= 25.4; }
    const Point9 offsets = {x, y, z, a, b, c, u, v, w};
    // The renderer transforms with its own copy, taken here, and never reads
    // the canon's - so in renderer mode there is nothing to tell the canon.
    if(GCodeRenderer::active()) { GCodeRenderer::set_g5x(offsets); return; }
    maybe_new_line();
    forward("set_g5x_offset", g5x_index, x, y, z, a, b, c, u, v, w);
}

void SET_G92_OFFSET(double x, double y, double z,
                    double a, double b, double c,
                    double u, double v, double w) {
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; u /= 25.4; v /= 25.4; w /= 25.4; }
    const Point9 offsets = {x, y, z, a, b, c, u, v, w};
    if(GCodeRenderer::active()) { GCodeRenderer::set_g92(offsets); return; }
    maybe_new_line();
    forward("set_g92_offset", x, y, z, a, b, c, u, v, w);
}

void SET_XY_ROTATION(double t) {
    if(GCodeRenderer::active()) { GCodeRenderer::set_rotation_xy(t); return; }
    maybe_new_line();
    forward("set_xy_rotation", t);
};

void USE_LENGTH_UNITS(CANON_UNITS u) { metric = u == CANON_UNITS_MM; }

void SELECT_PLANE(CANON_PLANE pl) {
    GCodeRenderer::note_plane(static_cast<int>(pl));
    // The plane reaches the record through note_plane and the arc segmenter;
    // nothing on a rendered parse reads the canon's own copy.
    if(GCodeRenderer::active()) return;
    maybe_new_line();
    forward("set_plane", static_cast<int>(pl));
}

void SET_TRAVERSE_RATE(double rate) {
    // A traverse carries no rate into the record - rapid_length is a length,
    // not a time - so a rendered parse has nothing to say about this.
    if(GCodeRenderer::active()) return;
    maybe_new_line();
    forward("set_traverse_rate", rate);
}

void SET_FEED_MODE(int /*spindle*/, int /*mode*/) {
#if 0
    maybe_new_line();
    forward("set_feed_mode", mode);
#endif
}

void CHANGE_TOOL() {
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        GCodeRenderer::append(GCodeRenderer::ChangeTool, renderer_line_number(),
                     selected_tool, 0, 0, 0, 0, 0, 0, 0, 0);
        return;
    }
    maybe_new_line();
    forward("change_tool", selected_tool);
}

void CHANGE_TOOL_NUMBER(int /*pocket*/) {
    maybe_new_line();
    if(interp_error) return;
}

void RELOAD_TOOLDATA(void) {
    return;
}

/* XXX: This needs to be re-thought.  Sometimes feed rate is not in linear
 * units--e.g., it could be inverse time feed mode.  in that case, it's wrong
 * to convert from mm to inch here.  but the gcode time estimate gets inverse
 * time feed wrong anyway..
 */
void SET_FEED_RATE(double rate) {
    if(metric) rate /= 25.4;
    if(GCodeRenderer::active()) {
        // The rate reaches the program in every move's own length table, so
        // there is nothing left for the canon to be told - and telling it is
        // not cheap: the interpreter reports an F word whether or not it
        // changed anything (interp_execute.cc branches on `block->f_flag`
        // alone), so CAM output with adaptive feed lands here once per move.
        //
        // Arcs are unaffected: the renderer segments an arc with the rate
        // current at the time, which is this one.
        GCodeRenderer::feed_rate(rate);
        return;
    }
    maybe_new_line();
    forward("set_feed_rate", rate);
}

void DWELL(double time) {
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        // Rendered like any other event, and not a progress report of its
        // own: a G81/G82 cycle emits one of these per hole.
        GCodeRenderer::append(GCodeRenderer::Dwell, renderer_line_number(),
                     time, 0, 0, 0, 0, 0, 0, 0, 0);
        return;
    }
    maybe_new_line();
    forward("dwell", time);
}

void MESSAGE(char *comment) {
    maybe_new_line();
    forward("message", comment);
}

void LOG(char * /*s*/) {}
void LOGOPEN(char * /*f*/) {}
void LOGAPPEND(char * /*f*/) {}
void LOGCLOSE() {}

void COMMENT(const char *comment) {
    maybe_new_line();
    forward("comment", comment);
    if(interp_error) return;
    GCodeRenderer::comment(comment);
}

void SET_TOOL_TABLE_ENTRY(int /*pocket*/, int /*toolno*/, const EmcPose& /*offset*/, double /*diameter*/,
                          double /*frontangle*/, double /*backangle*/, int /*orientation*/) {
}

void USE_TOOL_LENGTH_OFFSET(const EmcPose& offset) {
    tool_offset = offset;
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        if(metric) {
            GCodeRenderer::append(GCodeRenderer::ToolOffset, renderer_line_number(),
                    offset.tran.x / 25.4, offset.tran.y / 25.4,
                    offset.tran.z / 25.4,
                    offset.a, offset.b, offset.c,
                    offset.u / 25.4, offset.v / 25.4, offset.w / 25.4);
        } else {
            GCodeRenderer::append(GCodeRenderer::ToolOffset, renderer_line_number(),
                    offset.tran.x, offset.tran.y, offset.tran.z,
                    offset.a, offset.b, offset.c,
                    offset.u, offset.v, offset.w);
        }
        return;
    }
    maybe_new_line();
    const double scale = metric ? 25.4 : 1.0;
    forward("tool_offset",
            offset.tran.x / scale, offset.tran.y / scale, offset.tran.z / scale,
            offset.a, offset.b, offset.c,
            offset.u / scale, offset.v / scale, offset.w / scale);
}

void SET_FEED_REFERENCE(double /*reference*/) { }
void SET_CUTTER_RADIUS_COMPENSATION(double /*radius*/) {}
void START_CUTTER_RADIUS_COMPENSATION(int /*direction*/) {}
void STOP_CUTTER_RADIUS_COMPENSATION(int /*direction*/) {}
void START_SPEED_FEED_SYNCH() {}
void START_SPEED_FEED_SYNCH(int /*spindle*/, double /*sync*/, bool /*vel*/) {}
void STOP_SPEED_FEED_SYNCH() {}
void START_SPINDLE_COUNTERCLOCKWISE(int /*spindle*/, int /*wait_for_at_speed*/) {}
void START_SPINDLE_CLOCKWISE(int /*spindle*/, int /*wait_for_at_speed*/) {}
void SET_SPINDLE_MODE(int /*spindle*/, double) {}
void STOP_SPINDLE_TURNING(int /*spindle*/, int /*wait_for_at_speed*/) {}
void SET_SPINDLE_SPEED(int spindle, double rpm) {
    if(spindle == 0) GCodeRenderer::spindle_speed(rpm);
}
void ORIENT_SPINDLE(int /*spindle*/, double /*d*/, int /*i*/) {}
void WAIT_SPINDLE_ORIENT_COMPLETE(int /*s*/, double /*timeout*/) {}
void PROGRAM_STOP() {}
void PROGRAM_END() {}
void FINISH() {}
void ON_RESET() {}
void PALLET_SHUTTLE() {}
void SELECT_TOOL(int tool) {selected_tool = tool;}
void UPDATE_TAG(const StateTag& /*tag*/) {}
void OPTIONAL_PROGRAM_STOP() {}
int  GET_EXTERNAL_TC_FAULT() {return 0;}
int  GET_EXTERNAL_TC_REASON() {return 0;}


extern bool GET_BLOCK_DELETE(void) {
    int bd = 0;
    canon_guard([&]{
        // PyObject_IsTrue, not a bool cast: any object the canon returns
        // answers this, as it always has.
        bd = PyObject_IsTrue(py::handle(callback).attr("get_block_delete")().ptr());
    });
    return bd;
}

void CANON_ERROR(const char * /*fmt*/, ...) {};
void CLAMP_AXIS(CANON_AXIS /*axis*/) {}
bool GET_OPTIONAL_PROGRAM_STOP() { return false;}
void SET_OPTIONAL_PROGRAM_STOP(bool /*state*/) {}
void SPINDLE_RETRACT_TRAVERSE() {}
void SPINDLE_RETRACT() {}
void STOP_CUTTER_RADIUS_COMPENSATION() {}
void USE_NO_SPINDLE_FORCE() {}
void SET_BLOCK_DELETE(bool /*enabled*/) {}

void DISABLE_FEED_OVERRIDE() {}
void DISABLE_FEED_HOLD() {}
void ENABLE_FEED_HOLD() {}
void DISABLE_SPEED_OVERRIDE(int /*spindle*/) {}
void ENABLE_FEED_OVERRIDE() {}
void ENABLE_SPEED_OVERRIDE(int /*spindle*/) {}
void MIST_OFF() {}
void FLOOD_OFF() {}
void MIST_ON() {}
void FLOOD_ON() {}
void CLEAR_AUX_OUTPUT_BIT(int /*bit*/) {}
void SET_AUX_OUTPUT_BIT(int /*bit*/) {}
void SET_AUX_OUTPUT_VALUE(int /*index*/, double /*value*/) {}
void CLEAR_MOTION_OUTPUT_BIT(int /*bit*/) {}
void SET_MOTION_OUTPUT_BIT(int /*bit*/) {}
void SET_MOTION_OUTPUT_VALUE(int /*index*/, double /*value*/) {}
void TURN_PROBE_ON() {}
void TURN_PROBE_OFF() {}
int UNLOCK_ROTARY(int /*line_no*/, int /*joint_num*/) {return 0;}
int LOCK_ROTARY(int /*line_no*/, int /*joint_num*/) {return 0;}
void INTERP_ABORT(int /*reason*/, const char * /*message*/) {}

void STRAIGHT_PROBE(int line_number, 
                    double x, double y, double z, 
                    double a, double b, double c,
                    double u, double v, double w, unsigned char /*probe_type*/) {
    _pos_x=x; _pos_y=y; _pos_z=z; 
    _pos_a=a; _pos_b=b; _pos_c=c;
    _pos_u=u; _pos_v=v; _pos_w=w;
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; u /= 25.4; v /= 25.4; w /= 25.4; }
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        GCodeRenderer::append(GCodeRenderer::Probe, line_number,
                              x, y, z, a, b, c, u, v, w);
        return;
    }
    maybe_new_line(line_number);
    forward("straight_probe", x, y, z, a, b, c, u, v, w);

}
void RIGID_TAP(int line_number,
               double x, double y, double z, double /*scale*/) {
    if(metric) { x /= 25.4; y /= 25.4; z /= 25.4; }
    if(GCodeRenderer::active()) {
        if(interp_error) return;
        // a..w are zero, exactly the arguments the `rigid_tap` callback does
        // not have; the renderer joins x,y,z to the chain point's a..w.
        GCodeRenderer::append(GCodeRenderer::RigidTap, line_number,
                              x, y, z, 0, 0, 0, 0, 0, 0);
        return;
    }
    maybe_new_line(line_number);
    forward("rigid_tap", x, y, z);
}
double GET_EXTERNAL_MOTION_CONTROL_TOLERANCE() { return 0.1; }
double GET_EXTERNAL_MOTION_CONTROL_NAIVECAM_TOLERANCE() { return 0.1; }
double GET_EXTERNAL_PROBE_POSITION_X() { return _pos_x; }
double GET_EXTERNAL_PROBE_POSITION_Y() { return _pos_y; }
double GET_EXTERNAL_PROBE_POSITION_Z() { return _pos_z; }
double GET_EXTERNAL_PROBE_POSITION_A() { return _pos_a; }
double GET_EXTERNAL_PROBE_POSITION_B() { return _pos_b; }
double GET_EXTERNAL_PROBE_POSITION_C() { return _pos_c; }
double GET_EXTERNAL_PROBE_POSITION_U() { return _pos_u; }
double GET_EXTERNAL_PROBE_POSITION_V() { return _pos_v; }
double GET_EXTERNAL_PROBE_POSITION_W() { return _pos_w; }
double GET_EXTERNAL_PROBE_VALUE() { return 0.0; }
int GET_EXTERNAL_PROBE_TRIPPED_VALUE() { return 0; }
double GET_EXTERNAL_POSITION_X() { return _pos_x; }
double GET_EXTERNAL_POSITION_Y() { return _pos_y; }
double GET_EXTERNAL_POSITION_Z() { return _pos_z; }
double GET_EXTERNAL_POSITION_A() { return _pos_a; }
double GET_EXTERNAL_POSITION_B() { return _pos_b; }
double GET_EXTERNAL_POSITION_C() { return _pos_c; }
double GET_EXTERNAL_POSITION_U() { return _pos_u; }
double GET_EXTERNAL_POSITION_V() { return _pos_v; }
double GET_EXTERNAL_POSITION_W() { return _pos_w; }
void INIT_CANON() {}

void SET_PARAMETER_FILE_NAME(const char *name)
{
  strncpy(_parameter_file_name, name, PARAMETER_FILE_NAME_LENGTH);
}

void GET_EXTERNAL_PARAMETER_FILE_NAME(char *name, int max_size) {
    name[0] = 0;
    // Not on the interp_error protocol: a canon without the attribute has
    // always just left the name empty, with the error still on the indicator.
    try {
        std::string file = py::cast<std::string>(
                py::handle(callback).attr("parameter_file"));
        memset(name, 0, max_size);
        strncpy(name, file.c_str(), max_size - 1);
    } catch(py::error_already_set &e) {
        e.restore();
    }
}
CANON_UNITS GET_EXTERNAL_LENGTH_UNIT_TYPE() { return CANON_UNITS_INCHES; }
CANON_TOOL_TABLE GET_EXTERNAL_TOOL_TABLE(int pocket) {
    CANON_TOOL_TABLE tdata = {-1,-1,{{0,0,0},0,0,0,0,0,0},0,0,0,0,{}};
    canon_guard([&]{
        py::object result = py::handle(callback).attr("get_tool")(pocket);
        if(!PyTuple_Check(result.ptr()) || PyTuple_GET_SIZE(result.ptr()) != 14)
            throw py::type_error("get_tool: expected a tuple of 14 items");
        py::tuple t = py::reinterpret_borrow<py::tuple>(result);
        // Filled only once every field parsed, so a bad entry leaves the
        // caller the same all-defaults table PyArg_ParseTuple used to.
        CANON_TOOL_TABLE got = tdata;
        got.toolno         = py::cast<int>(t[0]);
        got.offset.tran.x  = py::cast<double>(t[1]);
        got.offset.tran.y  = py::cast<double>(t[2]);
        got.offset.tran.z  = py::cast<double>(t[3]);
        got.offset.a       = py::cast<double>(t[4]);
        got.offset.b       = py::cast<double>(t[5]);
        got.offset.c       = py::cast<double>(t[6]);
        got.offset.u       = py::cast<double>(t[7]);
        got.offset.v       = py::cast<double>(t[8]);
        got.offset.w       = py::cast<double>(t[9]);
        got.diameter       = py::cast<double>(t[10]);
        got.frontangle     = py::cast<double>(t[11]);
        got.backangle      = py::cast<double>(t[12]);
        got.orientation    = py::cast<int>(t[13]);
        tdata = got;
    });
    return tdata;
}

int GET_EXTERNAL_DIGITAL_INPUT(int /*index*/, int def) { return def; }
double GET_EXTERNAL_ANALOG_INPUT(int /*index*/, double def) { return def; }
int WAIT(int /*index*/, int /*input_type*/, int /*wait_type*/, double /*timeout*/) { return 0;}

static void user_defined_function(int num, double arg1, double arg2) {
    if(interp_error) return;
    if(GCodeRenderer::active()) {
        GCodeRenderer::append(GCodeRenderer::M1xx, renderer_line_number(),
                     num, arg1, arg2, 0, 0, 0, 0, 0, 0);
        return;
    }
    maybe_new_line();
    forward("user_defined_function", num, arg1, arg2);
}

void SET_FEED_REFERENCE(CANON_FEED_REFERENCE /*ref*/) {}
int GET_EXTERNAL_QUEUE_EMPTY() { return true; }
CANON_DIRECTION GET_EXTERNAL_SPINDLE(int) { return CANON_STOPPED; }
int GET_EXTERNAL_TOOL_SLOT() { return 0; }
int GET_EXTERNAL_SELECTED_TOOL_SLOT() { return 0; }
double GET_EXTERNAL_FEED_RATE() { return 1; }
double GET_EXTERNAL_TRAVERSE_RATE() { return 0; }
int GET_EXTERNAL_FLOOD() { return 0; }
int GET_EXTERNAL_MIST() { return 0; }
CANON_PLANE GET_EXTERNAL_PLANE() { return CANON_PLANE::XY; }
double GET_EXTERNAL_SPEED(int /*spindle*/) { return 0; }
void DISABLE_ADAPTIVE_FEED() {} 
void ENABLE_ADAPTIVE_FEED() {} 

int GET_EXTERNAL_FEED_OVERRIDE_ENABLE() {return 1;}
int GET_EXTERNAL_SPINDLE_OVERRIDE_ENABLE(int /*spindle*/) {return 1;}
int GET_EXTERNAL_ADAPTIVE_FEED_ENABLE() {return 0;}
int GET_EXTERNAL_FEED_HOLD_ENABLE() {return 1;}

int GET_EXTERNAL_OFFSET_APPLIED() {return 0;}
EmcPose GET_EXTERNAL_OFFSETS() {
    EmcPose e;
    e.tran.x = 0;
    e.tran.y = 0;
    e.tran.z = 0;
    e.a      = 0;
    e.b      = 0;
    e.c      = 0;
    e.u      = 0;
    e.v      = 0;
    e.w      = 0;
    return e;
};

int GET_EXTERNAL_AXIS_MASK() {
    int mask = 7;                                       /* XYZABC */
    canon_guard([&]{
        py::object result = py::handle(callback).attr("get_axis_mask")();
        if(!PyLong_Check(result.ptr())) { interp_error ++; return; }
        mask = (int)PyLong_AsLong(result.ptr());
    });
    return mask;
}

double GET_EXTERNAL_TOOL_LENGTH_XOFFSET() {
    return tool_offset.tran.x;
}
double GET_EXTERNAL_TOOL_LENGTH_YOFFSET() {
    return tool_offset.tran.y;
}
double GET_EXTERNAL_TOOL_LENGTH_ZOFFSET() {
    return tool_offset.tran.z;
}
double GET_EXTERNAL_TOOL_LENGTH_AOFFSET() {
    return tool_offset.a;
}
double GET_EXTERNAL_TOOL_LENGTH_BOFFSET() {
    return tool_offset.b;
}
double GET_EXTERNAL_TOOL_LENGTH_COFFSET() {
    return tool_offset.c;
}
double GET_EXTERNAL_TOOL_LENGTH_UOFFSET() {
    return tool_offset.u;
}
double GET_EXTERNAL_TOOL_LENGTH_VOFFSET() {
    return tool_offset.v;
}
double GET_EXTERNAL_TOOL_LENGTH_WOFFSET() {
    return tool_offset.w;
}

// An exact type check, not py::cast: these two reject an int where a float is
// wanted, and the message is the one callers have always seen.
static double exact_float(const char *func, py::handle p) {
    if(!PyFloat_Check(p.ptr()))
        throw py::type_error(std::string(func) + ": Expected float, got "
                             + Py_TYPE(p.ptr())->tp_name);
    return PyFloat_AS_DOUBLE(p.ptr());
}

// The machine's unit scales. Both are constants for the whole parse, and both
// are asked for repeatedly: arc_segments() wants the length units once per arc
// for its small-arc test, which in renderer mode is once per arc *rendered* -
// thousands of Python calls for a number that cannot move. On the callback
// protocol they are left alone, because there the canon was going to be called
// per event anyway and a canon may legitimately watch the traffic. Cached only
// while a renderer is armed; parse_file drops both at the start of a parse.
static double length_units_ = 0.0;
static bool length_units_known = false;
static double angle_units_ = 0.0;
static bool angle_units_known = false;

double GET_EXTERNAL_ANGLE_UNITS() {
    if(angle_units_known) return angle_units_;
    double dresult = 1.0;
    canon_guard([&]{
        // Note the mismatch, kept: the method is `angular`, the message says
        // `angle`.
        dresult = exact_float("get_external_angle_units",
                py::handle(callback).attr("get_external_angular_units")());
    });
    if(GCodeRenderer::active() && !interp_error) {
        angle_units_ = dresult;
        angle_units_known = true;
    }
    return dresult;
}

double GET_EXTERNAL_LENGTH_UNITS() {
    if(length_units_known) return length_units_;
    double dresult = 0.03937007874016;
    canon_guard([&]{
        dresult = exact_float("get_external_length_units",
                py::handle(callback).attr("get_external_length_units")());
    });
    if(GCodeRenderer::active() && !interp_error) {
        length_units_ = dresult;
        length_units_known = true;
    }
    return dresult;
}

// True to stop the parse. A failed call aborts it too, and - unlike the
// forwarders - without touching interp_error: parse_file returns on this
// answer alone, carrying whatever exception is set.
static bool check_abort() {
    try {
        py::object result = py::handle(callback).attr("check_abort")();
        if(PyObject_IsTrue(result.ptr())) {
            PyErr_Format(PyExc_KeyboardInterrupt, "Load aborted");
            return true;
        }
    } catch(py::error_already_set &e) {
        e.restore();
        return true;
    }
    return false;
}

USER_DEFINED_FUNCTION_TYPE USER_DEFINED_FUNCTION[USER_DEFINED_FUNCTION_NUM];

CANON_MOTION_MODE motion_mode;
/* G64_R_PLANNER: preview module ignores the planner-mode args (no motion) */
void SET_MOTION_CONTROL_MODE(CANON_MOTION_MODE mode, double /*tolerance*/, int /*planner_type*/, double /*scurve_peak_scale*/) { motion_mode = mode; }
void SET_MOTION_CONTROL_MODE(double /*tolerance*/) { }
void SET_MOTION_CONTROL_MODE(CANON_MOTION_MODE mode) { motion_mode = mode; }
CANON_MOTION_MODE GET_EXTERNAL_MOTION_CONTROL_MODE() { return motion_mode; }
void SET_NAIVECAM_TOLERANCE(double /*tolerance*/) { }

#define RESULT_OK (result == INTERP_OK || result == INTERP_EXECUTE_FINISH)

// The parse. `initcodes` is the list of the new signature or null for the
// legacy one; the two overloads registered on the module below pick between
// them the way the pair of PyArg_ParseTuple attempts used to.
//
// Every failure exit throws, and the ones that must not skip the epilogue -
// anything after the interpreter is open - reach it through `out_error`.
static py::object parse_file(const char *f, py::handle canon,
                             py::handle initcodes,
                             const char *unitcode, const char *initcode,
                             const char *interpname) {
    int error_line_offset = 0;
    // How often the parse stops to report progress and let the GUI breathe.
    // A GUI's progress bar and its abort button both hang off this one tick -
    // the report moves the bar, and check_abort() is what pumps the event
    // loop the button is waiting in. It used to be one second, from when a
    // preview was built one Python call per move and a big file took minutes;
    // a rendered million-move file now parses in under a second, so the bar
    // moved exactly once, at the end. 100ms reads as continuous, and what it
    // gates is two Python calls - nothing beside a parse.
    // steady_clock, not gettimeofday: a wall clock can step backwards under
    // NTP and stall the tick for as long as the step.
    using clock = std::chrono::steady_clock;
    const auto tick = std::chrono::milliseconds(100);
    clock::time_point last;
    int result = INTERP_OK;

    // Borrowed for the length of the parse, as it always has been: nothing
    // holds a reference to the canon after this returns.
    callback = canon.ptr();

    // Protocol selection, once, before anything is interpreted: a mode that
    // could flip mid-parse would leave the canon's program half in each
    // protocol, and a per-call attribute check would cost more than the
    // renderer saves.
    // Before arming: a renderer reads its starting state off the canon, and a
    // failed read has to be visible rather than cleared by the reset below.
    interp_error = 0;
    if(!GCodeRenderer::arm(callback)) throw py::error_already_set();
    last_delivered_sequence_number = -1;

    if(pinterp) {
        delete pinterp;
        pinterp = NULL;
    }
    if(interpname && *interpname)
        pinterp = interp_from_shlib(interpname);
    if(!pinterp)
        pinterp = new Interp;

    for(int i=0; i<USER_DEFINED_FUNCTION_NUM; i++) 
        USER_DEFINED_FUNCTION[i] = user_defined_function;

    last = clock::now();

    metric=false;
    length_units_known = false;
    angle_units_known = false;
    last_sequence_number = -1;

    _pos_x = _pos_y = _pos_z = _pos_a = _pos_b = _pos_c = 0;
    _pos_u = _pos_v = _pos_w = 0;

    pinterp->init();
    pinterp->open(f);

    maybe_new_line();

    if(initcodes) {
        Py_ssize_t ncodes = PyList_Size(initcodes.ptr());
        for(Py_ssize_t i=0; i<ncodes && RESULT_OK; i++)
        {
            PyObject *item = PyList_GetItem(initcodes.ptr(), i);
            if(!item) throw py::error_already_set();
            const char *code = PyUnicode_AsUTF8(item);
            if(!code) throw py::error_already_set();
            result = pinterp->read(code);
            if(!RESULT_OK) goto out_error;
            result = pinterp->execute();
        }
    }
    if(unitcode && RESULT_OK) {
        result = pinterp->read(unitcode);
        if(!RESULT_OK) goto out_error;
        result = pinterp->execute();
    }

    if(initcode && RESULT_OK) {
        result = pinterp->read(initcode);
        if(!RESULT_OK) goto out_error;
        result = pinterp->execute();
    }

    while(!interp_error && RESULT_OK) {
        error_line_offset = 1;
        result = pinterp->read();
        clock::time_point now = clock::now();
        if(now - last >= tick) {
            // Bounds how stale a canon's progress report can get through a
            // long run of moves on one line, and keeps check_abort() - which
            // pumps AXIS's event loop - from running with a pending exception
            // raised by the report.
            GCodeRenderer::progress();
            if(interp_error) break;
            if(check_abort()) {
                GCodeRenderer::finish();
                throw py::error_already_set();
            }
            last = now;
        }
        if(!RESULT_OK) break;
        error_line_offset = 0;
        result = pinterp->execute();
    }
out_error:
    if(pinterp)
    {
        auto interp = dynamic_cast<Interp*>(pinterp);
        if(interp) interp->_setup.use_lazy_close = false;
        pinterp->close();
    }
    if(interp_error) {
        // Hand over what was rendered before the failure: a partial preview is
        // what a partial program has always produced.
        GCodeRenderer::finish();
        if(!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError,
                    "interp_error > 0 but no Python exception set");
        } else {
            // seems a PyErr_Ocurred(), but no exception was set ?
            // so return error info that can be caught and handled
            PyErr_Format(PyExc_RuntimeError,"parse_file interp_error");
            fprintf(stderr,"!!!%s: parse_file() f=%s\n"
                    "!!!interp_error=%d result=%d last_sequence_number=%d\n",
                    __FILE__,f,interp_error,result,last_sequence_number);
        }
        throw py::error_already_set();
    }
    PyErr_Clear();
    maybe_new_line();
    GCodeRenderer::finish();
    if(PyErr_Occurred()) { interp_error = 1; goto out_error; }
    return py::make_tuple(result, last_sequence_number + error_line_offset);
}


static int maxerror = -1;

static char savedError[LINELEN+1];
static py::str rs274_strerror(int err) {
    // The text belongs to the interpreter the last parse built; without one
    // there is nothing to ask, and dereferencing it would end the process.
    if(!pinterp)
        throw std::runtime_error("gcode.strerror: no interpreter yet - "
                                 "call gcode.parse first");
    pinterp->error_text(err, savedError, LINELEN);
    return py::str(savedError);
}

// One (min, max) box over every point of every move handed in, plus the box
// with each point's tool offset added. Legacy API: the renderer accumulates
// its own extents during the parse and never comes through here.
static py::tuple rs274_calc_extents(py::args args) {
    double min_x = 9e99, min_y = 9e99, min_z = 9e99,
           min_xt = 9e99, min_yt = 9e99, min_zt = 9e99,
           max_x = -9e99, max_y = -9e99, max_z = -9e99,
           max_xt = -9e99, max_yt = -9e99, max_zt = -9e99;
    for(py::handle group : args) {
        py::sequence segs = py::reinterpret_borrow<py::sequence>(group);
        size_t n = py::len(segs);
        double xe = 0, ye = 0, ze = 0, xt = 0, yt = 0, zt = 0;
        for(size_t j=0; j<n; j++) {
            py::sequence seg = segs[j].cast<py::sequence>();
            // The two accepted shapes - (line, start, end, tooloffset) and
            // (line, start, end, ?, tooloffset) - differ only in where the
            // offset sits, so both reduce to these three reads.
            py::sequence start = seg[1].cast<py::sequence>();
            py::sequence end = seg[2].cast<py::sequence>();
            py::sequence tool = seg[py::len(seg) - 1].cast<py::sequence>();
            double xs = start[0].cast<double>();
            double ys = start[1].cast<double>();
            double zs = start[2].cast<double>();
            xe = end[0].cast<double>();
            ye = end[1].cast<double>();
            ze = end[2].cast<double>();
            xt = tool[0].cast<double>();
            yt = tool[1].cast<double>();
            zt = tool[2].cast<double>();
            max_x = std::max(max_x, xs);
            max_y = std::max(max_y, ys);
            max_z = std::max(max_z, zs);
            min_x = std::min(min_x, xs);
            min_y = std::min(min_y, ys);
            min_z = std::min(min_z, zs);
            max_xt = std::max(max_xt, xs+xt);
            max_yt = std::max(max_yt, ys+yt);
            max_zt = std::max(max_zt, zs+zt);
            min_xt = std::min(min_xt, xs+xt);
            min_yt = std::min(min_yt, ys+yt);
            min_zt = std::min(min_zt, zs+zt);
        }
        // The last segment's endpoint: every other one is some segment's
        // start and was taken above.
        if(n > 0) {
            max_x = std::max(max_x, xe);
            max_y = std::max(max_y, ye);
            max_z = std::max(max_z, ze);
            min_x = std::min(min_x, xe);
            min_y = std::min(min_y, ye);
            min_z = std::min(min_z, ze);
            max_xt = std::max(max_xt, xe+xt);
            max_yt = std::max(max_yt, ye+yt);
            max_zt = std::max(max_zt, ze+zt);
            min_xt = std::min(min_xt, xe+xt);
            min_yt = std::min(min_yt, ye+yt);
            min_zt = std::min(min_zt, ze+zt);
        }
    }
    auto box = [](double a, double b, double c) {
        py::list l(3);
        l[0] = a; l[1] = b; l[2] = c;
        return l;
    };
    return py::make_tuple(box(min_x, min_y, min_z), box(max_x, max_y, max_z),
                          box(min_xt, min_yt, min_zt),
                          box(max_xt, max_yt, max_zt));
}

// The two exact-type attribute reads the arc entry point has always made. An
// int where a float is wanted is an error, and the message is the old one.
static int attr_int(py::handle o, const char *name) {
    py::object v = o.attr(name);
    if(!PyLong_Check(v.ptr()))
        throw py::type_error(std::string(name) + ": Expected int, got "
                             + Py_TYPE(v.ptr())->tp_name);
    return (int)PyLong_AsLong(v.ptr());
}

static double attr_double(py::handle o, const char *name) {
    py::object v = o.attr(name);
    if(!PyFloat_Check(v.ptr()))
        throw py::type_error(std::string(name) + ": Expected float, got "
                             + Py_TYPE(v.ptr())->tp_name);
    return PyFloat_AS_DOUBLE(v.ptr());
}

static py::list rs274_arc_to_segments(py::handle canon,
        double x1, double y1, double cx, double cy, int rot,
        double z1, double a, double b, double c, double u, double v, double w,
        int max_segments) {
    Point9 o, g5xoffset, g92offset;

    py::sequence lo = canon.attr("lo").cast<py::sequence>();
    if(py::len(lo) != 9)
        throw py::value_error("arc_to_segments: canon.lo is not nine numbers");
    for(size_t i=0; i<9; i++) o[i] = lo[i].cast<double>();
    int plane = attr_int(canon, "plane");
    double rotation_cos = attr_double(canon, "rotation_cos");
    double rotation_sin = attr_double(canon, "rotation_sin");
    static const char *const G5X[9] = {"g5x_offset_x", "g5x_offset_y",
        "g5x_offset_z", "g5x_offset_a", "g5x_offset_b", "g5x_offset_c",
        "g5x_offset_u", "g5x_offset_v", "g5x_offset_w"};
    static const char *const G92[9] = {"g92_offset_x", "g92_offset_y",
        "g92_offset_z", "g92_offset_a", "g92_offset_b", "g92_offset_c",
        "g92_offset_u", "g92_offset_v", "g92_offset_w"};
    for(int i=0; i<9; i++) {
        g5xoffset[i] = attr_double(canon, G5X[i]);
        g92offset[i] = attr_double(canon, G92[i]);
    }

    std::vector<Point9> pts;
    int steps = arc_segments(o, plane, rotation_cos, rotation_sin,
                             g5xoffset, g92offset,
                             x1, y1, cx, cy, rot, z1, a, b, c, u, v, w,
                             max_segments, pts);
    py::list segs(steps);
    for(int i=0; i<steps; i++) {
        const Point9 &p = pts[(size_t)i];
        segs[i] = py::make_tuple(p[0], p[1], p[2], p[3], p[4], p[5],
                                 p[6], p[7], p[8]);
    }
    return segs;
}

PYBIND11_MODULE(gcode, m) {
    m.doc() = "Interface to EMC rs274ngc interpreter";

    linecode_register(m);
    preview_geometry_register(m);

    // Registration order is the dispatch order: the list form is tried first,
    // exactly as the pair of PyArg_ParseTuple attempts did.
    m.def("parse", [](const char *f, py::handle canon, py::list initcodes,
                      const char *interpname) {
                return parse_file(f, canon, initcodes, nullptr, nullptr,
                                  interpname);
            },
            py::arg("filename"), py::arg("canon"), py::arg("initcodes"),
            py::arg("interpname") = py::none(),
            "Parse a G-Code file");
    m.def("parse", [](const char *f, py::handle canon, const char *unitcode,
                      const char *initcode, const char *interpname) {
                return parse_file(f, canon, py::handle(), unitcode, initcode,
                                  interpname);
            },
            py::arg("filename"), py::arg("canon"),
            py::arg("unitcode") = py::none(), py::arg("initcode") = py::none(),
            py::arg("interpname") = py::none(),
            "Parse a G-Code file");

    m.def("strerror", &rs274_strerror, py::arg("error"),
            "Convert a numeric error to a string");
    m.def("calc_extents", &rs274_calc_extents,
            "Calculate information about extents of gcode");
    m.def("arc_to_segments", &rs274_arc_to_segments,
            py::arg("canon"), py::arg("x1"), py::arg("y1"),
            py::arg("cx"), py::arg("cy"), py::arg("rot"), py::arg("z1"),
            py::arg("a"), py::arg("b"), py::arg("c"),
            py::arg("u"), py::arg("v"), py::arg("w"),
            py::arg("max_segments") = 128,
            "Convert an arc to straight segments");

    m.attr("MAX_ERROR") = maxerror;
    m.attr("MIN_ERROR") = static_cast<int>(INTERP_MIN_ERROR);
}
// vim:ts=8:sts=4:sw=4:et:
