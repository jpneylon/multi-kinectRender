#ifndef __VR_WINDOW_CLASS_H__
#define __VR_WINDOW_CLASS_H__

#include <gtkmm.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <assert.h>
#include <time.h>

#include "VRender.h"
#include "socket_class.h"

#define STARTDIR "/home/anand/code/data"
#define MAX_VOLUME_SIDE 300
#define MASK_BONUS 16
#define MAX_RADIUS 1
#define TIMER_SIZE 100



typedef struct ipcBarrier_st
{
    int count;
    bool sense;
    bool allExit;
} ipcBarrier_t;



class VR_Window : public Gtk::Box
{
  public:
    VR_Window();
    virtual ~VR_Window();

    void open_file();
    void open_socket_connection( int nsockets );
    void print_file();
    char *get_file_name() { return point_cloud_list_file; };
    void initialize_vrender();
    void create_render_window();

  private:
    bool on_idle();
    bool on_timer();
    void on_click();
    void select_file();
    void read_socket_connection();
    void update_render_buffer();
    void set_render_density();
    void set_render_brightness();
    void set_render_offset();
    void set_render_scale();
    void update_socket_data();
    void update_render_zoom(gdouble x, gdouble y);
    void update_render_translation(gdouble x, gdouble y);
    void update_render_rotation(gdouble x, gdouble y);
    virtual bool render_button_press_event(GdkEventButton *event);
    virtual bool render_motion_notify_event(GdkEventMotion *event);

    CS_Socket *server_socket;
    VRender *vrender;
    Cloud *cloud;

    int tcp_instruction[2]; // [0]: instruction type [1]: data size
    float3 volume_origin;

    char *point_cloud_list_file;
    bool pc_file_open;
    bool adaptive_world_sizing;
    double socket_timer[TIMER_SIZE];
    int socket_timer_idx;
    uint numKinects;

    Gtk::Image      render_image;
    Gtk::Label      fps_update;
    Gtk::Label      socket_update;
    Gtk::Label      fps_label;
    Gtk::Label      socket_label;
    Gtk::Box        fps_box;
    Gtk::Box        socket_box;

    Glib::RefPtr<Gtk::Adjustment>     dens_adjust;
    Glib::RefPtr<Gtk::Adjustment>     bright_adjust;
    Glib::RefPtr<Gtk::Adjustment>     offset_adjust;
    Glib::RefPtr<Gtk::Adjustment>     scale_adjust;
};






#endif // __VR_WINDOW_CLASS_H__
