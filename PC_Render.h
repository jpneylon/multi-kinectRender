#ifndef __PC_RENDER_CLASS_H__
#define __PC_RENDER_CLASS_H__


#include <gtkmm.h>
#include "VR_Window.h"
#include "spinner.h"

class PC_Render : public Gtk::Window
{
  public:
    PC_Render();
    virtual ~PC_Render();

  protected:
    void toggle_render_window();
    void file_close();
    void file_open();
    void file_print();
    void delete_event();
    void create_vr_window();

    // Child Widgets
    Gtk::Box        viewBox;
    Gtk::Label      cwdLabel;
    Gtk::Box        mainBox;
    Gtk::Label      label;
    Gtk::Box        cwdBox;

    Gtk::Widget *menuBar;
    Gtk::Widget *toolBar;

    Glib::RefPtr<Gtk::UIManager> manager;
    Glib::RefPtr<Gtk::ActionGroup> actionGroup;

    // Variables
    CameraSpinner   *cam_counter;
    VR_Window       *vr_window;
    bool            renderer_open;
};

#endif // __PC_RENDER_CLASS_H__
