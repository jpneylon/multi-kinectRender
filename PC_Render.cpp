
#include "PC_Render.h"
#include <gtkmm/application.h>


int main(int argc, char *argv[])
{
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create( argc, argv, "jacko.multi_kinect_render" );
    PC_Render multi_kinect_render;
    return app->run(multi_kinect_render);
}





PC_Render::PC_Render() :
    mainBox( Gtk::ORIENTATION_VERTICAL )
{
    renderer_open = false;

    /* Create the main window */
    set_title("Point Cloud Renderer");
    set_position( Gtk::WIN_POS_CENTER );
    set_size_request( BUFFER_SIZE+128, BUFFER_SIZE+128);
    set_resizable(true);
    add( mainBox );

    mainBox.set_border_width(1);

    viewBox.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
    viewBox.set_border_width(1);
    viewBox.set_size_request(-1, -1);

    actionGroup = Gtk::ActionGroup::create();

    // File Sub Menu Items
    actionGroup->add( Gtk::Action::create("MenuFile", "_File") );
    actionGroup->add( Gtk::Action::create("Open", Gtk::Stock::OPEN),
        sigc::mem_fun( *this, &PC_Render::file_open) );
    actionGroup->add( Gtk::Action::create("Save", Gtk::Stock::SAVE),
        sigc::mem_fun( *this, &PC_Render::file_print) );
    actionGroup->add( Gtk::Action::create("Close", Gtk::Stock::CLOSE),
        sigc::mem_fun( *this, &PC_Render::file_close) );
    actionGroup->add( Gtk::Action::create("Quit", Gtk::Stock::QUIT),
        sigc::mem_fun( *this, &PC_Render::delete_event) );

    manager = Gtk::UIManager::create();
    manager->insert_action_group( actionGroup );
    add_accel_group( manager->get_accel_group() );

        Glib::ustring ui_info =
            "<ui>"
            "   <menubar name='MenuBar'>"
            "       <menu action='MenuFile'>"
            "           <menuitem action='Open'/>"
            "           <menuitem action='Save'/>"
            "           <menuitem action='Close'/>"
            "           <separator/>"
            "           <menuitem action='Quit'/>"
            "       </menu>"
            "   </menubar>"
            "   <toolbar name='ToolBar'>"
            "       <toolitem action='Open'/>"
            "       <toolitem action='Close'/>"
            "       <toolitem action='Quit'/>"
            "   </toolbar>"
            "</ui>";

        manager->add_ui_from_string(ui_info);

    menuBar = manager->get_widget("/MenuBar");
    mainBox.pack_start( *menuBar, Gtk::PACK_SHRINK );

    toolBar = manager->get_widget("/ToolBar");
    mainBox.pack_start( *toolBar, Gtk::PACK_SHRINK );

    label.set_text("CWD:");
    cwdLabel.set_text("...");

    cwdBox.set_orientation( Gtk::ORIENTATION_HORIZONTAL );
    cwdBox.set_border_width(0);
    cwdBox.pack_start( label, false, false, 2 );
    cwdBox.pack_start( cwdLabel, false, false, 2);

    mainBox.pack_start( viewBox, true, true, 2);
    mainBox.pack_start( cwdBox,  false, false, 2);

    show_all_children();
}


PC_Render::~PC_Render()
{
    delete_event();
}



void
PC_Render::file_close()
{
    if (renderer_open)
    {
        renderer_open = false;
        delete vr_window;
    }
    cwdLabel.set_text( "..." );
    show_all_children();
}


void
PC_Render::create_vr_window()
{
    int ncam = cam_counter->get_number_of_cameras();
    printf("\n Looking for %d camera connections...\n", ncam );
    if (ncam > 0)
    {
        vr_window = new VR_Window;
        vr_window->open_socket_connection( ncam );
        vr_window->initialize_vrender();
        viewBox.pack_start( *vr_window );
        vr_window->create_render_window();
        renderer_open = true;
    }
}


void
PC_Render::file_open()
{
    if (!renderer_open)
    {
        cam_counter = new CameraSpinner;
        cam_counter->show();
        cam_counter->signal_hide().connect(sigc::mem_fun(*this, &PC_Render::create_vr_window));
    }
    show_all_children();
}

void
PC_Render::file_print()
{
    if (renderer_open)
    {
        vr_window->print_file();
    }
}

void
PC_Render::delete_event()
{
    hide();
}

