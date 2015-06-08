#ifndef __CAMERA_SPINNER_CLASS_H__
#define __CAMERA_SPINNER_CLASS_H__


#include <gtkmm.h>



class CameraSpinner : public Gtk::Window
{
    public:
        CameraSpinner();
        virtual ~CameraSpinner(){};
        int get_number_of_cameras(){ return camera_count; };
    private:
        int camera_count;
        Gtk::SpinButton                 *spinner;
        Gtk::Button                     *spin_button;
        Gtk::Box                        *spin_box;
        Gtk::Frame                      *spin_frame;
        Glib::RefPtr<Gtk::Adjustment>    spin_adjust;
        void on_spin_value_changed();
};


CameraSpinner::CameraSpinner()
 : Gtk::Window( Gtk::WINDOW_POPUP )
{
    set_default_size( 100, 100 );
    set_position( Gtk::WIN_POS_CENTER_ALWAYS );
    set_modal( true );
    set_title(" Set Number of Kinect Cameras ");

    spin_adjust = Gtk::Adjustment::create( 1, 0, 10, 1, 1, 0);

    spinner = new Gtk::SpinButton( spin_adjust, 0.0, 0 );
    spin_button = new Gtk::Button( Gtk::Stock::QUIT );
    spin_button->set_label(" Continue ");

    spin_box = new Gtk::Box( Gtk::ORIENTATION_HORIZONTAL );
    spin_box->pack_start( spinner[0], true, true, 0 );
    spin_box->pack_start( spin_button[0], true, true, 0 );
    spin_box->set_border_width( 10 );

    camera_count = spinner->get_value_as_int();

    spin_button->signal_clicked().connect( sigc::mem_fun( *this, &Gtk::Window::hide ) );
    spinner->signal_value_changed().connect( sigc::mem_fun( *this, &CameraSpinner::on_spin_value_changed ) );

    spin_frame = new Gtk::Frame( "Set Number of Kinect Cameras" );
    spin_frame->add( spin_box[0] );
    add( spin_frame[0] );
    show_all_children();
}


void
CameraSpinner::on_spin_value_changed()
{
    camera_count = spinner->get_value_as_int();
}


#endif // __CAMERA_SPINNER_CLASS_H__
