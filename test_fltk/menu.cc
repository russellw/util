#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/fl_ask.H>

// Callback functions for menu items
void OpenFileCB(Fl_Widget*, void*) {
    fl_message("Open file dialog will appear here.");
}

void ExitCB(Fl_Widget*, void*) {
    exit(0);
}

void AboutCB(Fl_Widget*, void*) {
    fl_message("This is a simple GUI program using FLTK.");
}

int main(int argc, char **argv) {
    Fl_Window *window = new Fl_Window(400, 200, "Sample FLTK App");
    Fl_Menu_Bar *menu_bar = new Fl_Menu_Bar(0, 0, 400, 25);

    // Menu items
    menu_bar->add("File/Open", 0, OpenFileCB);
    menu_bar->add("File/Exit", 0, ExitCB);
    menu_bar->add("Help/About", 0, AboutCB);

    window->end();
    window->show(argc, argv);
    return Fl::run();
}
