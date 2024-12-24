cl /O2 /wd4530 resize-window.cpp /link User32.lib||exit /b
del *.obj
move *.exe \b
