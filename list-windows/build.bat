cl /O2 /wd4530 list-windows.cpp /link User32.lib||exit /b
del *.obj
move *.exe \b
