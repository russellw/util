rem go build -ldflags "-s -w"||exit /b

cl /O2 /wd4530 hsla_to_rgba.cpp||exit /b
del *.obj
