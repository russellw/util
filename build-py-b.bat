pyinstaller --onefile %1
move *.spec \t
rd /q /s build
move dist\*.exe \b
rd dist
dir /od \b\*.exe
