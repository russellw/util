@echo off
del /s *.exe
del /s *.obj

setlocal enabledelayedexpansion

for /d /r %%d in (*) do (
    if exist "%%d\target\" (
        echo Found target directory in "%%d"
        rmdir /s /q "%%d\target"
        echo Deleted target directory in "%%d"
    )
)
for /d /r %%d in (*) do (
    if exist "%%d\__pycache__\" (
        echo Found __pycache__ directory in "%%d"
        rmdir /s /q "%%d\__pycache__"
        echo Deleted __pycache__ directory in "%%d"
    )
)

endlocal
