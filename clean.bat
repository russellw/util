del /s *.exe
del /s *.obj

for /d /r %%d in (*) do (
    if exist "%%d\target\" (
        echo Found target directory in "%%d"
        rmdir /s /q "%%d\target"
        echo Deleted target directory in "%%d"
    )
)
