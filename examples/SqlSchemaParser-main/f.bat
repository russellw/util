clang-format -i --style=file SqlSchemaParser\*.cs||exit /b
clang-format -i --style=file TestProject1\*.cs||exit /b
clang-format -i --style=file ConsoleApp1\*.cs||exit /b
call clean-cs -i -r .||exit /b
git diff
