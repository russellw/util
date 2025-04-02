@echo off
clang-format -i --style=file *.cs||exit /b
git diff
