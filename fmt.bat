clang-format -i --style=file *.c *.cpp||exit /b
for %%x in (*.go) do gofmt -s -w %%x||exit /b
git diff
