rem C and C++
clang-format -i --style=file *.c *.cpp||exit /b

rem Go
for %%x in (*.go) do gofmt -s -w %%x||exit /b

rem JavaScript
call prettier --no-semi --print-width 132 -w .||exit /b

rem All the above
do-all-recur . comment-space -w||exit /b

rem Check the results
git diff
