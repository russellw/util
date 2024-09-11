rem C and C++
rem clang-format -i --style=file *.c *.cpp||exit /b

rem Go
gofmt -s -w .||exit /b
do-all-recur . sort-fns-go -w||exit /b

rem JavaScript
call prettier --no-semi --print-width 132 -w .||exit /b

rem All the above
do-all-recur . comment-space -w||exit /b

rem Check the results
git diff
