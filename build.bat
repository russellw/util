go build  -ldflags "-s -w" %*||exit /b
move *.exe \b
