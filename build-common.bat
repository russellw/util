go build -ldflags "-s -w" %* common.go||exit /b
move *.exe \b
