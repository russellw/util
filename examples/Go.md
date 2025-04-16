When a simple source file is not enough, and an actual project is needed:

```
md foo
mv foo.go foo\main.go
cd foo
go mod init foo
go mod tidy
```

## GPT instructions
Write a Go program to .

The name of the input file should be supplied on the command line. If omitted, read standard input.

Output should be printed to standard output, unless the -w option is given, in which case it should overwrite the input file (if there was one).

You don't need to bother returning errors from functions. All errors should be treated as fatal.
