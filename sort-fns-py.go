package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"unicode"
)

var fnRe = regexp.MustCompile(`^def\s+(\w+)`)
var writeBack bool

func beginFn(s string) string {
	match := fnRe.FindStringSubmatch(s)
	if len(match) > 1 {
		return match[1] // Return the function name (first captured group)
	}
	return ""
}

func endFn(s string) EndSpecialKind {
	if len(s) == 0 {
		return endSpecialNo
	}
	if unicode.IsSpace(rune(s[0])) {
		return endSpecialNo
	}
	return endSpecialExclude
}

func main() {
	flag.BoolVar(&writeBack, "w", false, "write modified files back to disk")
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		log.Fatal("No files were specified")
	}
	for _, path := range files {
		processFile(path)
	}
}

func processFile(path string) {
	if filepath.Ext(path) != ".py" {
		return
	}
	lines := readLines(path)
	old := lines
	chunks := parseChunks(isComment, beginFn, endFn, lines)
	sortChunks(chunks)
	lines = joinChunks(chunks)
	if eqStrings(old, lines) {
		return
	}
	if writeBack {
		fmt.Println(path)
		writeLines(path, lines)
	} else {
		printLines(lines)
	}
}
