package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
)

var writeBack bool
var fnRe = regexp.MustCompile(`function\s+(\w+)`)

// The input string may or may not contain a JavaScript function declaration
// Returns the function name if so, otherwise the empty string
func beginFn(s string) string {
	match := fnRe.FindStringSubmatch(s)
	if len(match) > 1 {
		return match[1] // Return the function name (first captured group)
	}
	return ""
}

func endFn(s string) EndSpecialKind {
	if s == "}" {
		return endSpecialInclude
	}
	return endSpecialNo
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
	if filepath.Ext(path) != ".js" {
		return
	}
	lines := readLines(path)
	old := lines
	chunks := parseChunks(isComment, beginFn, endFn, lines)
	sortChunks(chunks)
	specialSpace(chunks)
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
