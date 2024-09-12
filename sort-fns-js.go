package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
)

var fnRe = regexp.MustCompile(`^function\s+(\w+)`)
var writeBack bool

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
	for _, dent := range indentations(lines) {
		// The input string may or may not contain a JavaScript function declaration
		// Returns the function name if so, otherwise the empty string
		beginFn := func(s string) string {
			s = trimPrefixOrEmpty(s, dent)
			match := fnRe.FindStringSubmatch(s)
			if len(match) > 1 {
				return match[1] // Return the function name (first captured group)
			}
			return ""
		}

		endFn := func(s string) EndSpecialKind {
			s = trimPrefixOrEmpty(s, dent)
			if s == "}" {
				return endSpecialInclude
			}
			return endSpecialNo
		}

		chunks := parseChunks(isComment, beginFn, endFn, lines)
		sortChunks(chunks)
		specialSpace(chunks)
		lines = joinChunks(chunks)
	}
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
