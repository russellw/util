package main

import (
	"flag"
	"fmt"
	"log"
	"regexp"
	"sort"
)

var writeBack bool
var beginRe = regexp.MustCompile(`^\s*// SORT`)
var endRe = regexp.MustCompile(`^\s*//`)

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
	lines := readLines(path)
	old := copySlice(lines)
	for i := 0; ; {
		i = findString(beginRe, lines, i)
		if i < 0 {
			break
		}
		j := findString(endRe, lines, i+1)
		if j < 0 {
			log.Fatalf("%s:%d: no end marker", path, i+1)
		}
		sort.Strings(lines[i+1 : j])
		i = j + 1
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
