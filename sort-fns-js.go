package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
)

var writeBack bool
var lines []string
var fnRe = regexp.MustCompile(`function\s+(\w+)`)

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
	lines = readLines(path)
	fmt.Println(path)
}
