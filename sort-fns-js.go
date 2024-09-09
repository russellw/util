package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func main() {
	writeBack := flag.Bool("w", false, "write modified files back to disk")
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		log.Fatal("No files were specified")
	}
	for _, file := range files {
		processFile(file, *writeBack)
	}
}

func processFile(path string, writeBack bool) {
	if filepath.Ext(path) != ".js" {
		return
	}
}
