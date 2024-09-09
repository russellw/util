package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"unicode"
)

// Removes trailing whitespace from a string.
func trimTrailingWhitespace(line string) string {
	return strings.TrimRightFunc(line, unicode.IsSpace)
}

// Processes a file to remove trailing whitespace.
func processFile(file string, writeChanges bool) error {
	if isBinary(file) {
		return nil
	}

	inputFile, err := os.Open(file)
	if err != nil {
		return err
	}
	defer inputFile.Close()

	var outputLines []string
	changed := false

	scanner := bufio.NewScanner(inputFile)
	for scanner.Scan() {
		line := scanner.Text()
		trimmedLine := trimTrailingWhitespace(line)
		if line != trimmedLine {
			changed = true
		}
		outputLines = append(outputLines, trimmedLine)
	}

	if scanner.Err() != nil {
		return scanner.Err()
	}

	if changed {
		fmt.Println(file)
		if writeChanges {
			// Rewriting the file
			err := ioutil.WriteFile(file, []byte(strings.Join(outputLines, "\n")+"\n"), 0644)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// Processes files, checking for binary files and removing trailing whitespace.
func processFiles(files []string, writeChanges bool) error {
	for _, file := range files {
		err := processFile(file, writeChanges)
		if err != nil {
			log.Fatal(err)
		}
	}
	return nil
}

func main() {
	writeFlag := flag.Bool("w", false, "Rewrite files with trailing whitespace removed")
	flag.Parse()

	if len(flag.Args()) == 0 {
		log.Fatal("Usage: remove-trail-space [-w] file1 [file2 ...]")
	}

	err := processFiles(flag.Args(), *writeFlag)
	if err != nil {
		log.Fatal(err)
	}
}
