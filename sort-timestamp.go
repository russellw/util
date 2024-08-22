package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"time"
)

type fileInfo struct {
	name    string
	modTime time.Time
}

func main() {
	var files []fileInfo
	scanner := bufio.NewScanner(os.Stdin)

	// Read filenames from standard input
	for scanner.Scan() {
		filename := scanner.Text()
		info, err := os.Stat(filename)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file %s: %v\n", filename, err)
			continue
		}
		files = append(files, fileInfo{name: filename, modTime: info.ModTime()})
	}

	// Check for any errors encountered by the scanner
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		return
	}

	// Sort the files by modification time
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.Before(files[j].modTime)
	})

	// Print the sorted filenames
	for _, file := range files {
		fmt.Println(file.name)
	}
}
