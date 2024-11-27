package main

import (
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"strings"
)

func main() {
	// Get a list of all files in the current commit
	cmd := exec.Command("git", "ls-tree", "-r", "--name-only", "HEAD")
	output, err := cmd.Output()
	if err != nil {
		log.Fatalf("Error running git command: %v", err)
	}

	files := strings.Split(string(output), "\n")

	for _, file := range files {
		if file == "" {
			continue
		}

		// Use 'git cat-file' to check if a file is binary
		catCmd := exec.Command("git", "cat-file", "-p", "HEAD:"+file)
		catOutput, err := catCmd.Output()
		if err != nil {
			log.Printf("Skipping file %s due to error: %v\n", file, err)
			continue
		}

		// If the output contains a null byte, it's binary
		if bytes.Contains(catOutput, []byte{0}) {
			fmt.Printf("B %s\n", file)
		} else {
			fmt.Printf("  %s\n", file)
		}
	}
}
