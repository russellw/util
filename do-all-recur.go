package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

// Function to execute a program on a file
func runProgramOnFile(program string, args []string, filePath string) {
	// Append the file path to the args
	allArgs := append(args, filePath)

	// Execute the command
	cmd := exec.Command(program, allArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()

	if err != nil {
		fmt.Println(filePath)
		if exitError, ok := err.(*exec.ExitError); ok {
			log.Fatalf("program returned non-zero exit code: %v", exitError.ExitCode())
		}
		log.Fatalf("failed to execute program: %v", err)
	}
}

// Function to recursively traverse the directory and run the program on each file
func walkDirAndRunProgram(root string, program string, args []string) {
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			log.Fatalf("error accessing file path: %v", err)
		}
		if d.IsDir() {
			if ignore(d.Name()) {
				return filepath.SkipDir
			}
		} else {
			runProgramOnFile(program, args, path)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("Usage: %s dir program arg1 arg2... \n", os.Args[0])
	}

	// Directory
	dir := os.Args[1]

	// Program and its arguments
	program := os.Args[2]
	args := os.Args[3:]

	// Traverse the directory tree and run the program on each file
	walkDirAndRunProgram(dir, program, args)
}
