package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// isAbsolutePath checks if a file path is absolute on Windows.
func isAbsolutePath(path string) bool {
	// Check if the path starts with a backslash or forward slash, indicating an absolute path relative to the current drive
	if strings.HasPrefix(path, "\\") || strings.HasPrefix(path, "/") {
		return true
	}
	// Check for UNC paths or paths with drive letters (e.g., "C:\")
	return len(path) >= 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/')
}

// findDependencies recursively finds all dependencies of the given file.
func findDependencies(file string, visited map[string]bool) ([]string, error) {
	if visited[file] {
		return nil, nil // Avoid circular dependencies
	}

	visited[file] = true

	file, err := filepath.Abs(file)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var dependencies []string
	scanner := bufio.NewScanner(f)
	includeRegex := regexp.MustCompile(`^\s*#include\s+\"(.*?)\"`)

	for scanner.Scan() {
		line := scanner.Text()
		matches := includeRegex.FindStringSubmatch(line)
		if matches != nil {
			includePath := matches[1]
			if !isAbsolutePath(includePath) {
				includePath = filepath.Join(filepath.Dir(file), includePath)
			}

			absIncludePath, err := filepath.Abs(includePath)
			if err != nil {
				return nil, err
			}

			dependencies = append(dependencies, absIncludePath)
			subDependencies, err := findDependencies(absIncludePath, visited)
			if err != nil {
				return nil, err
			}

			dependencies = append(dependencies, subDependencies...)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return dependencies, nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: go run main.go <input_file>")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	dependencies, err := findDependencies(inputFile, make(map[string]bool))
	if err != nil {
		panic(err)
	}

	for _, dep := range dependencies {
		fmt.Println(dep)
	}
}
