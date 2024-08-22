package main

import (
    "bufio"
    "fmt"
    "io/fs"
    "os"
    "path/filepath"
    "strings"
    "unicode/utf8"
)

// isBinaryFile checks if a file is binary by scanning the first 1024 bytes.
func isBinaryFile(filename string) bool {
    file, err := os.Open(filename)
    if err != nil {
        return false
    }
    defer file.Close()

    buf := make([]byte, 1024)
    n, err := file.Read(buf)
    if err != nil {
        return false
    }

    return !utf8.Valid(buf[:n])
}

// hasTrailingWhitespace checks if a line in the file has trailing whitespace.
func hasTrailingWhitespace(filename string) (bool, error) {
    file, err := os.Open(filename)
    if err != nil {
        return false, err
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        if strings.HasSuffix(line, " ") || strings.HasSuffix(line, "\t") {
            return true, nil
        }
    }

    return false, scanner.Err()
}

// walkFunc is the function called for each file in the directory tree.
func walkFunc(path string, info fs.FileInfo, err error) error {
    if err != nil {
        return err
    }

    // Skip directories and symlinks.
    if info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
        return nil
    }

    // Check if the file is binary.
    if isBinaryFile(path) {
        return nil
    }

    // Check if the file has trailing whitespace.
    containsWhitespace, err := hasTrailingWhitespace(path)
    if err != nil {
        return err
    }

    if containsWhitespace {
        fmt.Println(path)
    }

    return nil
}

func main() {
    root := "."

    err := filepath.Walk(root, walkFunc)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
