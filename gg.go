package main

import (
	"bytes"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

// formatFileList formats a list of files to show up to 3 names, with an ellipsis if there are more
func formatFileList(files []string) string {
	if len(files) <= 3 {
		return strings.Join(files, ", ")
	}
	return fmt.Sprintf("%s, ...and %d more", strings.Join(files[:3], ", "), len(files)-3)
}

func generateCommitMessage(diffOutput []byte) string {
	// Parse the diff output into changed files and their types
	changedFiles := map[string][]string{
		"A": {}, // Added files
		"M": {}, // Modified files
		"D": {}, // Deleted files
	}

	// Process each line of diff output
	lines := strings.Split(string(diffOutput), "\n")
	for _, line := range lines {
		// Match lines of the form "<status>\t<filename>"
		if match := regexp.MustCompile(`^(A|M|D)\t(.+)$`).FindStringSubmatch(line); match != nil {
			status, file := match[1], match[2]
			changedFiles[status] = append(changedFiles[status], file)
		}
	}

	// Construct the message based on changes, including file names
	var messageParts []string
	for action, files := range changedFiles {
		if len(files) > 0 {
			var actionVerb string
			switch action {
			case "A":
				actionVerb = "Added"
			case "M":
				actionVerb = "Modified"
			case "D":
				actionVerb = "Deleted"
			}
			fileList := formatFileList(files)
			messageParts = append(messageParts, fmt.Sprintf("%s %d file(s): %s", actionVerb, len(files), fileList))
		}
	}

	// Default to "Updated files" if no specific actions were found
	if len(messageParts) == 0 {
		return "Updated files"
	}

	return strings.Join(messageParts, "; ")
}

func main() {
	// Step 1: Capture the `git diff` output
	diffCmd := exec.Command("git", "diff", "--name-status")
	diffOutput, err := diffCmd.Output()
	if err != nil {
		fmt.Println("Error running git diff:", err)
		return
	}

	// Step 2: Generate a commit message based on the diff
	commitMessage := generateCommitMessage(diffOutput)
	fmt.Println("Generated commit message:", commitMessage)

	// Step 3: Run `git commit` with the generated message
	commitCmd := exec.Command("git", "commit", "-a", "-m", commitMessage)
	commitCmd.Stdout = &bytes.Buffer{}
	commitCmd.Stderr = &bytes.Buffer{}
	err = commitCmd.Run()
	if err != nil {
		fmt.Println("Error running git commit:", err)
		fmt.Println("Output:", commitCmd.Stderr)
	} else {
		fmt.Println("Commit successful!")
	}
}
