package main

import (
	"bytes"
	"fmt"
	"os"
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

// generateCommitMessage constructs a message from git diff output
func generateCommitMessage(diffOutput []byte) string {
	changedFiles := map[string][]string{
		"A": {}, // Added files
		"M": {}, // Modified files
		"D": {}, // Deleted files
	}

	lines := strings.Split(string(diffOutput), "\n")
	for _, line := range lines {
		if match := regexp.MustCompile(`^(A|M|D)\t(.+)$`).FindStringSubmatch(line); match != nil {
			status, file := match[1], match[2]
			changedFiles[status] = append(changedFiles[status], file)
		}
	}

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

	if len(messageParts) == 0 {
		return "Updated files"
	}
	return strings.Join(messageParts, "; ")
}

func main() {
	// Stage all changes (new and modified)
	addCmd := exec.Command("git", "add", ".")
	addCmd.Run()

	// Capture the diff relative to the last commit
	diffCmd := exec.Command("git", "diff", "--name-status", "--cached")
	diffOutput, err := diffCmd.Output()
	if err != nil {
		fmt.Println("Error running git diff:", err)
		return
	}

	// Determine the commit message
	var commitMessage string
	if len(os.Args) > 1 {
		commitMessage = os.Args[1] // Use the provided message if available
	} else {
		commitMessage = generateCommitMessage(diffOutput)
	}
	fmt.Println("Commit message:", commitMessage)

	// Run `git commit`
	commitCmd := exec.Command("git", "commit", "-m", commitMessage)
       commitCmd.Stdout = &bytes.Buffer{}
       commitCmd.Stderr = &bytes.Buffer{}	
err = commitCmd.Run()
	if err != nil {
		fmt.Println("Error running git commit:", err)
		fmt.Println("Output:", commitCmd.Stderr)
		return
	}
	fmt.Println("Commit successful!")

	// Check for a remote repository
	remoteCmd := exec.Command("git", "remote")
	remoteOutput, err := remoteCmd.Output()
	if err != nil {
		fmt.Println("Error checking git remote:", err)
		return
	}

	// Push only if a remote repository exists
	if len(remoteOutput) > 0 {
		fmt.Println("Remote repository detected. Pushing changes...")
		pushCmd := exec.Command("git", "push")
		pushCmd.Stdout = os.Stdout
		pushCmd.Stderr = os.Stderr
		err = pushCmd.Run()
		if err != nil {
			fmt.Println("Error running git push:", err)
			return
		}
		fmt.Println("Push successful!")
	} else {
		fmt.Println("No remote repository found. Skipping push.")
	}
}
