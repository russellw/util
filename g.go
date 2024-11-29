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
	// Step 1: Capture the `git diff` output
	diffCmd := exec.Command("git", "diff", "--name-status")
	diffOutput, err := diffCmd.Output()
	if err != nil {
		fmt.Println("Error running git diff:", err)
		return
	}

	// Step 2: Generate or use provided commit message
	var commitMessage string
	if len(os.Args) > 1 {
		commitMessage = os.Args[1] // Use provided commit message
	} else {
		commitMessage = generateCommitMessage(diffOutput) // Generate commit message
	}
	fmt.Println("Commit message:", commitMessage)

	// Step 3: Run `git commit` with the message
	commitCmd := exec.Command("git", "commit", "-a", "-m", commitMessage)
	commitCmd.Stdout = &bytes.Buffer{}
	commitCmd.Stderr = &bytes.Buffer{}
	err = commitCmd.Run()
	if err != nil {
		fmt.Println("Error running git commit:", err)
		fmt.Println("Output:", commitCmd.Stderr)
		return
	}
	fmt.Println("Commit successful!")

	// Step 4: Check if a remote repository exists
	remoteCmd := exec.Command("git", "remote")
	remoteOutput, err := remoteCmd.Output()
	if err != nil {
		fmt.Println("Error checking git remote:", err)
		return
	}

	if len(remoteOutput) > 0 { // Remote repository exists
		fmt.Println("Remote repository detected, pushing changes...")
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
