package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
	"unicode"
)

func main() {
	// Define command-line flags
	overwrite := flag.Bool("w", false, "overwrite the input file")
	flag.Parse()

	// Ensure a filename is provided
	if flag.NArg() != 1 {
		fmt.Println("Usage: css-comment [-w] <filename>")
		return
	}
	filename := flag.Arg(0)

	// Read the file
	content, err := os.ReadFile(filename)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	// Process the CSS content
	modifiedContent := processCSS(string(content))

	// Handle output based on the -w flag
	if *overwrite {
		err := os.WriteFile(filename, []byte(modifiedContent), 0644)
		if err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	} else {
		fmt.Println(modifiedContent)
	}
}

func processCSS(content string) string {
	var output []string
	lines := strings.Split(content, "\n")

	commentRegex := regexp.MustCompile(`//(.*)`)                 // Find C++ style comments
	trailingRegex := regexp.MustCompile(`(.+?)(/\*.*?\*/\s*)?$`) // Separate code from trailing comments

	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])

		// Convert C++ style comments to C style
		line = commentRegex.ReplaceAllStringFunc(line, func(match string) string {
			comment := strings.TrimSpace(strings.TrimPrefix(match, "//"))
			return fmt.Sprintf("/* %s */", formatComment(comment))
		})

		// Handle trailing comments
		if strings.Contains(line, "/*") && !strings.HasPrefix(line, "/*") {
			matches := trailingRegex.FindStringSubmatch(line)
			if len(matches) == 3 && matches[2] != "" {
				comment := formatComment(strings.Trim(matches[2], "/* "))
				// Add comment on a separate line and remove from original line
				output = append(output, fmt.Sprintf("/* %s */", comment))
				line = strings.TrimSpace(matches[1])
			}
		}

		output = append(output, line)
	}
	return strings.Join(output, "\n")
}

// Ensures comment format: capitalized first letter, and one space padding
func formatComment(comment string) string {
	comment = strings.TrimSpace(comment)
	if len(comment) == 0 {
		return ""
	}
	runes := []rune(comment)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}
