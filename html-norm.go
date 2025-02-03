package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

var (
	// Regex for finding heading tags and their content
	headingRegex = regexp.MustCompile(`(<h[1-6]>)(.*?)(</h[1-6]>)`)
	// Regex for finding </li> tags
	liEndRegex = regexp.MustCompile(`</li>`)
	// Regex for finding <strong> tags
	strongRegex = regexp.MustCompile(`<strong(>|\s[^>]*>)`)
)

func main() {
	writeFlag := flag.Bool("w", false, "write result to source file instead of stdout")
	flag.Parse()

	var input io.Reader
	var inputPath string

	if flag.NArg() > 0 {
		inputPath = flag.Arg(0)
		file, err := os.Open(inputPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open file: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()
		input = file
	} else {
		input = os.Stdin
	}

	// Read entire input into memory
	content, err := io.ReadAll(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read input: %v\n", err)
		os.Exit(1)
	}

	// Normalize the content
	normalized := normalize(string(content))

	// Prepare output
	var output io.Writer
	if *writeFlag && inputPath != "" {
		file, err := os.Create(inputPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create output file: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()
		output = file
	} else {
		output = os.Stdout
	}

	// Write the result
	writer := bufio.NewWriter(output)
	_, err = writer.WriteString(normalized)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write output: %v\n", err)
		os.Exit(1)
	}
	writer.Flush()
}

func normalize(content string) string {
	// Replace <strong> with <b>
	content = strongRegex.ReplaceAllString(content, "<b$1")
	content = strings.ReplaceAll(content, "</strong>", "</b>")

	// Remove </li> tags
	content = liEndRegex.ReplaceAllString(content, "")

	// Convert heading text to sentence case
	content = headingRegex.ReplaceAllStringFunc(content, func(match string) string {
		parts := headingRegex.FindStringSubmatch(match)
		if len(parts) != 4 {
			return match
		}
		
		openTag := parts[1]
		text := parts[2]
		closeTag := parts[3]

		// Convert text to sentence case
		words := strings.Fields(text)
		if len(words) > 0 {
			// Capitalize first word
			words[0] = strings.Title(strings.ToLower(words[0]))
			// Convert rest to lower case
			for i := 1; i < len(words); i++ {
				words[i] = strings.ToLower(words[i])
			}
			text = strings.Join(words, " ")
		}

		return openTag + text + closeTag
	})

	return content
}
