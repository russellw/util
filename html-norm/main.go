package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/net/html"
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

	// Parse HTML
	doc, err := html.Parse(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse HTML: %v\n", err)
		os.Exit(1)
	}

	// Normalize the document
	normalizeNode(doc)

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
	err = html.Render(output, doc)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to render HTML: %v\n", err)
		os.Exit(1)
	}
}

func normalizeNode(n *html.Node) {
	if n.Type == html.ElementNode {
		// Replace <strong> with <b>
		if n.Data == "strong" {
			n.Data = "b"
		}

		// Remove </li> by marking it as self-closing
		if n.Data == "li" {
			n.DataAtom = 0
		}

		// Convert heading text to sentence case
		if isHeading(n.Data) {
			normalizeHeadingText(n)
		}
	}

	// Recursively process child nodes
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		normalizeNode(c)
	}
}

func isHeading(tag string) bool {
	return len(tag) == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '6'
}

func normalizeHeadingText(n *html.Node) {
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if c.Type == html.TextNode {
			// Convert to sentence case
			words := strings.Fields(c.Data)
			if len(words) > 0 {
				// Capitalize first word
				words[0] = strings.Title(strings.ToLower(words[0]))
				// Convert rest to lower case
				for i := 1; i < len(words); i++ {
					words[i] = strings.ToLower(words[i])
				}
				c.Data = strings.Join(words, " ")
			}
		}
	}
}
