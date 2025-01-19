package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"time"
)

func main() {
	// Parse command line flags
	writeFlag := flag.Bool("w", false, "Overwrite the input file with the timestamped HTML")
	flag.Parse()

	var content []byte
	var err error

	// Check if a filename is provided or read from standard input
	if len(flag.Args()) == 1 {
		// Read the input file
		inputFile := flag.Args()[0]
		content, err = ioutil.ReadFile(inputFile)
		if err != nil {
			panic(err)
		}
	} else if len(flag.Args()) == 0 {
		// Read from standard input
		content, err = io.ReadAll(os.Stdin)
		if err != nil {
			panic(err)
		}
	} else {
		panic("Please provide at most one HTML file as an argument")
	}

	// Get the current timestamp in ISO 8601 format
	timestamp := time.Now().Format("2006-01-02 15:04:05")

	// Find the closing </style> tag to insert the timestamp after it
	html := string(content)
	styleEndIndex := strings.Index(html, "</style>")
	if styleEndIndex == -1 {
		panic("No </style> tag found in the HTML")
	}

	// Insert the timestamp after the </style> tag
	timestampedHTML := html[:styleEndIndex+8] + timestamp + "<br>" + html[styleEndIndex+8:]

	// Handle output
	if len(flag.Args()) == 1 && *writeFlag {
		// Overwrite the file if -w flag is provided
		inputFile := flag.Args()[0]
		err = ioutil.WriteFile(inputFile, []byte(timestampedHTML), 0644)
		if err != nil {
			panic(err)
		}
	} else {
		// Print the updated HTML to standard output
		fmt.Print(timestampedHTML)
	}
}
