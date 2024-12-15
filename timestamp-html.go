package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"strings"
	"time"
)

func main() {
	// Parse command line flags
	writeFlag := flag.Bool("w", false, "Overwrite the input file with the timestamped HTML")
	flag.Parse()

	// Get the input file name from the command line arguments
	if len(flag.Args()) != 1 {
		panic("Please provide the HTML file as an argument")
	}
	inputFile := flag.Args()[0]

	// Read the input file
	content, err := ioutil.ReadFile(inputFile)
	if err != nil {
		panic(err)
	}

	// Get the current timestamp in ISO 8601 format
	timestamp := time.Now().Format("2006-01-02T15:04:05")

	// Find the closing </style> tag to insert the timestamp after it
	html := string(content)
	styleEndIndex := strings.Index(html, "</style>")
	if styleEndIndex == -1 {
		panic("No </style> tag found in the HTML")
	}

	// Insert the timestamp after the </style> tag
	timestampedHTML := html[:styleEndIndex+8] + timestamp + "<br>" + html[styleEndIndex+8:]

	// Print the timestamp to standard output
	fmt.Println("Timestamp inserted:", timestamp)

	// If the -w flag is provided, overwrite the file
	if *writeFlag {
		err = ioutil.WriteFile(inputFile, []byte(timestampedHTML), 0644)
		if err != nil {
			panic(err)
		}
	} else {
		// Otherwise, print the updated HTML to standard output
		fmt.Println(timestampedHTML)
	}
}
