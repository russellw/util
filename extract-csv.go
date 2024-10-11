package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strings"
	"unicode"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("Usage: %s <column_list> <input_file>", os.Args[0])
	}

	columnList := os.Args[1]
	inputFile := os.Args[2]

	// Open the CSV file
	file, err := os.Open(inputFile)
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	// Read the CSV file
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Failed to read CSV file: %v", err)
	}

	// Convert column letters to column indices (0-based)
	indices, err := parseColumnList(columnList)
	if err != nil {
		log.Fatalf("Invalid column list: %v", err)
	}

	// Iterate over rows and print extracted columns
	for _, row := range records {
		// Extract the requested columns
		var extracted []string
		for _, index := range indices {
			if index < len(row) {
				extracted = append(extracted, row[index])
			} else {
				extracted = append(extracted, "")
			}
		}

		// Check if all extracted columns are empty
		if allColumnsBlank(extracted) {
			continue
		}

		// Print the extracted row as CSV
		fmt.Println(strings.Join(extracted, ","))
	}
}

// parseColumnList converts the column list string (e.g., "CB") to column indices (0-based)
func parseColumnList(columnList string) ([]int, error) {
	var indices []int
	for _, letter := range columnList {
		if !unicode.IsLetter(letter) {
			return nil, fmt.Errorf("invalid character: %c", letter)
		}

		// Convert letter to 0-based index
		index := int(unicode.ToUpper(letter)) - 'A'
		indices = append(indices, index)
	}
	return indices, nil
}

// allColumnsBlank checks if all columns in the row are blank
func allColumnsBlank(columns []string) bool {
	for _, column := range columns {
		if column != "" {
			return false
		}
	}
	return true
}
