package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <ApiUrl> <InputJSONFile>")
		os.Exit(1)
	}

	apiUrl := os.Args[1]
	jsonFile := os.Args[2]

	// Read the JSON file
	data, err := ioutil.ReadFile(jsonFile)
	if err != nil {
		log.Fatalf("Failed to read JSON file: %v", err)
	}

	// Create HTTP POST request
	req, err := http.NewRequest("POST", apiUrl, bytes.NewBuffer(data))
	if err != nil {
		log.Fatalf("Failed to create HTTP request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// Read and print the response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read response body: %v", err)
	}

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Body:\n%s\n", string(body))
}
