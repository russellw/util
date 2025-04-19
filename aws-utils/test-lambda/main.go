package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/lambda"
	"gopkg.in/ini.v1"
)

const configFile = "aws.ini"
const configSection = "lambda"

func findConfigFile() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		path := filepath.Join(dir, configFile)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", fmt.Errorf("config file %s not found", configFile)
}

func loadConfig() (string, string, error) {
	path, err := findConfigFile()
	if err != nil {
		return "", "", err
	}
	cfg, err := ini.Load(path)
	if err != nil {
		return "", "", err
	}
	section := cfg.Section(configSection)
	return section.Key("aws_access_key_id").String(), section.Key("aws_secret_access_key").String(), nil
}

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <LambdaFunctionName> <InputJSONFile>")
		os.Exit(1)
	}

	functionName := os.Args[1]
	jsonFile := os.Args[2]

	// Load AWS credentials
	accessKeyID, secretAccessKey, err := loadConfig()
	if err != nil {
		log.Fatalf("Failed to load AWS credentials: %v", err)
	}

	// Create a new AWS session
	sess, err := session.NewSession(&aws.Config{
		Credentials: credentials.NewStaticCredentials(accessKeyID, secretAccessKey, ""),
		Region:      aws.String("eu-west-1"), // Change the region if necessary
	})
	if err != nil {
		log.Fatalf("Failed to create AWS session: %v", err)
	}

	// Read the JSON file
	data, err := ioutil.ReadFile(jsonFile)
	if err != nil {
		log.Fatalf("Failed to read JSON file: %v", err)
	}

	// Create a payload map to wrap `body` as a JSON string
	var payload map[string]interface{}
	if err := json.Unmarshal(data, &payload); err != nil {
		log.Fatalf("Invalid JSON format: %v", err)
	}

	// Ensure `body` is a JSON string
	if body, ok := payload["body"]; ok {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			log.Fatalf("Failed to marshal body: %v", err)
		}
		payload["body"] = string(bodyBytes)
	}

	// Marshal the modified payload to JSON bytes
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatalf("Failed to marshal final payload: %v", err)
	}

	// Create Lambda service client
	svc := lambda.New(sess)

	// Invoke the Lambda function
	input := &lambda.InvokeInput{
		FunctionName: aws.String(functionName),
		Payload:      payloadBytes,
	}
	result, err := svc.InvokeWithContext(context.Background(), input)
	if err != nil {
		log.Fatalf("Failed to invoke Lambda function: %v", err)
	}

	// Print the response from Lambda
	fmt.Printf("Lambda Response:\n%s\n", string(result.Payload))
}
