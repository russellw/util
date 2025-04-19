package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"gopkg.in/ini.v1"
)

// findConfigFile looks for aws.ini in the current directory
// and then moves up the directory tree until it finds the file or reaches the root.
func findConfigFile(filename string) (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		configPath := filepath.Join(dir, filename)
		if _, err := os.Stat(configPath); err == nil {
			return configPath, nil
		}

		// Move up one directory
		parentDir := filepath.Dir(dir)
		if parentDir == dir { // If we've reached the root
			break
		}
		dir = parentDir
	}

	return "", fmt.Errorf("%s not found in any parent directories", filename)
}

func main() {
	// Look for aws.ini in current or parent directories
	configFile, err := findConfigFile("aws.ini")
	if err != nil {
		log.Fatalf("Failed to find config file: %v", err)
	}

	// Load the AWS configuration file
	cfg, err := ini.Load(configFile)
	if err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	// Retrieve AWS credentials from the config file
	awsAccessKeyID := cfg.Section("aws").Key("aws_access_key_id").String()
	awsSecretAccessKey := cfg.Section("aws").Key("aws_secret_access_key").String()

	// Create a new session with credentials
	sess, err := session.NewSession(&aws.Config{
		Region:      aws.String("eu-west-1"), // specify your region
		Credentials: credentials.NewStaticCredentials(awsAccessKeyID, awsSecretAccessKey, ""),
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	// Create an S3 service client
	s3Client := s3.New(sess)

	// List S3 buckets
	result, err := s3Client.ListBuckets(nil)
	if err != nil {
		log.Fatalf("Failed to list buckets: %v", err)
	}

	for _, bucket := range result.Buckets {
		fmt.Println(*bucket.Name)
	}
}
