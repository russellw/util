package main

import (
	"flag"
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

// findConfigFile searches for aws.ini in the current and parent directories.
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

// deleteFile deletes the specified file from the S3 bucket.
func deleteFile(bucketName, fileName string, s3Client *s3.S3) error {
	input := &s3.DeleteObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(fileName),
	}

	_, err := s3Client.DeleteObject(input)
	if err != nil {
		return fmt.Errorf("failed to delete file %s from bucket %s: %v", fileName, bucketName, err)
	}

	fmt.Printf("Successfully deleted %s from bucket %s\n", fileName, bucketName)
	return nil
}

func main() {
	flag.Parse()

	// Validate arguments
	args := flag.Args()
	if len(args) < 2 {
		log.Fatal("Usage: go run delete_from_s3.go <bucket_name> <file_name>")
	}
	bucketName := args[0]
	fileName := args[1]

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
		Region:      aws.String("eu-west-1"),
		Credentials: credentials.NewStaticCredentials(awsAccessKeyID, awsSecretAccessKey, ""),
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	// Create an S3 service client
	s3Client := s3.New(sess)

	// Delete the file
	err = deleteFile(bucketName, fileName, s3Client)
	if err != nil {
		log.Fatalf("Failed to delete file: %v", err)
	}
}
