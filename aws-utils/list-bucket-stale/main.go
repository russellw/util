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

// compareFiles compares files in the S3 bucket with local files by last modified time.
func compareFiles(bucketName, localDir string, s3Client *s3.S3) error {
	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
	}

	// Iterate through all objects in the bucket
	err := s3Client.ListObjectsV2Pages(input, func(page *s3.ListObjectsV2Output, lastPage bool) bool {
		for _, object := range page.Contents {
			remoteKey := *object.Key
			remoteLastModified := *object.LastModified

			localPath := filepath.Join(localDir, remoteKey)
			localInfo, err := os.Stat(localPath)

			if err != nil {
				if os.IsNotExist(err) {
					fmt.Printf("Local file missing: %s\n", localPath)
				} else {
					fmt.Printf("Error checking local file %s: %v\n", localPath, err)
				}
				continue
			}

			// Compare last modified times
			if localInfo.ModTime().After(remoteLastModified) {
				fmt.Printf("Local file newer: %s\n", localPath)
			}
		}
		return !lastPage
	})

	return err
}

func main() {
	// Check if a bucket name and local directory were provided as arguments
	if len(os.Args) < 3 {
		log.Fatal("Usage: program <bucket-name> <local-directory>")
	}
	bucketName := os.Args[1]
	localDir := os.Args[2]

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

	// Compare S3 files with local files
	err = compareFiles(bucketName, localDir, s3Client)
	if err != nil {
		log.Fatalf("Error comparing files: %v", err)
	}
}
