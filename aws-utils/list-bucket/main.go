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

// listBucketFiles lists all files in the specified S3 bucket and prints their metadata.
func listBucketFiles(bucketName string, s3Client *s3.S3) error {
	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
	}

	// Iterate through all objects in the bucket
	err := s3Client.ListObjectsV2Pages(input, func(page *s3.ListObjectsV2Output, lastPage bool) bool {
		for _, object := range page.Contents {
			// Print basic file information
			fmt.Printf("Name: %s\n", *object.Key)
			fmt.Printf("Size: %d bytes\n", *object.Size)
			fmt.Printf("Last Modified: %v\n", *object.LastModified)
			fmt.Printf("Storage Class: %s\n", *object.StorageClass)

			// Retrieve additional metadata with HeadObject
			headInput := &s3.HeadObjectInput{
				Bucket: aws.String(bucketName),
				Key:    object.Key,
			}
			headOutput, err := s3Client.HeadObject(headInput)
			if err != nil {
				fmt.Printf("Error retrieving metadata for %s: %v\n", *object.Key, err)
				continue
			}

			// Print additional metadata
			if headOutput.ContentType != nil {
				fmt.Printf("Content Type: %s\n", *headOutput.ContentType)
			}
			if headOutput.ContentEncoding != nil {
				fmt.Printf("Content Encoding: %s\n", *headOutput.ContentEncoding)
			}
			if headOutput.CacheControl != nil {
				fmt.Printf("Cache Control: %s\n", *headOutput.CacheControl)
			}
			if headOutput.ETag != nil {
				fmt.Printf("ETag: %s\n", *headOutput.ETag)
			}
			if headOutput.ContentLanguage != nil {
				fmt.Printf("Content Language: %s\n", *headOutput.ContentLanguage)
			}
			if headOutput.Metadata != nil && len(headOutput.Metadata) > 0 {
				fmt.Println("Custom Metadata:")
				for key, value := range headOutput.Metadata {
					fmt.Printf("  %s: %s\n", key, value)
				}
			}
			fmt.Println("------")
		}
		return !lastPage
	})

	return err
}

func main() {
	// Check if a bucket name was provided as an argument
	if len(os.Args) < 2 {
		log.Fatal("Please provide the name of the S3 bucket as a command-line argument")
	}
	bucketName := os.Args[1]

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

	// List all files in the specified bucket with metadata
	err = listBucketFiles(bucketName, s3Client)
	if err != nil {
		log.Fatalf("Failed to list files in bucket %s: %v", bucketName, err)
	}
}
