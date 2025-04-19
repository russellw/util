package main

import (
	"fmt"
	"log"
	"mime"
	"os"
	"path/filepath"
	"strings"

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

// determineContentType returns the MIME type based on file extension.
func determineContentType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		// If no extension, default to .html MIME type
		return "text/html; charset=utf-8"
	}
	mimeType := mime.TypeByExtension(ext)
	if mimeType == "" {
		// Default to binary if no MIME type is found
		mimeType = "application/octet-stream"
	}
	return mimeType
}

// getCacheControl determines the appropriate cache policy based on the file type.
func getCacheControl(filename string) string {
	imageExtensions := []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"}
	ext := strings.ToLower(filepath.Ext(filename))
	for _, imgExt := range imageExtensions {
		if ext == imgExt {
			return "public, max-age=31536000, immutable"
		}
	}
	return "no-cache, no-store, must-revalidate"
}

// uploadFile uploads the specified file to the S3 bucket with the correct content type and cache-control metadata.
func uploadFile(bucketName, fileName string, s3Client *s3.S3) error {
	// Open the file
	file, err := os.Open(fileName)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", fileName, err)
	}
	defer file.Close()

	// Determine the content type
	contentType := determineContentType(fileName)

	// Determine the cache policy
	cacheControl := getCacheControl(fileName)

	// Prepare the PutObjectInput
	input := &s3.PutObjectInput{
		Bucket:      aws.String(bucketName),
		Key:         aws.String(filepath.Base(fileName)),
		Body:        file,
		ContentType: aws.String(contentType),
		CacheControl: aws.String(cacheControl),
	}

	// Upload the file
	_, err = s3Client.PutObject(input)
	if err != nil {
		return fmt.Errorf("failed to upload file to bucket %s: %v", bucketName, err)
	}

	fmt.Printf("Successfully uploaded %s to bucket %s with content type %s\n", fileName, bucketName, contentType)
	fmt.Printf("Cache control applied: %s\n", cacheControl)
	return nil
}

func main() {
	// Validate arguments
	args := os.Args
	if len(args) < 3 {
		log.Fatal("Usage: go run upload_to_s3.go <bucket_name> <file_name>")
	}
	bucketName := args[1]
	fileName := args[2]

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

	// Upload the file
	err = uploadFile(bucketName, fileName, s3Client)
	if err != nil {
		log.Fatalf("Failed to upload file: %v", err)
	}
}
