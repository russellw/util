package main

import (
	"fmt"
	"log"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"time"

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

// listS3Objects retrieves the list of objects in the specified S3 bucket.
func listS3Objects(bucketName string, s3Client *s3.S3) (map[string]time.Time, error) {
	objects := make(map[string]time.Time)

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
	}

	err := s3Client.ListObjectsV2Pages(input, func(page *s3.ListObjectsV2Output, lastPage bool) bool {
		for _, obj := range page.Contents {
			if obj.Key != nil && obj.LastModified != nil {
				objects[*obj.Key] = *obj.LastModified
			}
		}
		return true
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list objects in bucket %s: %v", bucketName, err)
	}

	return objects, nil
}

// uploadFile uploads the specified file to the S3 bucket with the correct content type and cache-control metadata.
func uploadFile(bucketName, filePath, fileName string, s3Client *s3.S3) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filePath, err)
	}
	defer file.Close()

	contentType := determineContentType(fileName)
	cacheControl := getCacheControl(fileName)

	input := &s3.PutObjectInput{
		Bucket:       aws.String(bucketName),
		Key:          aws.String(fileName),
		Body:         file,
		ContentType:  aws.String(contentType),
		CacheControl: aws.String(cacheControl),
	}

	_, err = s3Client.PutObject(input)
	if err != nil {
		return fmt.Errorf("failed to upload file %s: %v", filePath, err)
	}

	fmt.Printf("Uploaded %s to bucket %s\n", fileName, bucketName)
	return nil
}

func main() {
	args := os.Args
	if len(args) < 3 {
		log.Fatal("Usage: go run upload_to_s3.go <bucket_name> <directory_path>")
	}
	bucketName := args[1]
	dirPath := args[2]

	configFile, err := findConfigFile("aws.ini")
	if err != nil {
		log.Fatalf("Failed to find config file: %v", err)
	}

	cfg, err := ini.Load(configFile)
	if err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	awsAccessKeyID := cfg.Section("aws").Key("aws_access_key_id").String()
	awsSecretAccessKey := cfg.Section("aws").Key("aws_secret_access_key").String()

	sess, err := session.NewSession(&aws.Config{
		Region:      aws.String("eu-west-1"),
		Credentials: credentials.NewStaticCredentials(awsAccessKeyID, awsSecretAccessKey, ""),
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	s3Client := s3.New(sess)

	s3Objects, err := listS3Objects(bucketName, s3Client)
	if err != nil {
		log.Fatalf("Failed to list S3 objects: %v", err)
	}

	err = filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		key := filepath.Base(path)
		lastModified, exists := s3Objects[key]

		if !exists || info.ModTime().Add(time.Hour).After(lastModified) {
			if err := uploadFile(bucketName, path, key, s3Client); err != nil {
				return err
			}
		} else {
			fmt.Printf("Skipping %s (already up-to-date)\n", key)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Error processing directory: %v", err)
	}
}
