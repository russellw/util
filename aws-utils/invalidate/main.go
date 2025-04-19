package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/cloudfront"
	"gopkg.in/ini.v1"
)

// searchConfigFile looks for aws.ini in the current directory or parent directories
func searchConfigFile(filename string) (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		configPath := filepath.Join(dir, filename)
		if _, err := os.Stat(configPath); err == nil {
			return configPath, nil
		}
		parentDir := filepath.Dir(dir)
		if parentDir == dir { // reached root directory
			break
		}
		dir = parentDir
	}
	return "", fmt.Errorf("%s not found", filename)
}

func main() {
	// Find aws.ini file
	configFile, err := searchConfigFile("aws.ini")
	if err != nil {
		log.Fatalf("Error finding aws.ini: %v", err)
	}

	// Load the configuration
	cfg, err := ini.Load(configFile)
	if err != nil {
		log.Fatalf("Failed to load config file: %v", err)
	}

	// Read values from the ini file
	awsAccessKeyID := cfg.Section("aws").Key("aws_access_key_id").String()
	awsSecretAccessKey := cfg.Section("aws").Key("aws_secret_access_key").String()
	distributionID := cfg.Section("aws").Key("cloudfront_distribution_id").String()

	if awsAccessKeyID == "" || awsSecretAccessKey == "" || distributionID == "" {
		log.Fatal("Missing required configuration values in aws.ini")
	}

	// Create a new AWS session
	sess, err := session.NewSession(&aws.Config{
		Credentials: credentials.NewStaticCredentials(awsAccessKeyID, awsSecretAccessKey, ""),
		Region:      aws.String("us-east-1"), // CloudFront requires us-east-1
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	// Create CloudFront client
	svc := cloudfront.New(sess)

	// Create the invalidation request
	invalidationInput := &cloudfront.CreateInvalidationInput{
		DistributionId: aws.String(distributionID),
		InvalidationBatch: &cloudfront.InvalidationBatch{
			CallerReference: aws.String(fmt.Sprintf("invalidation-%d", time.Now().Unix())),
			Paths: &cloudfront.Paths{
				Quantity: aws.Int64(1),
				Items:    []*string{aws.String("/*")},
			},
		},
	}

	// Send the invalidation request
	resp, err := svc.CreateInvalidation(invalidationInput)
	if err != nil {
		log.Fatalf("Failed to create CloudFront invalidation: %v", err)
	}

	fmt.Printf("Invalidation created: %s\n", *resp.Invalidation.Id)
}
