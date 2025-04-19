package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/route53"
	"gopkg.in/ini.v1"
)

func findAWSINIFile() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		filePath := filepath.Join(dir, "aws.ini")
		if _, err := os.Stat(filePath); err == nil {
			return filePath, nil
		}

		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			break
		}
		dir = parentDir
	}

	return "", fmt.Errorf("aws.ini file not found")
}

func getHostedZoneIDByName(r53 *route53.Route53, hostedZoneName string) (string, error) {
	input := &route53.ListHostedZonesInput{}

	for {
		output, err := r53.ListHostedZones(input)
		if err != nil {
			return "", fmt.Errorf("error listing hosted zones: %w", err)
		}

		for _, zone := range output.HostedZones {
			if aws.StringValue(zone.Name) == hostedZoneName {
				return aws.StringValue(zone.Id), nil
			}
		}

		if !aws.BoolValue(output.IsTruncated) {
			break
		}

		input.Marker = output.NextMarker
	}

	return "", fmt.Errorf("hosted zone with name %s not found", hostedZoneName)
}

func main() {
	// Parse command-line arguments
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <hosted-zone-name>")
	}
	hostedZoneName := os.Args[1]

	// Ensure hosted zone name ends with a dot
	if hostedZoneName[len(hostedZoneName)-1] != '.' {
		hostedZoneName += "."
	}

	// Locate the aws.ini file
	awsINIFile, err := findAWSINIFile()
	if err != nil {
		log.Fatalf("Error finding aws.ini file: %v", err)
	}

	// Parse the aws.ini file
	cfg, err := ini.Load(awsINIFile)
	if err != nil {
		log.Fatalf("Error loading aws.ini file: %v", err)
	}

	awsSection := cfg.Section("route53")
	awsAccessKeyID := awsSection.Key("aws_access_key_id").String()
	awsSecretAccessKey := awsSection.Key("aws_secret_access_key").String()

	if awsAccessKeyID == "" || awsSecretAccessKey == "" {
		log.Fatal("aws_access_key_id or aws_secret_access_key is missing in aws.ini")
	}

	// Create AWS session
	sess, err := session.NewSession(&aws.Config{
		Region:      aws.String("us-east-1"), // Adjust as needed
		Credentials: credentials.NewStaticCredentials(awsAccessKeyID, awsSecretAccessKey, ""),
	})
	if err != nil {
		log.Fatalf("Error creating AWS session: %v", err)
	}

	// Create Route 53 client
	r53 := route53.New(sess)

	// Get hosted zone ID by name
	hostedZoneID, err := getHostedZoneIDByName(r53, hostedZoneName)
	if err != nil {
		log.Fatalf("Error finding hosted zone ID: %v", err)
	}

	// List records in the hosted zone
	input := &route53.ListResourceRecordSetsInput{
		HostedZoneId: aws.String(hostedZoneID),
	}

	fmt.Printf("Records in hosted zone %s:\n", hostedZoneName)
	for {
		output, err := r53.ListResourceRecordSets(input)
		if err != nil {
			log.Fatalf("Error listing resource record sets: %v", err)
		}

		for _, recordSet := range output.ResourceRecordSets {
			fmt.Printf("Name: %s, Type: %s\n", aws.StringValue(recordSet.Name), aws.StringValue(recordSet.Type))
			if len(recordSet.ResourceRecords) > 0 {
				for _, record := range recordSet.ResourceRecords {
					fmt.Printf("  Value: %s\n", aws.StringValue(record.Value))
				}
			} else if recordSet.AliasTarget != nil {
				fmt.Printf("  Alias Target: %s\n", aws.StringValue(recordSet.AliasTarget.DNSName))
				fmt.Printf("    Hosted Zone ID: %s\n", aws.StringValue(recordSet.AliasTarget.HostedZoneId))
				fmt.Printf("    Evaluate Target Health: %t\n", aws.BoolValue(recordSet.AliasTarget.EvaluateTargetHealth))
			}
		}

		if !aws.BoolValue(output.IsTruncated) {
			break
		}

		input.StartRecordName = output.NextRecordName
		input.StartRecordType = output.NextRecordType
	}
}
