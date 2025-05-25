use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use aws_config::{Region, BehaviorVersion};
use aws_credential_types::Credentials;
use aws_sdk_route53::{Client, types::ResourceRecordSet};
use configparser::ini::Ini;
use anyhow::{Result, Context, anyhow};

/// Find the aws.ini file by walking up the directory tree
fn find_aws_ini_file() -> Result<PathBuf> {
    let mut current_dir = env::current_dir().context("Failed to get current directory")?;
    
    loop {
        let ini_path = current_dir.join("aws.ini");
        if ini_path.exists() {
            return Ok(ini_path);
        }
        
        // Move to parent directory
        if let Some(parent) = current_dir.parent() {
            current_dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    
    Err(anyhow!("aws.ini file not found"))
}

/// Get hosted zone ID by name
async fn get_hosted_zone_id_by_name(client: &Client, hosted_zone_name: &str) -> Result<String> {
    let mut marker: Option<String> = None;
    
    loop {
        let mut request = client.list_hosted_zones();
        if let Some(ref m) = marker {
            request = request.marker(m.clone());
        }
        
        let output = request.send().await
            .context("Error listing hosted zones")?;
        
        // Check each hosted zone
        if let Some(zones) = output.hosted_zones() {
            for zone in zones {
                if let Some(name) = zone.name() {
                    if name == hosted_zone_name {
                        if let Some(id) = zone.id() {
                            return Ok(id.to_string());
                        }
                    }
                }
            }
        }
        
        // Check if there are more results
        if !output.is_truncated().unwrap_or(false) {
            break;
        }
        
        marker = output.next_marker().map(|s| s.to_string());
    }
    
    Err(anyhow!("Hosted zone with name {} not found", hosted_zone_name))
}

/// Print resource record set information
fn print_record_set(record_set: &ResourceRecordSet) {
    let name = record_set.name().unwrap_or("N/A");
    let record_type = record_set.type_().map(|t| t.as_str()).unwrap_or("N/A");
    
    println!("Name: {}, Type: {}", name, record_type);
    
    // Print resource records
    if let Some(records) = record_set.resource_records() {
        if !records.is_empty() {
            for record in records {
                if let Some(value) = record.value() {
                    println!("  Value: {}", value);
                }
            }
        }
    }
    
    // Print alias target information
    if let Some(alias_target) = record_set.alias_target() {
        if let Some(dns_name) = alias_target.dns_name() {
            println!("  Alias Target: {}", dns_name);
        }
        if let Some(hosted_zone_id) = alias_target.hosted_zone_id() {
            println!("    Hosted Zone ID: {}", hosted_zone_id);
        }
        println!("    Evaluate Target Health: {}", 
                alias_target.evaluate_target_health().unwrap_or(false));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <hosted-zone-name>", args[0]);
        process::exit(1);
    }
    
    let mut hosted_zone_name = args[1].clone();
    
    // Ensure hosted zone name ends with a dot
    if !hosted_zone_name.ends_with('.') {
        hosted_zone_name.push('.');
    }
    
    // Find aws.ini file
    let aws_ini_path = find_aws_ini_file()
        .context("Error finding aws.ini file")?;
    
    // Parse aws.ini file
    let mut config = Ini::new();
    config.load(&aws_ini_path)
        .context("Error loading aws.ini file")?;
    
    let aws_access_key_id = config.get("route53", "aws_access_key_id")
        .ok_or_else(|| anyhow!("aws_access_key_id is missing in aws.ini"))?;
    
    let aws_secret_access_key = config.get("route53", "aws_secret_access_key")
        .ok_or_else(|| anyhow!("aws_secret_access_key is missing in aws.ini"))?;
    
    // Create AWS credentials
    let credentials = Credentials::new(
        aws_access_key_id,
        aws_secret_access_key,
        None, // session token
        None, // expiration
        "static" // provider name
    );
    
    // Create AWS config
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(credentials)
        .load()
        .await;
    
    // Create Route53 client
    let client = Client::new(&config);
    
    // Get hosted zone ID by name
    let hosted_zone_id = get_hosted_zone_id_by_name(&client, &hosted_zone_name).await
        .context("Error finding hosted zone ID")?;
    
    // List records in the hosted zone
    println!("Records in hosted zone {}:", hosted_zone_name);
    
    let mut start_record_name: Option<String> = None;
    let mut start_record_type: Option<String> = None;
    
    loop {
        let mut request = client.list_resource_record_sets()
            .hosted_zone_id(&hosted_zone_id);
        
        if let Some(ref name) = start_record_name {
            request = request.start_record_name(name);
        }
        
        if let Some(ref record_type) = start_record_type {
            request = request.start_record_type(record_type.parse().unwrap());
        }
        
        let output = request.send().await
            .context("Error listing resource record sets")?;
        
        // Print each record set
        if let Some(record_sets) = output.resource_record_sets() {
            for record_set in record_sets {
                print_record_set(record_set);
            }
        }
        
        // Check if there are more results
        if !output.is_truncated().unwrap_or(false) {
            break;
        }
        
        start_record_name = output.next_record_name().map(|s| s.to_string());
        start_record_type = output.next_record_type().map(|t| t.as_str().to_string());
    }
    
    Ok(())
}
