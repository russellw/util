#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// Define interfaces for our statistics
interface FileStats {
  totalFiles: number;
  totalDirectories: number;
  totalSize: number;
  fileTypes: Record<string, number>;
  largestFile: { path: string; size: number };
  smallestFile: { path: string; size: number };
  oldestFile: { path: string; date: Date };
  newestFile: { path: string; date: Date };
  avgFileDepth: number;
  duplicates: { size: number; paths: string[] }[];
  depthDistribution: Record<number, number>;
  sizeDistribution: {
    "< 1KB": number;
    "1KB-100KB": number;
    "100KB-1MB": number;
    "1MB-10MB": number;
    "10MB-100MB": number;
    "100MB-1GB": number;
    "> 1GB": number;
  };
}

// Define options for directory scanning
interface ScanOptions {
  maxDepth?: number;
  excludePatterns?: RegExp[];
  detectDuplicates?: boolean;
}

// Initialize stats object
const stats: FileStats = {
  totalFiles: 0,
  totalDirectories: 0,
  totalSize: 0,
  fileTypes: {},
  largestFile: { path: "", size: 0 },
  smallestFile: { path: "", size: Number.MAX_SAFE_INTEGER },
  oldestFile: { path: "", date: new Date() },
  newestFile: { path: "", date: new Date(0) },
  avgFileDepth: 0,
  duplicates: [],
  depthDistribution: {},
  sizeDistribution: {
    "< 1KB": 0,
    "1KB-100KB": 0,
    "100KB-1MB": 0,
    "1MB-10MB": 0,
    "10MB-100MB": 0,
    "100MB-1GB": 0,
    "> 1GB": 0,
  },
};

// Map to track files by size for potential duplicate detection
const filesBySize: Map<number, string[]> = new Map();
// Map to track files by hash for confirmed duplicate detection
const filesByHash: Map<string, string[]> = new Map();

// Process command line arguments
const args = process.argv.slice(2);
const options: ScanOptions = {
  maxDepth: undefined,
  excludePatterns: [/node_modules/, /\.git/],
  detectDuplicates: true,
};

// Parse command line args
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--max-depth' && args[i + 1]) {
    options.maxDepth = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i] === '--exclude' && args[i + 1]) {
    options.excludePatterns = options.excludePatterns || [];
    options.excludePatterns.push(new RegExp(args[i + 1]));
    i++;
  } else if (args[i] === '--no-duplicates') {
    options.detectDuplicates = false;
  }
}

// Calculate file hash for duplicate detection
function calculateFileHash(filePath: string): string {
  const fileBuffer = fs.readFileSync(filePath);
  const hashSum = crypto.createHash('md5');
  hashSum.update(fileBuffer);
  return hashSum.digest('hex');
}

// Categorize file size
function categorizeSizeDistribution(size: number): void {
  if (size < 1024) {
    stats.sizeDistribution["< 1KB"]++;
  } else if (size < 102400) {
    stats.sizeDistribution["1KB-100KB"]++;
  } else if (size < 1048576) {
    stats.sizeDistribution["100KB-1MB"]++;
  } else if (size < 10485760) {
    stats.sizeDistribution["1MB-10MB"]++;
  } else if (size < 104857600) {
    stats.sizeDistribution["10MB-100MB"]++;
  } else if (size < 1073741824) {
    stats.sizeDistribution["100MB-1GB"]++;
  } else {
    stats.sizeDistribution["> 1GB"]++;
  }
}

// Convert bytes to human-readable format
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Main recursive scan function
function scanDirectory(dir: string, currentDepth: number = 0, totalDepth: number = 0): void {
  // Check if we've exceeded max depth
  if (options.maxDepth !== undefined && currentDepth > options.maxDepth) {
    return;
  }
  
  try {
    // Get list of files/dirs in current directory
    const items = fs.readdirSync(dir);
    
    // Process each item
    for (const item of items) {
      const itemPath = path.join(dir, item);
      
      // Check if item matches exclude patterns
      const shouldExclude = options.excludePatterns?.some(pattern => 
        pattern.test(itemPath)
      );
      
      if (shouldExclude) {
        continue;
      }
      
      try {
        const itemStat = fs.statSync(itemPath);
        
        if (itemStat.isDirectory()) {
          stats.totalDirectories++;
          stats.depthDistribution[currentDepth] = (stats.depthDistribution[currentDepth] || 0) + 1;
          scanDirectory(itemPath, currentDepth + 1, totalDepth + 1);
        } else if (itemStat.isFile()) {
          const fileSize = itemStat.size;
          const fileExt = path.extname(itemPath).toLowerCase() || '(no extension)';
          const modifiedDate = itemStat.mtime;
          
          // Update stats
          stats.totalFiles++;
          stats.totalSize += fileSize;
          stats.fileTypes[fileExt] = (stats.fileTypes[fileExt] || 0) + 1;
          stats.depthDistribution[currentDepth] = (stats.depthDistribution[currentDepth] || 0) + 1;
          
          // Update file size distribution
          categorizeSizeDistribution(fileSize);
          
          // Track for avg depth calculation
          totalDepth += currentDepth;
          
          // Update largest file info
          if (fileSize > stats.largestFile.size) {
            stats.largestFile = { path: itemPath, size: fileSize };
          }
          
          // Update smallest file info
          if (fileSize < stats.smallestFile.size) {
            stats.smallestFile = { path: itemPath, size: fileSize };
          }
          
          // Update oldest file info
          if (modifiedDate < stats.oldestFile.date) {
            stats.oldestFile = { path: itemPath, date: modifiedDate };
          }
          
          // Update newest file info
          if (modifiedDate > stats.newestFile.date) {
            stats.newestFile = { path: itemPath, date: modifiedDate };
          }
          
          // Track files by size for duplicate detection
          if (options.detectDuplicates) {
            if (!filesBySize.has(fileSize)) {
              filesBySize.set(fileSize, []);
            }
            filesBySize.get(fileSize)?.push(itemPath);
          }
        }
      } catch (err) {
        console.error(`Error processing ${itemPath}: ${err}`);
      }
    }
  } catch (err) {
    console.error(`Error reading directory ${dir}: ${err}`);
  }
}

// Main execution function
function main(): void {
  console.log("Starting directory scan...");
  const startTime = Date.now();
  
  // Current directory is the default
  const directoryToScan = process.cwd();
  
  // Start scanning
  scanDirectory(directoryToScan);
  
  // Calculate average file depth
  stats.avgFileDepth = stats.totalFiles > 0 ? stats.totalFiles / stats.totalFiles : 0;
  
  // Find duplicates if option is enabled
  if (options.detectDuplicates) {
    console.log("Checking for duplicate files...");
    
    // Only check files with the same size
    for (const [size, filePaths] of filesBySize.entries()) {
      if (filePaths.length > 1) {
        // We have multiple files with the same size - calculate hash to confirm duplicates
        for (const filePath of filePaths) {
          try {
            const hash = calculateFileHash(filePath);
            if (!filesByHash.has(hash)) {
              filesByHash.set(hash, []);
            }
            filesByHash.get(hash)?.push(filePath);
          } catch (err) {
            console.error(`Error hashing file ${filePath}: ${err}`);
          }
        }
      }
    }
    
    // Collect confirmed duplicates (files with same hash)
    for (const [hash, filePaths] of filesByHash.entries()) {
      if (filePaths.length > 1) {
        const firstFile = filePaths[0];
        const fileSize = fs.statSync(firstFile).size;
        stats.duplicates.push({
          size: fileSize,
          paths: filePaths
        });
      }
    }
    
    // Sort duplicates by size (largest first)
    stats.duplicates.sort((a, b) => b.size - a.size);
  }
  
  // Print results
  const endTime = Date.now();
  
  console.log("\n=============================================");
  console.log(`Directory Statistics for: ${directoryToScan}`);
  console.log("=============================================");
  
  console.log(`\nScan completed in ${(endTime - startTime) / 1000} seconds`);
  console.log(`Total Files: ${stats.totalFiles}`);
  console.log(`Total Directories: ${stats.totalDirectories}`);
  console.log(`Total Size: ${formatBytes(stats.totalSize)}`);
  
  console.log("\n--- File Types ---");
  const sortedTypes = Object.entries(stats.fileTypes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);
  
  for (const [ext, count] of sortedTypes) {
    const percentage = ((count / stats.totalFiles) * 100).toFixed(1);
    console.log(`${ext}: ${count} files (${percentage}%)`);
  }
  
  console.log("\n--- Size Distribution ---");
  for (const [sizeRange, count] of Object.entries(stats.sizeDistribution)) {
    if (count > 0) {
      const percentage = ((count / stats.totalFiles) * 100).toFixed(1);
      console.log(`${sizeRange}: ${count} files (${percentage}%)`);
    }
  }
  
  console.log("\n--- Depth Distribution ---");
  for (const [depth, count] of Object.entries(stats.depthDistribution).sort((a, b) => Number(a[0]) - Number(b[0]))) {
    console.log(`Depth ${depth}: ${count} items`);
  }
  
  console.log("\n--- Notable Files ---");
  console.log(`Largest File: ${stats.largestFile.path} (${formatBytes(stats.largestFile.size)})`);
  console.log(`Smallest File: ${stats.smallestFile.path} (${formatBytes(stats.smallestFile.size)})`);
  console.log(`Newest File: ${stats.newestFile.path} (${stats.newestFile.date.toISOString()})`);
  console.log(`Oldest File: ${stats.oldestFile.path} (${stats.oldestFile.date.toISOString()})`);
  
  if (options.detectDuplicates && stats.duplicates.length > 0) {
    console.log("\n--- Duplicate Files ---");
    console.log(`Found ${stats.duplicates.length} sets of duplicate files`);
    
    // Show top 5 largest duplicate sets
    for (let i = 0; i < Math.min(5, stats.duplicates.length); i++) {
      const dupSet = stats.duplicates[i];
      console.log(`\nDuplicate set #${i+1} - ${formatBytes(dupSet.size)} each, ${dupSet.paths.length} copies:`);
      for (const filePath of dupSet.paths.slice(0, 3)) {
        console.log(`  - ${filePath}`);
      }
      if (dupSet.paths.length > 3) {
        console.log(`  ... and ${dupSet.paths.length - 3} more`);
      }
    }
    
    // Calculate wasted space
    const wastedSpace = stats.duplicates.reduce((total, dupSet) => {
      return total + (dupSet.size * (dupSet.paths.length - 1));
    }, 0);
    
    console.log(`\nTotal space wasted by duplicates: ${formatBytes(wastedSpace)}`);
  }
}

// Run the program
main();