const fs = require('fs');
const path = require('path');

// Get the current directory
const currentDir = process.cwd();

// Read all files in the current directory
fs.readdir(currentDir, (err, files) => {
    if (err) {
        console.error('Error reading directory:', err);
        return;
    }

    console.log(`Files in directory: ${currentDir}\n`);
    
    // Process each file
    files.forEach((file, index) => {
        // Get file stats to determine if it's a directory or file
        const filePath = path.join(currentDir, file);
        const stats = fs.statSync(filePath);
        
        // Format the output
        const fileType = stats.isDirectory() ? 'Directory' : 'File';
        const fileSize = stats.isDirectory() ? '-' : `${stats.size} bytes`;
        
        console.log(`${index + 1}. ${file} (${fileType}, ${fileSize})`);
    });
});