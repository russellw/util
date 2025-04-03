# Define the input and output file paths
$inputFilePath = "input.csv"
$outputFilePath = "output.csv"

# Import the CSV file
$data = Import-Csv -Path $inputFilePath

# Iterate over each row in the data and make a change
foreach ($row in $data) {
    # Assuming there is a column named 'Number' to increment by 1
    $row.Number = [int]$row.Number + 1
}

# Export the modified data to a new CSV file
$data | Export-Csv -Path $outputFilePath -NoTypeInformation

Write-Host "CSV modification complete. Output saved to $outputFilePath"
