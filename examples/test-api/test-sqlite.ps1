# Load SQLite Assembly
Add-Type -TypeDefinition @"
using System;
using System.Data.SQLite;
public class SQLiteHelper {
    public SQLiteConnection CreateConnection(string databasePath) {
        return new SQLiteConnection(String.Format("Data Source={0};Version=3;", databasePath));
    }
}
"@ -Language CSharp

# Create SQLite Helper Object
$helper = [SQLiteHelper]::new()

# Define database path
$databasePath = "C:\t\database.db"

# Create connection to SQLite database
$conn = $helper.CreateConnection($databasePath)
$conn.Open()

# Function to create table
function Create-Table {
    param (
        [SQLiteConnection]$connection
    )
    $cmd = $connection.CreateCommand()
    $cmd.CommandText = @"
CREATE TABLE IF NOT EXISTS People (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Age INTEGER NOT NULL
);
"@
    $cmd.ExecuteNonQuery() | Out-Null
    Write-Output "Table created or already exists."
}

# Function to insert data
function Insert-Data {
    param (
        [SQLiteConnection]$connection,
        [string]$name,
        [int]$age
    )
    $cmd = $connection.CreateCommand()
    $cmd.CommandText = "INSERT INTO People (Name, Age) VALUES (@name, @age)"
    $cmd.Parameters.Add((New-Object Data.SQLite.SQLiteParameter("@name", $name))) | Out-Null
    $cmd.Parameters.Add((New-Object Data.SQLite.SQLiteParameter("@age", $age))) | Out-Null
    $cmd.ExecuteNonQuery() | Out-Null
    Write-Output "Data inserted."
}

# Function to retrieve data
function Retrieve-Data {
    param (
        [SQLiteConnection]$connection
    )
    $cmd = $connection.CreateCommand()
    $cmd.CommandText = "SELECT * FROM People"
    $reader = $cmd.ExecuteReader()
    $results = @()
    while ($reader.Read()) {
        $results += [PSCustomObject]@{
            ID = $reader["ID"]
            Name = $reader["Name"]
            Age = $reader["Age"]
        }
    }
    $reader.Close()
    return $results
}

# Create the table
Create-Table -connection $conn

# Insert some data
Insert-Data -connection $conn -name "John Doe" -age 30
Insert-Data -connection $conn -name "Jane Doe" -age 25

# Retrieve and display data
$people = Retrieve-Data -connection $conn
$people | Format-Table -AutoSize

# Close the connection
$conn.Close()
