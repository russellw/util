# Check-ZoneInformation.ps1
# Recursively checks files in the current directory for non-default zone information

# Zone identifiers and their meanings
$zoneDescriptions = @{
    '0' = 'Local Machine'
    '1' = 'Local Intranet'
    '2' = 'Trusted Sites'
    '3' = 'Internet'
    '4' = 'Restricted Sites'
}

function Get-ZoneIdentifier {
    param (
        [string]$FilePath
    )
    
    try {
        $zone = Get-Content -Path "${FilePath}:Zone.Identifier" -Stream Zone.Identifier -ErrorAction Stop
        $zoneId = ($zone | Select-String -Pattern "ZoneId=(\d+)").Matches.Groups[1].Value
        return $zoneId
    }
    catch {
        return $null
    }
}

function Format-ZoneInfo {
    param (
        [string]$ZoneId
    )
    
    if ($zoneDescriptions.ContainsKey($ZoneId)) {
        return "$ZoneId ($($zoneDescriptions[$ZoneId]))"
    }
    return "$ZoneId (Unknown)"
}

# Get all files recursively in the current directory
$files = Get-ChildItem -Recurse -File

$foundZoneInfo = $false

foreach ($file in $files) {
    $zoneId = Get-ZoneIdentifier -FilePath $file.FullName
    
    if ($zoneId -ne $null) {
        $foundZoneInfo = $true
        $formattedZone = Format-ZoneInfo -ZoneId $zoneId
        Write-Host "`nFile: $($file.FullName)"
        Write-Host "Zone: $formattedZone"
        
        # Get additional metadata from Zone.Identifier if available
        try {
            $zoneContent = Get-Content -Path "$($file.FullName):Zone.Identifier" -Stream Zone.Identifier -ErrorAction Stop
            $referrerUrl = ($zoneContent | Select-String -Pattern "ReferrerUrl=(.+)").Matches.Groups[1].Value
            $hostUrl = ($zoneContent | Select-String -Pattern "HostUrl=(.+)").Matches.Groups[1].Value
            
            if ($referrerUrl) {
                Write-Host "Referrer URL: $referrerUrl"
            }
            if ($hostUrl) {
                Write-Host "Host URL: $hostUrl"
            }
        }
        catch {
            # Ignore errors reading additional metadata
        }
    }
}

if (-not $foundZoneInfo) {
    Write-Host "`nNo files with non-default zone information were found."
}
