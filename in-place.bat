@echo off
setlocal

:: Check for minimum arguments
if "%2"=="" (
    echo Usage: %0 file filter-program [arg1] [arg2] [...]
    exit /b 1
)

:: Variables
set file=%1
set filter_program=%2

:: Temporary file
set temp_file=%TEMP%\1

:: Check if the file exists
if not exist %file% (
    echo Error: File %file% does not exist.
    exit /b 1
)

:: Run the filter program, redirecting input and output
%filter_program% %3 %4 %5 < %file% > %temp_file%
if errorlevel 1 (
    echo Error: Filter program failed.
    exit /b 1
)

:: Backup the original file
move %file% %TEMP%
if errorlevel 1 (
    echo Error: Failed to back up the original file.
    exit /b 1
)

:: Replace the original file with the new content
move %temp_file% %file%

endlocal
exit /b 0
