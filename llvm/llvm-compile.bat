@echo off
setlocal enabledelayedexpansion

:: Check if source file was provided
if "%~1"=="" (
    echo Usage: %~nx0 source.c [optimization_level]
    echo Optimization levels: 0, 1, 2, 3, s, z
    exit /b 1
)

:: Set default optimization level if not specified
set "OPT_LEVEL=2"
if not "%~2"=="" set "OPT_LEVEL=%~2"

:: Verify optimization level is valid
echo %OPT_LEVEL%| findstr /r "^[0123sz]$" >nul
if errorlevel 1 (
    echo Invalid optimization level: %OPT_LEVEL%
    echo Valid levels are: 0, 1, 2, 3, s, z
    exit /b 1
)

:: Get filename without extension
set "SOURCE=%~1"
set "BASENAME=%~n1"

:: Check if clang is available
where clang >nul 2>nul
if errorlevel 1 (
    echo Error: clang not found in PATH
    echo Please install LLVM and ensure it's in your system PATH
    exit /b 1
)

:: Compile to LLVM IR with optimizations
echo Compiling %SOURCE% to LLVM IR with -O%OPT_LEVEL% optimization...
clang -S -emit-llvm -O%OPT_LEVEL% -o "%BASENAME%.ll" "%SOURCE%"

if errorlevel 1 (
    echo Compilation failed
    exit /b 1
) else (
    echo Successfully generated %BASENAME%.ll
)
