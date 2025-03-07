#!/usr/bin/env python3
import glob
import os

# Create obj directories if they don't exist
os.makedirs("obj", exist_ok=True)
os.makedirs("obj/tests", exist_ok=True)  # New directory for test object files

# Write the ninja build file
f = open("build.ninja", "w")

# Define compiler and flags
f.write("cxx = clang\n")
f.write(
    "cxxflags = -std=c++17 -c -ferror-limit=1 -Isrc -I\\boost -Werror -Wno-switch\n"
)
f.write("\n")

# Define the rule for C++ compilation
f.write("rule cxx\n")
f.write("  command = $cxx $cxxflags $in -o $out\n")
f.write("  description = CXX $out\n")
f.write("\n")

# Define the rule for linking
f.write("rule link\n")
f.write("  command = $cxx $in -o $out\n")
f.write("  description = LINK $out\n")
f.write("\n")

# Get all source files
src_files = glob.glob("src/*.cpp")
header_files = glob.glob("src/*.h")
test_files = glob.glob("unit-tests/*.cpp")  # Get all test files

# Create object file targets for each source file
obj_files = []
for src in src_files:
    obj = os.path.basename(src).replace(".cpp", ".o")
    obj_path = f"obj\\{obj}"
    obj_files.append(obj_path)
    # Each source file implicitly depends on headers through includes
    f.write(f"build {obj_path}: cxx {src}")
    # Add headers as order-only dependencies
    if header_files:
        f.write(" |")
        for header in header_files:
            f.write(f" {header}")
    f.write("\n")

# Add main.cpp compilation
main_obj = "obj\\main.o"
f.write(f"build {main_obj}: cxx main.cpp")
if header_files:
    f.write(" |")
    for header in header_files:
        f.write(f" {header}")
f.write("\n\n")

# Create the final executable target
f.write("build basic.exe: link")
for obj in obj_files:
    f.write(f" {obj}")
f.write(f" {main_obj}\n")

# Add a default target
f.write("\ndefault basic.exe\n\n")

# Create object file targets for each test file
test_obj_files = []
for test in test_files:
    test_obj = os.path.basename(test).replace(".cpp", ".o")
    test_obj_path = f"obj\\tests\\{test_obj}"  # Put test objects in obj/tests directory
    test_obj_files.append(test_obj_path)

    # Each test file compilation
    f.write(f"build {test_obj_path}: cxx {test}")
    if header_files:
        f.write(" |")
        for header in header_files:
            f.write(f" {header}")
    f.write("\n")

# Create the test executable target
f.write("\nbuild test.exe: link")
for obj in obj_files:  # Include all main source objects
    f.write(f" {obj}")
for test_obj in test_obj_files:  # Include all test objects
    f.write(f" {test_obj}")
f.write("\n")
