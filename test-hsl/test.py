import random
import subprocess
from colormath.color_objects import HSLColor, sRGBColor
from colormath.color_conversions import convert_color

# Path to the compiled Go binary
GO_PROGRAM_PATH = "hsla_to_rgba.exe"

def hsla_to_rgba_python(h, s, l, a):
    """Convert HSLA to RGBA using Python's colormath library."""
    hsl = HSLColor(h / 360.0, s / 100.0, l / 100.0)  # Normalize H, S, L
    rgb = convert_color(hsl, sRGBColor)
    r = int(rgb.clamped_rgb_r * 255)
    g = int(rgb.clamped_rgb_g * 255)
    b = int(rgb.clamped_rgb_b * 255)
    alpha = int(a * 255)
    return f"#{r:02x}{g:02x}{b:02x}{alpha:02x}"

def hsla_to_rgba_go(h, s, l, a):
    """Call the Go program to convert HSLA to RGBA."""
    result = subprocess.run(
        [GO_PROGRAM_PATH, str(h), str(s), str(l), str(a)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Go program failed: {result.stderr.strip()}")
    return result.stdout.strip()

def main():
    """Main testing loop."""
    for _ in range(1000):  # Adjust the number of iterations as needed
        # Generate random HSLA values
        h = random.uniform(0, 360)  # Hue: 0 to 360
        s = random.uniform(0, 100)  # Saturation: 0 to 100%
        l = random.uniform(0, 100)  # Lightness: 0 to 100%
        a = random.uniform(0, 1)    # Alpha: 0 to 1
        
        # Get results from both implementations
        rgba_python = hsla_to_rgba_python(h, s, l, a)
        rgba_go = hsla_to_rgba_go(h, s, l, a)

        # Compare results
        if rgba_python != rgba_go:
            print(f"Discrepancy found!")
            print(f"HSLA: ({h}, {s}, {l}, {a})")
            print(f"Python: {rgba_python}")
            print(f"Go:     {rgba_go}")
            break
    else:
        print("All tests passed successfully!")

if __name__ == "__main__":
    main()
