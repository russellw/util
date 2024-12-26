#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdlib>

// Helper functions
static float min(float r, float g, float b) {
    return std::min(r, std::min(g, b));
}

static float max(float r, float g, float b) {
    return std::max(r, std::max(g, b));
}

static void hsl_to_rgb(float h, float s, float l, float& r, float& g, float& b) {
    auto hue_to_rgb = [](float p, float q, float t) {
        if (t < 0.0f) t += 1.0f;
        if (t > 1.0f) t -= 1.0f;
        if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
        if (t < 1.0f / 2.0f) return q;
        if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
        return p;
    };

    if (s == 0.0f) {
        r = g = b = l; // Achromatic
    } else {
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        r = hue_to_rgb(p, q, h + 1.0f / 3.0f);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0f / 3.0f);
    }
}

static std::string rgba_to_hex(float r, float g, float b, float a) {
    auto to_hex = [](float value) {
        int int_value = static_cast<int>(std::round(value * 255.0f));
        std::ostringstream stream;
        stream << std::hex << std::setw(2) << std::setfill('0') << int_value;
        return stream.str();
    };

    return "#" + to_hex(r) + to_hex(g) + to_hex(b) + to_hex(a);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <H> <S> <L> <A>\n";
        std::cerr << "H: Hue [0, 360]\n";
        std::cerr << "S: Saturation [0, 1]\n";
        std::cerr << "L: Lightness [0, 1]\n";
        std::cerr << "A: Alpha [0, 1]\n";
        return 1;
    }

    // Parse command-line arguments
    float h = std::stof(argv[1]);
    float s = std::stof(argv[2]);
    float l = std::stof(argv[3]);
    float a = std::stof(argv[4]);

    // Normalize H to [0, 1]
    h = std::fmod(h, 360.0f) / 360.0f;
    if (h < 0.0f) h += 1.0f;

    // Convert HSL to RGB
    float r, g, b;
    hsl_to_rgb(h, s, l, r, g, b);

    // Convert to RGBA hex string
    std::string rgba_hex = rgba_to_hex(r, g, b, a);

    // Output result
    std::cout << rgba_hex << std::endl;

    return 0;
}
