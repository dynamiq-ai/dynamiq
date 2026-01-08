import colorsys


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def rgb_to_hsl(rgb: tuple) -> tuple:
    """Convert RGB to HSL."""
    r, g, b = (x / 255.0 for x in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return (hue * 360, saturation * 100, lightness * 100)


def hsl_to_rgb(hsl: tuple) -> tuple:
    """Convert HSL to RGB."""
    hue = hsl[0] / 360.0
    saturation = hsl[1] / 100.0
    lightness = hsl[2] / 100.0
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_palette(base_color: str, style: str = "analogous") -> dict[str, str]:
    """
    Generate a color palette from a base color.

    Args:
        base_color: Base color in hex format (#RRGGBB)
        style: Palette style - "analogous", "complementary", "triadic", "monochromatic"

    Returns:
        Dictionary with primary, secondary, accent, dark, light colors
    """
    rgb = hex_to_rgb(base_color)
    hue, saturation, lightness = rgb_to_hsl(rgb)

    palette = {}

    if style == "analogous":
        # Colors adjacent on color wheel (±30 degrees)
        palette["primary"] = base_color
        palette["secondary"] = rgb_to_hex(hsl_to_rgb((hue + 30) % 360, saturation, lightness))
        palette["accent"] = rgb_to_hex(hsl_to_rgb((hue - 30) % 360, saturation, lightness))
        palette["dark"] = rgb_to_hex(hsl_to_rgb(hue, min(saturation + 10, 100), max(lightness - 30, 10)))
        palette["light"] = rgb_to_hex(hsl_to_rgb(hue, max(saturation - 20, 10), min(lightness + 40, 95)))

    elif style == "complementary":
        # Opposite on color wheel (180 degrees)
        palette["primary"] = base_color
        palette["secondary"] = rgb_to_hex(hsl_to_rgb((hue + 180) % 360, saturation, lightness))
        palette["accent"] = rgb_to_hex(hsl_to_rgb((hue + 150) % 360, saturation * 0.8, lightness))
        palette["dark"] = rgb_to_hex(hsl_to_rgb(hue, min(saturation + 10, 100), max(lightness - 30, 10)))
        palette["light"] = rgb_to_hex(hsl_to_rgb(hue, max(saturation - 20, 10), min(lightness + 40, 95)))

    elif style == "triadic":
        # Evenly spaced on color wheel (120 degrees)
        palette["primary"] = base_color
        palette["secondary"] = rgb_to_hex(hsl_to_rgb((hue + 120) % 360, saturation, lightness))
        palette["accent"] = rgb_to_hex(hsl_to_rgb((hue + 240) % 360, saturation, lightness))
        palette["dark"] = rgb_to_hex(hsl_to_rgb(hue, min(saturation + 10, 100), max(lightness - 30, 10)))
        palette["light"] = rgb_to_hex(hsl_to_rgb(hue, max(saturation - 20, 10), min(lightness + 40, 95)))

    elif style == "monochromatic":
        # Same hue, different saturation/lightness
        palette["primary"] = base_color
        palette["secondary"] = rgb_to_hex(hsl_to_rgb(hue, max(saturation - 20, 20), lightness))
        palette["accent"] = rgb_to_hex(hsl_to_rgb(hue, min(saturation + 20, 100), lightness))
        palette["dark"] = rgb_to_hex(hsl_to_rgb(hue, saturation, max(lightness - 30, 10)))
        palette["light"] = rgb_to_hex(hsl_to_rgb(hue, max(saturation - 30, 10), min(lightness + 40, 95)))

    return palette


# Predefined professional palettes
PROFESSIONAL_PALETTES = {
    "tech_startup": {
        "primary": "#2563EB",
        "secondary": "#7C3AED",
        "accent": "#F59E0B",
        "dark": "#1E293B",
        "light": "#F8FAFC"
    },
    "corporate": {
        "primary": "#1E40AF",
        "secondary": "#059669",
        "accent": "#DC2626",
        "dark": "#0F172A",
        "light": "#F1F5F9"
    },
    "creative": {
        "primary": "#EC4899",
        "secondary": "#8B5CF6",
        "accent": "#14B8A6",
        "dark": "#18181B",
        "light": "#FAFAFA"
    },
    "finance": {
        "primary": "#10B981",
        "secondary": "#0EA5E9",
        "accent": "#F59E0B",
        "dark": "#1F2937",
        "light": "#F9FAFB"
    },
    "healthcare": {
        "primary": "#06B6D4",
        "secondary": "#8B5CF6",
        "accent": "#F97316",
        "dark": "#0F172A",
        "light": "#F0F9FF"
    },
    "education": {
        "primary": "#3B82F6",
        "secondary": "#F59E0B",
        "accent": "#10B981",
        "dark": "#1E293B",
        "light": "#EFF6FF"
    },
    "modern_minimal": {
        "primary": "#18181B",
        "secondary": "#71717A",
        "accent": "#A1A1AA",
        "dark": "#09090B",
        "light": "#FAFAFA"
    },
    "bold_energy": {
        "primary": "#EF4444",
        "secondary": "#F97316",
        "accent": "#FBBF24",
        "dark": "#7F1D1D",
        "light": "#FEF2F2"
    }
}


def get_palette(name: str) -> dict[str, str]:
    """Get a predefined palette by name."""
    return PROFESSIONAL_PALETTES.get(name, PROFESSIONAL_PALETTES["corporate"])


def validate_contrast(fg_color: str, bg_color: str) -> float:
    """
    Calculate contrast ratio between two colors (WCAG standard).

    Args:
        fg_color: Foreground color hex
        bg_color: Background color hex

    Returns:
        Contrast ratio (1-21, higher is better)
    """
    def luminance(color_hex):
        rgb = [x / 255.0 for x in hex_to_rgb(color_hex)]
        rgb = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    l1 = luminance(fg_color)
    l2 = luminance(bg_color)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


# Example usage
if __name__ == "__main__":
    # Generate palette from base color
    palette = generate_palette("#2563EB", "analogous")
    print("Generated Palette:")
    for key, value in palette.items():
        print(f"  {key}: {value}")

    # Check contrast
    contrast = validate_contrast(palette["primary"], palette["light"])
    print(f"\nContrast ratio (primary on light): {contrast:.2f}")
    print(f"WCAG AA compliant: {contrast >= 4.5}")
