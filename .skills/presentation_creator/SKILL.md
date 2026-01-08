---
name: presentation_creator
version: "2.0.0"
description: Creates professional presentations from HTML with templates, color palettes, and PPTX conversion. Use for slide decks, pitch decks, reports, and business presentations.
author: Dynamiq Team
tags: [presentations, pptx, html, design, templates, slides]
dependencies:
  - python-pptx
  - Pillow
  - lxml
  - beautifulsoup4
supporting_files:
  - templates/default_template.html
  - templates/pitch_deck_template.html
  - templates/corporate_template.html
  - scripts/html_to_pptx.py
  - scripts/color_palettes.py
---

# Presentation Creator Skill

## Overview

This skill enables creating professional PowerPoint presentations through a sophisticated HTML-first workflow. Design beautiful slides with HTML/CSS, apply brand guidelines and color palettes, then convert to PPTX format.

**Key Capabilities:**
- Design slides using HTML with full styling control
- Apply professional color palettes and themes
- Use templates for consistent branding
- Convert HTML to PPTX with proper formatting
- Support for charts, tables, images, and multimedia

## Decision Tree: Choose Your Workflow

### 1. Creating from Scratch
**When:** No existing template, full creative control needed
→ Follow the **HTML Design & Convert** workflow below

### 2. Using Templates
**When:** Brand guidelines exist, consistent look required
→ Follow the **Template-Based Creation** workflow below

### 3. Editing Existing PPTX
**When:** Modifying an existing presentation
→ Use python-pptx for direct XML manipulation

---

## Workflow 1: HTML Design & Convert

### Step 1: Design Color Palette

**CRITICAL:** Never use default colors. Design a sophisticated palette based on:
- Subject matter (tech = blues/grays, finance = greens/blues, creative = bold colors)
- Target audience (executive = conservative, startup = modern)
- Brand identity (if applicable)

**Example Palettes:**

```python
# Tech Startup Palette
TECH_STARTUP = {
    "primary": "#2563EB",      # Electric Blue
    "secondary": "#7C3AED",    # Purple
    "accent": "#F59E0B",       # Amber
    "dark": "#1E293B",         # Slate
    "light": "#F8FAFC"         # Off-white
}

# Executive/Corporate Palette
CORPORATE = {
    "primary": "#1E40AF",      # Navy Blue
    "secondary": "#059669",    # Emerald
    "accent": "#DC2626",       # Red
    "dark": "#0F172A",         # Dark Slate
    "light": "#F1F5F9"         # Light Gray
}

# Creative/Modern Palette
MODERN = {
    "primary": "#EC4899",      # Pink
    "secondary": "#8B5CF6",    # Violet
    "accent": "#14B8A6",       # Teal
    "dark": "#18181B",         # Zinc
    "light": "#FAFAFA"         # Near White
}
```

Use the `color_palettes.py` script to generate harmonious palettes:

```python
from scripts.color_palettes import generate_palette

# Generate palette from base color
palette = generate_palette(base_color="#2563EB", style="analogous")
# Returns: primary, secondary, accent, dark, light colors
```

### Step 2: Create HTML Slides

Design slides using semantic HTML with inline styles for full control:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Presentation Title</title>
    <style>
        .slide {
            width: 960px;
            height: 540px;
            position: relative;
            page-break-after: always;
            background: #FFFFFF;
            padding: 60px;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        }

        .slide-title {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
            color: white;
            padding: 0;
        }

        .slide-title h1 {
            font-size: 72px;
            font-weight: 700;
            margin: 0 0 20px 0;
        }

        .slide-title h2 {
            font-size: 32px;
            font-weight: 300;
            margin: 0;
        }

        .slide-content h2 {
            font-size: 48px;
            color: #1E293B;
            margin: 0 0 40px 0;
            border-bottom: 4px solid #2563EB;
            padding-bottom: 15px;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            height: calc(100% - 120px);
        }

        .bullet-list li {
            font-size: 24px;
            line-height: 1.6;
            margin-bottom: 15px;
            color: #334155;
        }

        .bullet-list li::before {
            content: "▸";
            color: #2563EB;
            font-weight: bold;
            margin-right: 10px;
        }

        .highlight-box {
            background: #F8FAFC;
            border-left: 5px solid #2563EB;
            padding: 20px 30px;
            margin: 20px 0;
        }

        .metric {
            text-align: center;
            padding: 30px;
        }

        .metric-value {
            font-size: 64px;
            font-weight: 700;
            color: #2563EB;
            line-height: 1;
        }

        .metric-label {
            font-size: 20px;
            color: #64748B;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <!-- Title Slide -->
    <div class="slide slide-title">
        <h1>Q4 Sales Performance</h1>
        <h2>Executive Summary | January 2025</h2>
    </div>

    <!-- Content Slide: Two Column -->
    <div class="slide slide-content">
        <h2>Key Highlights</h2>
        <div class="two-column">
            <div>
                <ul class="bullet-list">
                    <li>Revenue up 25% YoY</li>
                    <li>342 new customers acquired</li>
                    <li>94% retention rate</li>
                    <li>Expanded into 3 new markets</li>
                </ul>
            </div>
            <div>
                <div class="metric">
                    <div class="metric-value">$5.2M</div>
                    <div class="metric-label">Total Revenue</div>
                </div>
                <div class="metric">
                    <div class="metric-value">30%</div>
                    <div class="metric-label">Profit Margin</div>
                </div>
            </div>
        </div>
    </div>

    <!-- More slides... -->
</body>
</html>
```

**Design Guidelines:**

1. **Typography Hierarchy**
   - Title slides: 72pt heading, 32pt subtitle
   - Content slides: 48pt heading, 24pt body, 20pt details
   - Use extreme size contrast for impact (72pt vs 11pt)

2. **Layout Principles**
   - Prefer two-column layouts with full-width headers
   - Never stack charts below text vertically (side-by-side instead)
   - Leave generous whitespace (min 60px padding)
   - Use grid systems for alignment

3. **Visual Elements**
   - Full-bleed background images with text overlays
   - Geometric patterns for visual interest
   - Border treatments (thick borders, gradients)
   - Icon integration for bullet points

### Step 3: Convert HTML to PPTX

Use the `html_to_pptx.py` script for conversion:

```python
from scripts.html_to_pptx import HTMLToPPTXConverter

# Initialize converter
converter = HTMLToPPTXConverter()

# Convert HTML file to PPTX
converter.convert(
    html_path="presentation.html",
    output_path="presentation.pptx",
    slide_width=960,  # 16:9 ratio (960x540)
    slide_height=540,
    preserve_formatting=True
)

# Result: presentation.pptx with all slides
```

**The script handles:**
- CSS to PowerPoint formatting conversion
- Text styling (fonts, colors, sizes)
- Layout positioning
- Images and media
- Tables and lists
- Background colors and gradients

### Step 4: Validate Output

**MANDATORY VALIDATION STEPS:**

1. **Visual Check:** Generate thumbnail grid
```python
from scripts.html_to_pptx import generate_thumbnails
generate_thumbnails("presentation.pptx", "thumbnails/")
# Creates grid of all slides for quick visual review
```

2. **Content Check:** Verify all text rendered correctly
3. **Formatting Check:** Ensure colors, fonts, sizes match HTML
4. **Layout Check:** Confirm positioning and alignment

---

## Workflow 2: Template-Based Creation

### Step 1: Choose Template

Available templates:
- `default_template.html` - Clean, minimal design for any topic
- `pitch_deck_template.html` - Startup pitch deck with investor focus
- `corporate_template.html` - Professional corporate presentation

### Step 2: Customize Template

Read template file and modify:

```python
from dynamiq.nodes.tools import FileReadTool

# Read template
template = file_read(".skills/presentation_creator/templates/pitch_deck_template.html")

# Replace placeholders
presentation_html = template.replace("{{COMPANY_NAME}}", "Acme Inc")
presentation_html = presentation_html.replace("{{TAGLINE}}", "Building the Future")
presentation_html = presentation_html.replace("{{PRIMARY_COLOR}}", "#2563EB")

# Add custom slides to template
custom_slide = '''
<div class="slide slide-content">
    <h2>{{SLIDE_TITLE}}</h2>
    <div class="content">{{SLIDE_CONTENT}}</div>
</div>
'''
```

### Step 3: Convert and Validate

Same as Workflow 1, Step 3-4.

---

## Advanced Features

### Charts and Graphs

For data visualizations, generate charts separately and embed as images:

```python
import matplotlib.pyplot as plt
from python-pptx import Presentation
from python-pptx.util import Inches

# Generate chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['Q1', 'Q2', 'Q3', 'Q4'], [2.3, 3.1, 2.8, 4.2])
ax.set_title('Quarterly Revenue ($M)')
plt.savefig('chart.png', dpi=150, bbox_inches='tight')

# Add to slide programmatically after HTML conversion
prs = Presentation('presentation.pptx')
slide = prs.slides[2]  # Third slide
slide.shapes.add_picture('chart.png', Inches(1), Inches(2), width=Inches(8))
prs.save('presentation.pptx')
```

### Speaker Notes

Add speaker notes to slides:

```python
from python-pptx import Presentation

prs = Presentation('presentation.pptx')
slide = prs.slides[0]

# Add notes
notes_slide = slide.notes_slide
text_frame = notes_slide.notes_text_frame
text_frame.text = "Welcome everyone. This presentation covers Q4 results..."

prs.save('presentation.pptx')
```

### Animations (Post-Conversion)

Animations must be added via python-pptx after conversion:

```python
from python-pptx import Presentation
from python-pptx.enum.shapes import MSO_SHAPE_TYPE

prs = Presentation('presentation.pptx')
slide = prs.slides[1]

# Add entrance animation to shapes
for shape in slide.shapes:
    if shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
        # Animation API (simplified)
        shape.click_action.hyperlink.address = "ppaction://animate?effect=fadeIn"

prs.save('presentation.pptx')
```

---

## Templates Reference

### Pitch Deck Template Structure

```
1. Title Slide - Company name, tagline, logo
2. Problem - What problem are you solving?
3. Solution - Your product/service
4. Market Size - TAM, SAM, SOM
5. Product Demo - Screenshots/mockups
6. Business Model - How you make money
7. Traction - Metrics, growth, customers
8. Competition - Competitive landscape
9. Team - Founders and key hires
10. Financials - Projections, runway
11. Ask - Funding request and use of funds
12. Thank You - Contact information
```

### Corporate Template Structure

```
1. Title Slide - Presentation title, date, presenter
2. Agenda - Overview of topics
3. Executive Summary - Key points
4-N. Content Slides - Main material
N+1. Recommendations - Action items
N+2. Q&A - Questions slide
N+3. Thank You - Contact details
```

---

## Best Practices

### Color Psychology

- **Blue:** Trust, professionalism, technology (finance, tech, corporate)
- **Green:** Growth, money, health (finance, sustainability, wellness)
- **Red:** Energy, urgency, passion (sales, marketing, bold statements)
- **Purple:** Creativity, luxury, innovation (creative, premium products)
- **Orange:** Enthusiasm, affordability, friendliness (consumer, retail)

### Typography Rules

1. **Maximum 2 font families** per presentation
2. **Sans-serif for headings** (Helvetica, Arial, Segoe UI)
3. **Serif for body** (optional, for formal presentations)
4. **Minimum 18pt** for any text (24pt preferred for body)
5. **Line height 1.4-1.6** for readability

### Slide Composition

1. **Rule of thirds:** Position key elements at intersection points
2. **Z-pattern:** Eyes scan top-left → top-right → bottom-left → bottom-right
3. **Whitespace:** 30-40% of slide should be empty
4. **Hierarchy:** Most important info = largest, highest, leftmost

### Data Visualization

- **Bar charts:** Comparing quantities across categories
- **Line charts:** Showing trends over time
- **Pie charts:** Parts of a whole (max 5-6 slices)
- **Tables:** Precise numbers, detailed comparisons (avoid for trends)

**Never:**
- Use 3D charts (distorts data)
- Stack too many data series (max 3-4)
- Use rainbow colors (pick 2-3 from palette)

---

## Error Handling

Common issues and solutions:

### Issue: Text overflow
```python
# Solution: Reduce font size or split into multiple slides
if len(text) > 500:
    font_size = max(18, 24 - len(text) // 100)
```

### Issue: Images not appearing
```python
# Solution: Use absolute paths and verify file exists
import os
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
```

### Issue: Inconsistent formatting
```python
# Solution: Use CSS reset in HTML
"""
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
"""
```

---

## Complete Example

See `examples/create_pitch_deck.py` for a full workflow example creating a 12-slide pitch deck from scratch with custom branding, charts, and animations.

---

## Dependencies Installation

```bash
pip install python-pptx Pillow lxml beautifulsoup4 matplotlib
```

## Sources

Based on production-grade patterns from Anthropic's skills repository and modern presentation design principles.
