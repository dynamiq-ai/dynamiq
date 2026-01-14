---
name: presentation_creator
version: "2.0.0"
description: Creates professional PowerPoint presentations with python-pptx
tags: [presentations, pptx, design]
dependencies:
  - python-pptx
---

# Presentation Creator Skill

## Quick Start with python-pptx

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def create_presentation(title, slides_data, output_path):
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    # Tech color palette
    PRIMARY = RGBColor(37, 99, 235)    # Blue #2563EB
    SECONDARY = RGBColor(124, 58, 237)  # Purple #7C3AED
    ACCENT = RGBColor(245, 158, 11)     # Amber #F59E0B
    DARK = RGBColor(30, 41, 59)         # Dark #1E293B

    # Title slide
    title_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_layout)
    title_shape = slide.shapes.title
    subtitle = slide.placeholders[1]
    title_shape.text = title
    subtitle.text = "Data Analysis Results"

    # Content slides
    for slide_info in slides_data:
        blank_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_layout)

        # Title
        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.5),
            Inches(9), Inches(0.8)
        )
        title_frame = title_box.text_frame
        title_frame.text = slide_info['title']
        p = title_frame.paragraphs[0]
        p.font.size = Pt(44)
        p.font.bold = True
        p.font.color.rgb = DARK

        # Content
        content_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(1.5),
            Inches(9), Inches(3.5)
        )
        content_frame = content_box.text_frame

        for item in slide_info.get('bullets', []):
            p = content_frame.add_paragraph()
            p.text = item
            p.font.size = Pt(20)
            p.font.color.rgb = DARK
            p.level = 0

    prs.save(output_path)
    return output_path
```

## Color Palettes

**Tech** (Modern, Professional):
- Primary: #2563EB (Electric Blue)
- Secondary: #7C3AED (Purple)
- Accent: #F59E0B (Amber)
- Dark: #1E293B (Slate)

**Corporate** (Traditional, Trustworthy):
- Primary: #1E40AF (Navy)
- Secondary: #059669 (Green)
- Accent: #DC2626 (Red)
- Dark: #0F172A (Dark Slate)
