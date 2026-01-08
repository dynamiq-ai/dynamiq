---
name: document_creator
version: "1.0.0"
description: Creates professional documents (PDFs, Word documents, presentations) using Python libraries
author: Dynamiq Team
tags: [documents, pdf, docx, presentations, reports]
dependencies:
  - reportlab
  - python-docx
  - python-pptx
supporting_files:
  - templates/report_template.html
  - examples/sample_report.py
---

# Document Creator Skill

## Overview
This skill enables creating professional documents in various formats using Python libraries. Use this skill when you need to generate reports, presentations, or documents programmatically.

## Available Libraries

### PDF Generation (reportlab)
- **reportlab**: Professional PDF generation library
- Create PDFs from scratch with text, images, tables, and charts
- Support for custom layouts and styling

### Word Documents (python-docx)
- **python-docx**: Microsoft Word document creation
- Add paragraphs, headings, tables, and images
- Apply styles and formatting

### PowerPoint Presentations (python-pptx)
- **python-pptx**: PowerPoint presentation creation
- Create slides with text, images, and shapes
- Support for charts and tables

## Instructions

When creating documents:

1. **Use PythonCodeExecutor tool** to run document generation code
2. **Access templates** via FileReadTool from `.skills/document_creator/templates/`
3. **Save outputs** to FileStore using the `write_file()` helper function
4. **Return file paths** so the user can access the generated documents

## Example: Creating a PDF Report

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def run(params=None, helpers=None, **context):
    """
    Generate a PDF report.

    Args:
        params: Dictionary with report parameters
            - title: Report title
            - content: Report content sections (list of dicts)
            - author: Author name
        helpers: File helpers (write_file, read_file, etc.)

    Returns:
        dict: Result with file_path
    """
    title = params.get("title", "Report")
    content = params.get("content", [])
    author = params.get("author", "Dynamiq")

    # Create PDF
    pdf_path = "report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, title)

    # Author
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, f"By: {author}")

    # Content sections
    y_position = height - 180
    c.setFont("Helvetica", 11)

    for section in content:
        section_title = section.get("title", "")
        section_text = section.get("text", "")

        # Section title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, section_title)
        y_position -= 30

        # Section text
        c.setFont("Helvetica", 11)
        text_lines = section_text.split('\n')
        for line in text_lines:
            if y_position < 100:  # New page if needed
                c.showPage()
                y_position = height - 100
            c.drawString(100, y_position, line[:80])  # Truncate long lines
            y_position -= 20

        y_position -= 10  # Extra space between sections

    c.save()

    # Save to FileStore
    with open(pdf_path, 'rb') as f:
        helpers['write_file']('reports/output.pdf', f.read())

    return {
        "status": "success",
        "file_path": "reports/output.pdf",
        "message": f"PDF report '{title}' created successfully"
    }
```

## Example: Creating a Word Document

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def run(params=None, helpers=None, **context):
    """
    Generate a Word document.

    Args:
        params: Dictionary with document parameters
            - title: Document title
            - sections: List of sections with headings and paragraphs
        helpers: File helpers

    Returns:
        dict: Result with file_path
    """
    title = params.get("title", "Document")
    sections = params.get("sections", [])

    # Create document
    doc = Document()

    # Add title
    title_para = doc.add_heading(title, level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add sections
    for section in sections:
        heading = section.get("heading", "")
        paragraphs = section.get("paragraphs", [])

        # Add heading
        doc.add_heading(heading, level=1)

        # Add paragraphs
        for para_text in paragraphs:
            para = doc.add_paragraph(para_text)
            para.style.font.size = Pt(11)

    # Save document
    docx_path = "document.docx"
    doc.save(docx_path)

    # Save to FileStore
    with open(docx_path, 'rb') as f:
        helpers['write_file']('documents/output.docx', f.read())

    return {
        "status": "success",
        "file_path": "documents/output.docx",
        "message": f"Word document '{title}' created successfully"
    }
```

## Example: Creating a PowerPoint Presentation

```python
from pptx import Presentation
from pptx.util import Inches, Pt

def run(params=None, helpers=None, **context):
    """
    Generate a PowerPoint presentation.

    Args:
        params: Dictionary with presentation parameters
            - title: Presentation title
            - slides: List of slides with titles and content
        helpers: File helpers

    Returns:
        dict: Result with file_path
    """
    title = params.get("title", "Presentation")
    slides_data = params.get("slides", [])

    # Create presentation
    prs = Presentation()

    # Title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title_shape = slide.shapes.title
    subtitle = slide.placeholders[1]

    title_shape.text = title
    subtitle.text = "Generated by Dynamiq"

    # Content slides
    bullet_slide_layout = prs.slide_layouts[1]

    for slide_data in slides_data:
        slide = prs.slides.add_slide(bullet_slide_layout)
        shapes = slide.shapes

        title_shape = shapes.title
        body_shape = shapes.placeholders[1]

        title_shape.text = slide_data.get("title", "")

        tf = body_shape.text_frame
        for bullet in slide_data.get("bullets", []):
            p = tf.add_paragraph()
            p.text = bullet
            p.level = 0

    # Save presentation
    pptx_path = "presentation.pptx"
    prs.save(pptx_path)

    # Save to FileStore
    with open(pptx_path, 'rb') as f:
        helpers['write_file']('presentations/output.pptx', f.read())

    return {
        "status": "success",
        "file_path": "presentations/output.pptx",
        "message": f"PowerPoint presentation '{title}' created successfully"
    }
```

## Supporting Files

Access supporting files via FileReadTool:
- `read_file('.skills/document_creator/templates/report_template.html')` - HTML template for reports
- `read_file('.skills/document_creator/examples/sample_report.py')` - Complete example

## Tips

1. **Always validate input parameters** before creating documents
2. **Handle errors gracefully** and return meaningful error messages
3. **Use context managers** (with statements) for file operations
4. **Test with sample data** before creating complex documents
5. **Consider page breaks** for long content in PDFs and Word documents
6. **Use appropriate fonts and sizes** for professional appearance

## Common Patterns

### Adding Tables to PDFs
```python
from reportlab.platypus import Table, TableStyle

data = [
    ['Header 1', 'Header 2', 'Header 3'],
    ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
    ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'],
]

table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
```

### Adding Images to Word Documents
```python
from docx import Document
from docx.shared import Inches

doc = Document()
doc.add_picture('image.png', width=Inches(4))
```

### Adding Charts to PowerPoint
```python
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

chart_data = CategoryChartData()
chart_data.categories = ['Q1', 'Q2', 'Q3', 'Q4']
chart_data.add_series('Sales', (10, 20, 30, 40))

slide = prs.slides.add_slide(prs.slide_layouts[5])
chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(2), Inches(2),
    Inches(6), Inches(4),
    chart_data
).chart
```

## Error Handling

Always wrap document generation in try-except blocks:

```python
try:
    # Document generation code
    return {"status": "success", "file_path": "..."}
except Exception as e:
    return {
        "status": "error",
        "message": f"Failed to generate document: {str(e)}"
    }
```
