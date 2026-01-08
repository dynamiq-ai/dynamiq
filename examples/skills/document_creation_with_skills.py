"""
Example: Using Skills to Create Documents with an Agent

This example demonstrates:
1. Creating a FileStore and uploading a skill
2. Creating an agent with skills enabled
3. Using the agent to discover and load skills
4. Creating a document using the loaded skill
"""

import os
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import PythonCodeExecutor
from dynamiq.storages.file import InMemoryFileStore, FileStoreConfig

# Set your OpenAI API key
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")


def upload_document_creator_skill(file_store: InMemoryFileStore):
    """Upload the document_creator skill to FileStore.

    Args:
        file_store: FileStore instance to upload skill to
    """
    # Skill content with YAML frontmatter and markdown
    skill_content = """---
name: document_creator
version: "1.0.0"
description: Creates professional PDF documents using Python libraries
tags: [documents, pdf, reports]
dependencies:
  - reportlab
---

# Document Creator Skill

## Overview
This skill enables creating professional PDF documents using reportlab library.

## Instructions

When creating documents:

1. Use the PythonCodeExecutor tool to run document generation code
2. Save outputs to FileStore using `write_file()` helper
3. Return the file path so the user can access the generated document

## Example: Creating a PDF Report

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def run(params=None, helpers=None, **context):
    \"\"\"Generate a PDF report.

    Args:
        params: Dictionary with:
            - title: Report title
            - sections: List of sections with title and content

    Returns:
        dict: Result with file_path
    \"\"\"
    title = params.get("title", "Report")
    sections = params.get("sections", [])

    # Create PDF
    pdf_path = "report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, title)

    # Sections
    y_position = height - 160
    for section in sections:
        # Section title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, section.get("title", ""))
        y_position -= 30

        # Section content
        c.setFont("Helvetica", 11)
        content_lines = section.get("content", "").split('\\n')
        for line in content_lines:
            if y_position < 100:
                c.showPage()
                y_position = height - 100
            c.drawString(100, y_position, line[:80])
            y_position -= 20

        y_position -= 10

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

## Error Handling

Always wrap document generation in try-except blocks:

```python
try:
    # Document generation code
    return {"status": "success", "file_path": "..."}
except Exception as e:
    return {"status": "error", "message": f"Failed: {str(e)}"}
```
"""

    # Upload skill to FileStore
    file_store.store(".skills/document_creator/SKILL.md", skill_content.encode('utf-8'))
    print("✓ Uploaded document_creator skill to FileStore")


def main():
    """Main example function."""
    print("=" * 60)
    print("Skills Example: Document Creation with Agent")
    print("=" * 60)

    # Step 1: Create FileStore
    print("\n1. Creating FileStore...")
    file_store = InMemoryFileStore()

    # Step 2: Upload skill to FileStore
    print("\n2. Uploading document_creator skill to FileStore...")
    upload_document_creator_skill(file_store)

    # Step 3: Create agent with skills enabled
    print("\n3. Creating agent with skills enabled...")
    agent = Agent(
        name="DocumentAgent",
        llm=OpenAI(model="gpt-4o"),
        tools=[
            PythonCodeExecutor(file_store=file_store)
        ],
        file_store=FileStoreConfig(
            enabled=True,
            backend=file_store,
            agent_file_write_enabled=True
        ),
        skills_enabled=True,  # Enable skills support
        role="You are a helpful assistant that creates professional documents."
    )
    print(f"✓ Created agent: {agent.name}")
    print(f"✓ Skills enabled: {agent.skills_enabled}")
    print(f"✓ Available tools: {[tool.name for tool in agent.tools]}")

    # Step 4: Agent discovers skills
    print("\n4. Testing skill discovery...")
    print("\nAsking agent to list available skills...")

    list_result = agent.run(input_data={
        "input": "What skills do you have available? Use the SkillsTool to list them."
    })

    print("\nAgent response:")
    print("-" * 60)
    print(list_result.get("output", {}).get("content", "No response"))
    print("-" * 60)

    # Step 5: Agent loads and uses skill to create document
    print("\n5. Creating a document using the skill...")
    print("\nAsking agent to create a Q4 sales report...")

    create_result = agent.run(input_data={
        "input": """Please create a professional PDF report about Q4 2024 Sales Performance.

The report should include these sections:
- Executive Summary: Q4 showed 25% growth with $5.2M revenue
- Key Metrics: 342 new customers, 94% retention rate
- Regional Performance: North America $2.8M, Europe $1.6M, Asia $0.8M
- Recommendations: Continue investment in top regions

Use the document_creator skill to generate this PDF."""
    })

    print("\nAgent response:")
    print("-" * 60)
    print(create_result.get("output", {}).get("content", "No response"))
    print("-" * 60)

    # Step 6: Check generated files
    print("\n6. Checking generated files in FileStore...")
    all_files = file_store.list_files()
    print(f"\nFiles in FileStore ({len(all_files)} total):")
    for file_path in all_files:
        print(f"  - {file_path}")

    # Check if report was created
    report_path = "reports/output.pdf"
    if file_store.exists(report_path):
        report_content = file_store.retrieve(report_path)
        print(f"\n✓ Report created successfully!")
        print(f"  Path: {report_path}")
        print(f"  Size: {len(report_content)} bytes")

        # Optionally save to local filesystem
        with open("q4_sales_report.pdf", "wb") as f:
            f.write(report_content)
        print(f"  Saved to local file: q4_sales_report.pdf")
    else:
        print(f"\n✗ Report not found at {report_path}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
        print("⚠ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        exit(1)

    main()
