import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.utils.logger import logger


AGENT_ROLE = """
You are a Senior Data Analyst and Presentation Expert with access to specialized skills.

Your capabilities:
- Analyze data using statistical methods (math_analytics_pipeline skill)
- Create professional presentations (presentation_creator skill)
- Execute Python code dynamically for data processing
- Generate comprehensive reports with visualizations

Always:
- FORMAT YOUR RESPONSES IN MARKDOWN
- Use skills when they match the task requirements
- Generate actionable insights from data
- Create visually appealing presentations with proper color schemes
"""

ANALYSIS_PROMPT = """
I need a complete analysis and presentation workflow:

1. First, analyze the sales data (sales_data.csv):
   - Calculate key metrics: total revenue, average monthly revenue, growth rate
   - Identify top performing months
   - Compare regional performance
   - Generate statistical insights

2. Then, create a professional PowerPoint presentation with:
   - Title slide: "Sales Performance Analysis 2024"
   - Slide 2: Key Metrics (total revenue, avg monthly, customers)
   - Slide 3: Performance Highlights (top months, growth trends)
   - Slide 4: Regional Analysis (if multiple regions)
   - Slide 5: Recommendations (data-driven suggestions)

Use the Tech color palette (blue/purple/amber) for a modern professional look.
Save the presentation as 'sales_analysis_2024.pptx'.
"""

SKILLS_DIR = Path(__file__).resolve().parents[3] / "dynamiq" / "skills" / "builtin" / ".skills"


def read_skill_file(skill_name: str) -> bytes:
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    return skill_path.read_bytes()


def upload_skills_to_filestore(file_store: InMemoryFileStore) -> None:
    """Upload skills to FileStore."""
    logger.info("Uploading skills to FileStore...")

    file_store.store(
        ".skills/math_analytics_pipeline/SKILL.md",
        read_skill_file("math_analytics_pipeline")
    )

    file_store.store(
        ".skills/presentation_creator/SKILL.md",
        read_skill_file("presentation_creator")
    )

    logger.info(" Uploaded math_analytics_pipeline skill")
    logger.info(" Uploaded presentation_creator skill")


def create_sample_sales_data(file_store: InMemoryFileStore) -> pd.DataFrame:
    """Create and upload sample sales data."""
    logger.info("Creating sample sales data...")

    np.random.seed(42)
    n_months = 12

    base_revenue = 450000
    trend = np.linspace(0, 100000, n_months)  # Upward trend
    seasonality = 50000 * np.sin(np.linspace(0, 2 * np.pi, n_months))
    noise = np.random.normal(0, 30000, n_months)

    revenue = base_revenue + trend + seasonality + noise

    data = {
        'month': pd.date_range('2024-01-01', periods=n_months, freq='ME').strftime('%Y-%m'),
        'revenue': revenue.astype(int),
        'customers': np.random.randint(280, 420, n_months),
        'region': np.random.choice(['North America', 'Europe', 'Asia'], n_months),
        'product_sales': np.random.randint(800, 1500, n_months)
    }

    df = pd.DataFrame(data)

    csv_content = df.to_csv(index=False)
    file_store.store("sales_data.csv", csv_content.encode('utf-8'))

    logger.info(f" Created sales data: {len(df)} months")
    return df


def create_agent(file_store: InMemoryFileStore, tracing_handler=None) -> Agent:
    """Create and configure the agent with skills and tools."""
    logger.info("Creating agent with skills...")

    python_tool = PythonCodeExecutor(
        name="python_executor",
        file_store=file_store,
    )

    llm = OpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=16000,
    )

    file_store_config = FileStoreConfig(
        enabled=True,
        backend=file_store,
        agent_file_write_enabled=True
    )

    agent = Agent(
        name="AnalysisAgent",
        llm=llm,
        tools=[python_tool],
        role=AGENT_ROLE,
        max_loops=15,
        inference_mode=InferenceMode.XML,
        file_store=file_store_config,
        skills_enabled=True,
    )

    logger.info(f" Agent created with {len(agent.tools)} tools")
    logger.info(f" Skills enabled: {agent.skills_enabled}")

    return agent


def run_workflow(
    agent: Agent,
    prompt: str,
    files: list[io.BytesIO] = None,
    tracing_handler=None
) -> tuple[str, dict]:
    """Run the analysis and presentation workflow."""
    try:
        logger.info("=" * 80)
        logger.info("Starting Workflow")
        logger.info("=" * 80)

        input_data = {"input": prompt}
        if files:
            input_data["files"] = files

        callbacks = [tracing_handler] if tracing_handler else None

        run_config = RunnableConfig(callbacks=callbacks) if callbacks else None
        result = agent.run(input_data=input_data, config=run_config)

        content = result.output.get("content", "")
        files_generated = result.output.get("files", {})

        logger.info("=" * 80)
        logger.info("Workflow Complete")
        logger.info("=" * 80)

        return content, files_generated

    except Exception as e:
        logger.error(f"Workflow error: {e}", exc_info=True)
        return f"Error: {str(e)}", {}


def save_agent_files(files: list[io.BytesIO] | dict, output_dir: str = "./skill_outputs") -> None:
    """Save files generated by agent."""
    if not files:
        logger.info("No files generated by agent")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("Generated Files")
    logger.info("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(files, list):
        logger.info(f"Saving {len(files)} file(s)...")

        for file_bytesio in files:
            try:
                file_name = getattr(file_bytesio, "name", f"file_{id(file_bytesio)}.bin")
                file_description = getattr(file_bytesio, "description", "Generated file")

                file_data = file_bytesio.read()
                file_bytesio.seek(0)

                output_path = os.path.join(output_dir, file_name)

                with open(output_path, "wb") as f:
                    f.write(file_data)

                logger.info(f" {file_name}")
                logger.info(f"  Size: {len(file_data):,} bytes")
                logger.info(f"  Path: {output_path}")
                logger.info(f"  Description: {file_description}")
                logger.info("")

            except Exception as e:
                logger.error(f"✗ Failed to save {getattr(file_bytesio, 'name', 'unknown')}: {e}")

    elif isinstance(files, dict):
        logger.info(f"Saving {len(files)} file(s)...")

        for file_path, file_content in files.items():
            try:
                file_name = file_path.split("/")[-1]
                output_path = os.path.join(output_dir, file_name)

                if isinstance(file_content, bytes):
                    file_data = file_content
                elif isinstance(file_content, str):
                    file_data = file_content.encode('utf-8')
                else:
                    file_data = str(file_content).encode('utf-8')

                with open(output_path, "wb") as f:
                    f.write(file_data)

                logger.info(f" {file_name}")
                logger.info(f"  Size: {len(file_data):,} bytes")
                logger.info(f"  Path: {output_path}")
                logger.info("")

            except Exception as e:
                logger.error(f"✗ Failed to save {file_path}: {e}")

    logger.info("=" * 80)


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Skills Workflow with UI Tracing")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n[1/6] Setting up FileStore...")
    file_store = InMemoryFileStore()

    print("\n[2/6] Uploading skills...")
    upload_skills_to_filestore(file_store)

    print("\n[3/6] Creating sample sales data...")
    sample_data = create_sample_sales_data(file_store)
    print("\nSample data preview:")
    print(sample_data.head().to_string())
    print(f"\nTotal revenue: ${sample_data['revenue'].sum():,.0f}")
    print(f"Average monthly: ${sample_data['revenue'].mean():,.0f}")

    print("\n[4/6] Setting up tracing...")
    tracing_handler = TracingCallbackHandler()

    print("\n[5/6] Creating agent with skills...")
    agent = create_agent(file_store, tracing_handler)

    print("\n[6/6] Running analysis and presentation workflow...")
    print("\nPrompt:")
    print("-" * 80)
    print(ANALYSIS_PROMPT)
    print("-" * 80)

    output, files = run_workflow(
        agent=agent,
        prompt=ANALYSIS_PROMPT,
        tracing_handler=tracing_handler
    )

    print("\n" + "=" * 80)
    print("Agent Output")
    print("=" * 80)
    print(output)
    print("=" * 80)

    save_agent_files(files, output_dir="./skill_outputs")

    print("\n" + "=" * 80)
    print("FileStore Contents")
    print("=" * 80)
    all_files = file_store.list_files(recursive=True)
    print(f"Total files in FileStore: {len(all_files)}")
    for file_info in sorted(all_files, key=lambda f: f.path):
        size = len(file_store.retrieve(file_info.path))
        print(f"  • {file_info.path} ({size:,} bytes)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Workflow Summary")
    print("=" * 80)
    print(" Skills uploaded to FileStore")
    print(" Sample sales data created")
    print(" Agent analyzed data using math_analytics_pipeline skill")
    print(" Agent created presentation using presentation_creator skill")
    print(f" Generated {len(files) if files else 0} output file(s)")
    print(" All files saved to ./skill_outputs/")
    print("=" * 80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 Complete! Check ./skill_outputs/ for generated files.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
