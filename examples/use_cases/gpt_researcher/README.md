## Usage

### 1. **Set Up Environment Variables**
Before running the workflow, ensure you have the necessary API keys.

Add the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `TAVILY_API_KEY`: Your Tavily API key.
   - `ZENROWS_API_KEY`: Your ZenRows API key.

For serverless Pinecone connection:
   - `PINECONE_API_KEY`: Your Pinecone API key.
   - `PINECONE_CLOUD`: Your Pinecone cloud.
   - `PINECONE_REGION`: Your Pinecone region.


### 2. **Run the Workflow**

Choose between the two available research modes based on your requirements:

#### **Option 1: GPT Researcher (Concise Report ~ 2-3 Pages)**

Run the following command:
```bash
python main_gpt_researcher.py
```

#### **Option 2: Multi-Agent GPT Researcher (Comprehensive Report ~ 5-20 Pages)**
This mode uses an enhanced architecture based on the first one to create an in-depth research report.
The multi-agent system allows for deeper analysis, and expanded content generation.

Run the following command:
```bash
python main_gpt_researcher_multi.py
```
