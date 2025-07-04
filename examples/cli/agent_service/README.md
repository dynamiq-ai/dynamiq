# Dynamiq Test Service

This is a FastAPI-based AI service built with Dynamiq framework. It integrates TogetherAI and Exa APIs for intelligent query processing.

---

## Setup and Deployment

Make sure you have the Dynamiq CLI installed and configured. You can set DYNAMIQ_API_KEY in env or call next command for manual config
```bash
dynamiq config
```

### 1. Set Organization ID
Before setting an organization, you can list all available ones:

```bash
dynamiq org list
```
Then set the one you want to use:

```bash
dynamiq org set --id <ORG_ID>
```

### 2. Set Project ID
After setting your organization, list its projects:

```bash
dynamiq project list
```
Then set your desired project:

```bash
dynamiq project set --id <PROJECT_ID>
```

### 3. Create a New Service

```bash
dynamiq service create --name <SERVICE_NAME>
```

### 4. Deploy the Service with Environment Variables

```bash
dynamiq service deploy  --id <service_id> --source <home_path>/dynamiq/examples/cli/agent_service --env TOGETHER_API_KEY <your_together_api_key> --env EXA_API_KEY <your_exa_api_key>
```
