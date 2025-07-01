import click

from dynamiq.cli.config import load_settings, save_settings


@click.group()
def config():
    """Manage CLI configurations"""
    pass


@config.command()
def configure():
    host = input("Enter API host (e.g. https://api.example.com): ")
    token = input("Enter API token: ")

    config = load_settings()
    config.api_host = host
    config.api_token = token
    save_settings(config)

    print("\n✅ Configuration saved to .dynamiq/config.json")
    print("These values will be used automatically when you run Dynamiq commands.")


@config.command("show")
def show_creds():
    config = load_settings()
    host = config.api_host or "<not set>"
    token = config.api_token or "<not set>"
    org_id = config.org_id or "<not set>"
    project_id = config.project_id or "<not set>"
    masked_token = token[:4] + "..." if token != "<not set>" else token  # nosec B105

    print("\nCurrent Dynamiq CLI configuration:")
    print(f"DYNAMIQ API HOST: {host}")
    print(f"DYNAMIQ API TOKEN: {masked_token}")
    print(f"DYNAMIQ ORG ID: {org_id}")
    print(f"DYNAMIQ PROJECT ID: {project_id}")
