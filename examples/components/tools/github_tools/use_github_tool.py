import os

from dynamiq.connections import GitHub
from dynamiq.nodes.tools.github.repos import ListUserRepositories

connection = GitHub(api_key=os.getenv("GITHUB_API_KEY"))

list_user_orgs = ListUserRepositories(connection=connection)

res = list_user_orgs.run(input_data={"username": "your-github-username"})

print(res)
