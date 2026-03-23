from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.node import ConnectionNode


class GoogleSheetsBase(ConnectionNode):
    connection: GoogleOAuth2 = GoogleOAuth2()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = build("sheets", "v4", credentials=Credentials(token=self.connection.access_token))

    def close(self):
        """Closes the client."""
        self.client.close()
