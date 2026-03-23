from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from dynamiq.nodes.node import ConnectionNode


class GoogleCalendarBase(ConnectionNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = build("calendar", "v3", credentials=Credentials(token=self.connection.access_token))
