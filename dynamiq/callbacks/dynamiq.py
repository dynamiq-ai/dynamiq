from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.clients import DynamiqTracingClient


class DynamiqCallbackHandler(TracingCallbackHandler):
    client: DynamiqTracingClient | None = None

    def __init__(self, project_id: str, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.source_id = project_id
        if self.client is None:
            self.client = DynamiqTracingClient(api_key=api_key)
