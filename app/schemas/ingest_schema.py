from pydantic import BaseModel

class IngestRequest(BaseModel):
    folder_path: str