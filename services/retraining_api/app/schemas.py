from pydantic import BaseModel


class RetrainingResponse(BaseModel):
    status: str
    feedback_count: int
    candidate_version: str
    recommendation: str