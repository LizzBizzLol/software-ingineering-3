from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Русский текст для суммаризации")
    max_length: int = Field(256, ge=16, le=512)
    min_length: int = Field(30, ge=0, le=256)
    num_beams: int = Field(4, ge=1, le=8)

class SummarizeResponse(BaseModel):
    summary: str
    tokens_in: int
    tokens_out: int
