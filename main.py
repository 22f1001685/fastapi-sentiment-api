from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# AI Pipe client
# ===============================
client = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDE2ODVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.tMkhIuW5LJ3OJWCHKIFvD8J3Cv6k9VkQatCCRfFQYVs",
    base_url="https://aipipe.org/openai/v1"
)

# ===============================
# Request model
# ===============================
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

# ===============================
# Response model
# ===============================
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

# ===============================
# Structured output schema
# ===============================
sentiment_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

# ===============================
# POST /comment endpoint
# ===============================
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier. "
                        "Always return ONLY valid JSON with fields: "
                        "sentiment (positive/negative/neutral) and rating (1-5)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Classify the sentiment.

Rules:
- positive → rating 4 or 5
- neutral → rating 3
- negative → rating 1 or 2

Comment: {data.comment}
""",
                },
            ],
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {str(e)}"
        )

# ===============================
# Health check
# ===============================
@app.get("/")
def health():
    return {"status": "ok"}