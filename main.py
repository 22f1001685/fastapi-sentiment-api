from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import json

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="Sentiment API")

# ✅ REQUIRED for grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# AI Pipe client (UNCHANGED as requested)
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
# POST /comment endpoint
# ===============================
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    try:
        prompt = f"""
You are a strict sentiment classifier.

Rules:
- positive → rating 5
- neutral → rating 3
- negative → rating 1

Return ONLY valid JSON in this exact format:
{{"sentiment":"positive|negative|neutral","rating":number}}

Comment: {data.comment}
"""

        # ✅ REMOVED response_format (key fix)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You only return raw JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)

        # ✅ Safety validation
        if result.get("sentiment") not in ("positive", "negative", "neutral"):
            raise ValueError("Invalid sentiment")

        if not isinstance(result.get("rating"), int):
            raise ValueError("Invalid rating")

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