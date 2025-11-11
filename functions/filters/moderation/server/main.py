from fastapi import FastAPI, Request
from pydantic import BaseModel

from llm_guard.input_scanners import Toxicity
from llm_guard.input_scanners.toxicity import MatchType

app = FastAPI(title="LLM Guard Toxicity Filter")

# Initialize the scanner once at startup (fast)
scanner = Toxicity(threshold=0.5, match_type=MatchType.SENTENCE)


class PromptForm(BaseModel):
    prompt: str


@app.post("/scan")
async def scan(data: PromptForm):
    sanitized_prompt, is_valid, risk_score = scanner.scan(data.prompt)
    return {
        "input": data.prompt,
        "sanitized_prompt": sanitized_prompt,
        "is_valid": is_valid,
        "risk_score": risk_score,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
