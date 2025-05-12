from fastapi import APIRouter
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from automind.process.ollama import ollama_stream, run_ollama

ollama_router = APIRouter()


class QueryModelRequest(BaseModel):
    prompt: str


@ollama_router.get("/test_llama")
async def test_llama():
    return StreamingResponse(ollama_stream("generate numpy python code 50 lines."))


@ollama_router.post("/query_llama")
async def query_llama(req: QueryModelRequest):
    res = {}

    buffer_list = run_ollama(req.prompt)

    res["result"] = buffer_list

    return ORJSONResponse(res)
