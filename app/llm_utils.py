import os
import json
import hashlib
import asyncio
import random

async def safe_llm_batch_async(llm, prompts, timeout=90):
    """Run multiple LLM calls concurrently with timeout handling."""
    tasks = [llm.ainvoke(prompt) for prompt in prompts]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print("Timeout reached, skipping this batch.")
        return ["[]" for _ in prompts]
