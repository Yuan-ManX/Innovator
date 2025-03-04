import openai
import asyncio
from pydantic import BaseModel


class DoneResponseSchema(BaseModel):
    done: bool


async def main():
    client = openai.AsyncOpenAI()
    prompt = "What is the average wing speed of a swallow?"
    
    print("\n\n" + "#" * 40)
    print(f"Question: {prompt}")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "developer", "content": prompt}],
        store=False
    )
    
    answer = response.choices[0].message.content
    print("\n\n" + "#" * 40)
    print(f"Answer: {answer}")
    
    check_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": f"You are a strict critic. Given the following question, determine if the answer is a full answer to the question.\n\nQuestion: {prompt}\n\nAnswer: {answer}"
            }
        ],
        response_format="json"
    )
    
    check = DoneResponseSchema.parse_raw(check_response.choices[0].message.content)
    print("\n\n" + "#" * 40)
    print(f"LLM as judge: {'👍' if check.done else '👎'}")


asyncio.run(main())
