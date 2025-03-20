import openai
import asyncio


async def main():
    client = openai.AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant, if asked your name say Hello World."},
            {"role": "user", "content": "What is your name?"}
        ]
    )
    
    print(response.choices[0].message)


asyncio.run(main())
