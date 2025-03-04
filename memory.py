import openai
import asyncio
import os
import json
from pydantic import BaseModel
from serpapi import GoogleSearch


class DoneResponseSchema(BaseModel):
    done: bool


def search_google(query, location="Philadelphia, PA"):
    print("\n\n" + "#" * 40)
    print(f"Searching Google[{location}]: {query}\n\n")
    
    params = {
        "engine": "google",
        "api_key": os.getenv("SERP_API_KEY"),
        "q": query,
        "location": location
    }
    search = GoogleSearch(params)
    results = search.get_json()
    
    string_result = "\n\n".join([
        f"{res['title']}: {res['snippet']}\n{res['link']}"
        for res in results.get("organic_results", [])[:5]
    ])
    print(string_result)
    return string_result


async def complete_with_tools(client, args):
    completion = await client.chat.completions.create(**args)
    
    if completion.choices[0].message.tool_calls:
        tool_call = completion.choices[0].message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        
        print("\n\n" + "#" * 40)
        print(f"tool_calling: {tool_call.function.name}({json.dumps(tool_args)})")
        result = search_google(**tool_args)
        
        args["messages"].append(completion.choices[0].message)
        args["messages"].append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        return await complete_with_tools(client, args)
    
    print("\n\n" + "#" * 40)
    print(completion.choices[0].message.content)
    return completion


async def main():
    client = openai.AsyncOpenAI()
    prompt = "I want to buy a hoodie with a fur lined hood. It needs a full zipper. Near Times Square in NYC. Where can I buy one today at lunch time?"
    
    search_google_tool_config = {
        "type": "function",
        "function": {
            "name": "searchGoogle",
            "description": "Run a search query against Google for information from the web.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to send to Google."},
                    "location": {"type": ["string", "null"], "description": "What city to send the query from. This will affect localization and provide better information for location-specific queries."}
                },
                "required": ["query", "location"],
                "additionalProperties": False
            }
        }
    }
    
    completion = await complete_with_tools(client, {
        "messages": [{"role": "developer", "content": prompt}],
        "model": "gpt-4o",
        "tool_choice": "auto",
        "tools": [search_google_tool_config],
        "store": False
    })
    
    answer = completion.choices[0].message.content
    print("\n\n" + "#" * 40)
    print(f"Answer: {answer}")
    
    check_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "developer",
            "content": f"You are a strict critic. Given the following question, determine if the answer is a full answer to the question.\n\nQuestion: {prompt}\n\nAnswer: {answer}"
        }],
        response_format="json"
    )
    
    check = DoneResponseSchema.parse_raw(check_response.choices[0].message.content)
    print("\n\n" + "#" * 40)
    print(f"LLM as judge: {'👍' if check.done else '👎'}")


asyncio.run(main())
