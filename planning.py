import sys
import datetime
import openai
from tools import configs_array  


def complete_with_tools(messages, model="gpt-4o", tool_choice="auto", tools=None, store=False):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        tools=tools if tools else [],
        tool_choice=tool_choice,
    )
    return response


def main():
    goal = sys.argv[1] if len(sys.argv) > 1 else "I want to learn about building agents without a framework."
    
    prompt = f"""
    You are a helpful assistant working for a busy executive.
    Your tone is friendly but direct, they prefer short clear and direct writing.
    You try to accomplish the specific task you are given.
    You can use any of the tools available to you.
    Before you do any work you always make a plan using your Todo list.
    You can mark todos off on your todo list after they are complete.
    
    You summarize the actions you took by checking the done list then create a report.
    You always ask your assistant to checkGoalDone. If they say you are done you send the report to the user.
    If your assistant has feedback you add it to your todo list.
    
    Today is {datetime.datetime.now()}
    """
    
    completion = complete_with_tools(
        messages=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": goal}
        ],
        model="gpt-4o",
        tool_choice="auto",
        tools=configs_array,
        store=False
    )
    
    answer = completion["choices"][0]["message"]["content"]
    print("\n\n" + "#" * 40)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
