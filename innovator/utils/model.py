import openai
import json
import tools


openai.api_key = 'your-api-key'


def complete_with_tools(args):
    print(f"Calling llm with: {json.dumps(args['messages'][-1])[:500]}")

    completion = openai.ChatCompletion.create(**args)

    if 'tool_calls' in completion['choices'][0]['message']:
        tool_calls = completion['choices'][0]['message']['tool_calls']

        args['messages'].append(completion['choices'][0]['message'])

        tool_resps = []
        for tool_call in tool_calls:
            tool_args = json.loads(tool_call['function']['arguments'])
            
            print("\n\n" + "#" * 40)
            print(f"tool_calling: {tool_call['function']['name']}({json.dumps(tool_args)})")
            
            # Assuming tools.functions is a dictionary of tool functions
            result = tools.functions[tool_call['function']['name']](tool_args)

            # Append result message
            args['messages'].append({
                'role': 'tool',
                'tool_call_id': tool_call['id'],
                'content': result
            })
        
        return complete_with_tools(args)

    print("\n\n" + "#" * 40)
    print(completion['choices'][0]['message']['content'][:500])
    return completion
