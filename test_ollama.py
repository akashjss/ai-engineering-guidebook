import ollama
import asyncio
import json

async def test():
    client = ollama.AsyncClient()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "capital": {"type": "string"},
            "languages": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "capital", "languages"]
    }
    
    print("Testing with schema in 'format'...")
    try:
        resp = await client.chat(
            model='llama3.1:latest',
            messages=[{'role': 'user', 'content': 'Tell me about Canada.'}],
            format=schema
        )
        print("Success!")
        print(resp.message.content)
    except Exception as e:
        print(f"Failed with schema: {e}")

    print("\nTesting with 'json' in 'format'...")
    try:
        resp = await client.chat(
            model='llama3.1:latest',
            messages=[{'role': 'user', 'content': 'Tell me about Canada in JSON.'}],
            format="json"
        )
        print("Success!")
        print(resp.message.content)
    except Exception as e:
        print(f"Failed with 'json': {e}")

if __name__ == "__main__":
    asyncio.run(test())

