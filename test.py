from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434",  # ton serveur local
    api_key="sk-xxx"  # peut être dummy si local
)

response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Bonjour, comment ça va ?"}
    ]
)

print(response.choices[0].message.content)