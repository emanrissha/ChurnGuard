import os
import anthropic
from dotenv import load_dotenv
from src.rag.prompts import SYSTEM_PROMPT, build_customer_prompt
from src.rag.retriever import build_customer_context

load_dotenv()


def ask_churnguard(customer_id: str, question: str) -> str:
    context = build_customer_context(customer_id)
    if not context:
        return f"Customer {customer_id} not found in the database."

    prompt = build_customer_prompt(context, question)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    df = load_raw_data()
    customer_id = df["customerID"].iloc[0]

    print(f"Testing with customer: {customer_id}\n")

    questions = [
        "Why is this customer at risk?",
        "למה הלקוח הזה בסיכון?",
        "What should the CS team do immediately?"
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = ask_churnguard(customer_id, q)
        print(f"A: {answer}")
        print("-" * 50)