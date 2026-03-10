from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENAI_API_KEY" #change this to your own API KEY before running. 
)


def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    prompt = f"""
            You are an assistant that provides concise answers to questions based on the provided context.
            Answer the question using only the information from the retrieved chunks. 
            If the answer is not contained within the chunks, just summarise the context.
            Use the provided context to answer the question.

            Question:
            {query}

            Context:
            {context}

            Answer:
            """

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content
