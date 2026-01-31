# src/generation.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GenerationEngine:
    def __init__(self, model_name="llama3-70b-8192"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_answer(self, question: str, context: str) -> str:
        """
        Takes the retrieved context (text + graph data) and generates a response.
        """
        system_prompt = f"""
        You are a specialized Financial AI Assistant. 
        Your task is to answer the user's question using the provided context.
        
        CRITICAL RULES:
        1. The context includes both TEXT CHUNKS and GRAPH RELATIONSHIPS. 
        2. Use the relationships (e.g., 'Jensen Huang IS_CEO_OF NVIDIA') to verify facts.
        3. If the context doesn't have the answer, say you don't know. 
        4. Cite your sources (e.g., 'According to the Audio Transcript...').

        CONTEXT:
        {context}
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"❌ Error in generation: {str(e)}"
        
# import os
# from azure.ai.inference import ChatCompletionsClient
# from azure.core.credentials import AzureKeyCredential

# class GenerationEngine:
#     def __init__(self):
#         self.client = ChatCompletionsClient(
#             endpoint="https://models.inference.ai.azure.com",
#             credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN"))
#         )

#     def answer(self, question: str, context: str) -> str:
#         system_prompt = """
#         You are a Senior Financial AI. Answer the user's question using the provided context.
        
#         STRUCTURE RULES:
#         1. Use the 'GRAPH INSIGHTS' to bridge facts between different sections.
#         2. Always cite your source (e.g., 'Per NVIDIA Q1-2025 PDF...').
#         3. If metrics are provided (e.g. Revenue: $26B), use the exact values.
#         4. If a 'Multi-hop' relationship is found, explain the connection.
#         """
        
#         response = self.client.complete(
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
#             ],
#             model="gpt-4o",
#             temperature=0.2
#         )
#         return response.choices[0].message.content