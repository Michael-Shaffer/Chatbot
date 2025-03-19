from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image
import pytesseract
from PIL import Image as PILImage
import pandas as pd

# Update model paths to use Hugging Face model IDs directly
MODEL_ID = "google/gemma-2b-it"  # A good default choice that's freely available

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def generate_response(context, question, model, tokenizer):

    template = ChatPromptTemplate([
        ("system", "You are a helpful AI bot, Your name is polaris. \n"),
        ("system", """Answer the users questions using this context without mentioning that you were given context. 
                     The context includes text, tables, and figures. When referring to tables or figures, 
                     be specific about their content but natural in your response.
                     
                     Context: {context}\n"""),
        ("human", "Using the provided context answer this question: {question}\n"),
        ("ai", "Answer:")
    ])

    # If context is a dictionary, format it properly
    if isinstance(context, dict):
        formatted_context = f"""
        Main Text: {' '.join(context['text'])}
        
        Tables: {' '.join(context['tables'])}
        
        Figures: {' '.join([f"Figure: {fig['caption']} - {fig['text']}" for fig in context['figures']])}
        """
    else:
        formatted_context = context

    prompt = template.invoke({"context": formatted_context, "question": question})
    inputs = tokenizer(prompt.to_string(), return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    stop_token_id = tokenizer.encode("<end_of_turn>")[0]

    outputs = model.generate(
        **inputs,
        min_length = 200,
        max_new_tokens = 256,
        num_beams = 1,
        do_sample = False,
        streamer = streamer,
        # eos_token_id = stop_token_id
    )

    # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # return response
