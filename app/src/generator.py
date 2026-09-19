import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
logger = logging.getLogger(__name__)

def build_prompt(question: str, system_text:str, chunks: list[dict]) -> list[dict]:
  context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['chunk']}" for c in chunks)

  user_text = f"Context :{context} \nQuestion: {question} \nGet answer based on the context above."
  return [
      {"role": "system", "content": system_text},
       {"role": "user", "content": user_text},
]


def run_rag(
    messages, 
    generator:AutoModelForCausalLM, 
    g_tokenizer:AutoTokenizer, 
    device:torch.device, 
    max_new_tokens:int=512, 
    do_sample:bool=False):
  text = g_tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
  )
  input = g_tokenizer([text], return_tensors="pt").to(device)
  with torch.no_grad():
    output_ids = generator.generate(
        **input,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=None,
        top_p=None
        )
  content = output_ids[0][len(input["input_ids"][0]):].tolist()
  answer = g_tokenizer.decode(content, skip_special_tokens=True)
  return answer