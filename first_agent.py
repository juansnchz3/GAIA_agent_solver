import torch, os
from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool
from tools import (
    whisper_transcribe,
    csv_tool,
    excel_tool,
    calculator,
    web_search, 
    wiki_search, 
    arvix_search, 
    add, 
    subtract, 
    multiply, 
    divide, 
    modulus,
)
from langchain.prompts import PromptTemplate


# ---- ENVIRONMENT ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH="/sandbox/home/sanchezj/.cache/agents/Qwen2.5-14B-Instruct-GPTQ-Int8"

# ---- CONFIGURATION ----
model = TransformersModel(
    model_id=MODEL_PATH,
    device_map={"":0},  
    torch_dtype=torch.float16, 
    do_sample=True,
    temperature=0.1
)

tools=[
    DuckDuckGoSearchTool(),
    FinalAnswerTool(),
    whisper_transcribe,
    csv_tool,
    excel_tool,
    calculator,
    wiki_search, 
    arvix_search, 
    add, 
    subtract, 
    multiply, 
    divide, 
    modulus,
]

# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

prompt = PromptTemplate(
        input_variables=["user_question"], template=prompt_template
)

agent = CodeAgent(
    tools=tools,
    model=model,
    planning_interval=6,
    max_steps=12,
    stream_outputs=True
)

if __name__ == '__main__':
    user_question="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."

    formatted_prompt = prompt.format(
        user_question=user_question,
        diff=""
    )
    
    agent.run(user_question)
