import os
import gradio as gr
import requests
import inspect
import torch
import pandas as pd
from langchain.prompts import PromptTemplate
from huggingface_hub import HfApi, login
from types import SimpleNamespace

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, FinalAnswerTool, TransformersModel, PythonInterpreterTool, VisitWebpageTool, SpeechToTextTool
from tools import (
    execute_python_file,
    search_web_links_only,
    whisper_transcribe,
    download_file,
    reverse_if_reversed,
    fetch_url,
    csv_tool,
    excel_tool,
    calculator,
    youtube_transcript_or_whisper,
    web_search, 
    wiki_search, 
    arvix_search, 
    add, 
    subtract, 
    multiply, 
    divide, 
    modulus,
)

# Verificar versión de transformers
try:
    import transformers
    print(f"📦 Versión de transformers: {transformers.__version__}")
    
    # Verificar si la clase problemática existe
    try:
        from transformers import AutoModelForImageTextToText
        print("✅ AutoModelForImageTextToText disponible")
    except ImportError:
        print("❌ AutoModelForImageTextToText NO disponible - actualización necesaria")
        
except ImportError:
    print("❌ Transformers no instalado")

# --- Environment ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MODEL_PATH = "/sandbox/home/sanchezj/.cache/agents/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4" #"/sandbox/home/sanchezj/.cache/agents/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4" #"/sandbox/home/sanchezj/.cache/agents/DeepSeek-R1-Distill-Qwen-32B"

# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):     
        # Establish agent
        model = TransformersModel(
            model_id=MODEL_PATH,
            device_map='auto',  
            torch_dtype='float16',         
            #max_new_tokens=1024,
            #do_sample=False,
            #temperature=0,
            #repetition_penalty=1.05,
            #pad_token_id=None,
            #eos_token_id=None,
        )

        tools=[
            #DuckDuckGoSearchTool(),
            FinalAnswerTool(),
            #PythonInterpreterTool(),
            #SpeechToTextTool(),
            #VisitWebpageTool(),
            execute_python_file,
            search_web_links_only,
            wiki_search,
            whisper_transcribe,
            arvix_search,
            download_file,
            reverse_if_reversed,
            fetch_url,
            youtube_transcript_or_whisper,
            csv_tool,
            excel_tool,
            calculator,
            add, 
            subtract, 
            multiply, 
            divide, 
            modulus,
        ]

        with open("system_prompt2.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()

        self.prompt = PromptTemplate(
                input_variables=["user_question"], template=prompt_template
        )

        self.agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=["exec", "open", "os", "sys", "json", "csv", "wikipedia"],
            max_steps=10,
            stream_outputs=True
        )
        
        print("AI Agent has been initialized initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        formatted_prompt = self.prompt.format(
            user_question=question,
            diff=""
        )
        answer = self.agent.run(question)

        # Check if the answer has the correct format: str, int, float
        if isinstance(answer, (list, tuple)):
            answer = ", ".join(map(str, answer))
        elif isinstance(answer, dict):
            answer = json.dumps(answer, ensure_ascii=False)
        elif answer is None:
            answer = ""
        
        print(f"Agent returning answer: {answer}")
        return answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        footer = f"\nThe task_id is {task_id}"
        question_text = question_text + footer

        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            print(f"Question {i+1}")
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = "juansanchezd-final-assignment-agent"
    space_id_startup = "https://huggingface.co/spaces/juansanchezd/Final_Assignment_Agent/tree/main"

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")

    token = os.getenv("HF_TOKEN")
    login(token)
    api = HfApi(token=token)
    username = "juansanchezd" 
    profile = SimpleNamespace(username=username, access_token=token)

    status_text, results_df = run_and_submit_all(profile)

    print("\n=== STATE ===")
    print(status_text)

    if results_df is not None:
        print("\n=== ANSWER FROM THE AGENT ===")
        with pd.option_context("display.max_rows", None):
            print(results_df)
        results_df.to_csv("agent_results.csv", index=False)
        print("\nResultados guardados en agent_results.csv")

    #demo.launch(
    #    server_name="0.0.0.0",
    #    server_port=8501,
    #    debug=True,
    #    share=False
    #)
