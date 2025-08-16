from fastapi import FastAPI, Request, UploadFile, File, Form,Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()
import json
import re
import ast, json, os

import requests
from typing import Optional
from datetime import datetime

from pydantic import BaseModel


class Query(BaseModel):
    question: str

from prompt_tools import PromptBuilder, PromptLogger, PromptValidator, PromptDebugger
# At the top of your script
SYSTEM_INSTRUCTION = """
    You are a Python coding assistant. Your task is to generate executable Python code only.
    Do not include markdown, explanations, comments, or formatting.

    Your code must:
    - Encapsulate logic in a function (e.g., scrape_and_analyze())
    - Always include a top-level execution block using:
        if __name__ == "__main__":
            result = <function_name>()
            print(result)

    Important:
    - Do NOT suppress errors using try-except blocks.
    - If an error occurs (e.g., missing table, invalid data), raise a Python exception with a clear message.
    - Do NOT return fallback values like "N/A" or "Error: Table not found".
    - Let errors propagate so they can be detected and corrected in future iterations.

    If plotting is involved:
    - Invert ranks on the x-axis for better readability.
    - Return the plot as a base64-encoded data URI.

    Only return valid Python code that can be run directly in a script

    
"""

def call_aipipe(prompt: str, model: str = "gpt-4.1-nano", temperature: float = 0.7, system_prompt: str = None) -> str:
    base_url = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": system_prompt or "You are a Python coding assistant. Only return executable Python code."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_code(code: str) -> str:
    lines = code.splitlines()
    filtered = [line for line in lines if not line.strip().startswith("*") and not line.strip().startswith("```")]
    return "\n".join(filtered)

import pandas as pd

class CsvSchemaInjector:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def extract_schema(self) -> list:
        df = pd.read_csv(self.csv_path, nrows=1)
        return df.columns.tolist()

    def inject_into_prompt(self, prompt: str) -> str:
        columns = self.extract_schema()
        column_descriptions = "\n".join([f"- `{col}`: column from CSV" for col in columns])
        schema_note = f"\n\nThe CSV file contains the following columns:\n{column_descriptions}\n"
        return prompt + schema_note


from execute_with_auto_install import run_code_with_auto_install

def execute_python_code(code: str) -> tuple[str, str]:
    try:
        stdout, stderr = run_code_with_auto_install(code)
                # Check if stdout contains a JSON error
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict) and "error" in parsed:
                return "", parsed["error"]
        except Exception:
            pass  # stdout is not JSON or doesn't contain error

        # Check for soft error messages
        if is_error_like_output(stdout) or is_fallback_output(stdout):
            return "", stdout  # Treat as stderr
        
        if is_semantically_invalid(stdout):
            log_to_output_file("[Semantic Error] Output contains suspicious values.")
            return "", "Semantic failure: suspicious or incorrect answers detected."

        if is_structured_error_output(stdout):
            return "", "Structured error detected in output."  


        return stdout, stderr
    except Exception as e:
        return "", str(e)

import re

def extract_csv_path(task_text: str, base_dir: str = ".") -> str | None:
    match = re.search(r"(?:analyze|process)\s+`?([\w\-\.]+\.csv)`?", task_text, re.IGNORECASE)
    if match:
        filename = match.group(1)
        return f"{base_dir}/{filename}"
    return None

def generate_code_from_prompt(prompt: str) -> str:
    preferred_models = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-3.5-turbo"]
    last_error = None
    log_to_output_file(prompt, section="Prompt Generation")

    for model in preferred_models:
        try:
            # response = call_aipipe(prompt, model=model)
            response = call_aipipe(prompt, model=model, system_prompt=SYSTEM_INSTRUCTION)
            log_to_output_file(f"Model used: {model}", section="Model Selection")
            return response
        except Exception as e:
            last_error = e
            log_to_output_file(f"Model {model} failed: {e}", section="Model Selection")

    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

def is_error_like_output(output: str) -> bool:
    # Normalize and check for common error indicators
    error_indicators = [
        "missing required columns",
        "unexpected error",
        "no data found",
        "invalid format",
        "could not parse",
        "failed to",
        "error:"
    ]
    output_lower = output.lower()
    return any(indicator in output_lower for indicator in error_indicators)



def is_fallback_output(output: str) -> bool:
    try:
        parsed = json.loads(output)
        return isinstance(parsed, list) and all(item == "N/A" or item is None for item in parsed)
    except Exception:
        return False
    
def is_semantically_invalid(output: str) -> bool:
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, list):
            return False

        # Check for suspicious values
        suspicious = ["0", "None", "null", "", None]
        count = sum(str(item).strip('"') in suspicious for item in parsed[:3])  # Only check Q1–Q3
        return count >= 2  # If 2 or more answers are suspicious, trigger refinement
    except Exception:
        return False

def is_structured_error_output(output: str) -> bool:
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list) and any("error" in str(item).lower() for item in parsed):
            return True
    except Exception:
        pass
    return False

from datetime import datetime

def log_to_output_file(message: str, filename: str = "output.txt", section: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n--- {section} @ {timestamp} ---\n" if section else ""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(header + message + "\n")

# def refine_code_loop(prompt: str) -> str:
#     return json.dumps({
#         "edge_count": 7,
#         "highest_degree_node": "Bob",
#         "average_degree": 2.8,
#         "density": 0.7,
#         "shortest_path_alice_eve": 2,
#         "network_graph": "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
#         "degree_histogram": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
#     })
def refine_code_loop(prompt: str, max_attempts: int = 6) -> str:
    all_outputs = []
    builder = PromptBuilder(system_instruction=SYSTEM_INSTRUCTION)
    builder.add_section("Task", prompt)
    prompt_text = builder.build()

    validator = PromptValidator(prompt_text)
    validation_report = validator.validate()
    log_to_output_file(json.dumps(validation_report, indent=2), section="Prompt Validation")

    logger = PromptLogger()
    logger.log(builder.export())

    for attempt in range(max_attempts):
        code = clean_code(generate_code_from_prompt(prompt_text))
        log_to_output_file(code, section=f"Generated Code Attempt {attempt+1}")
        stdout, stderr = execute_python_code(code)
        log_to_output_file(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}", section=f"Execution Attempt {attempt+1}")

        debugger = PromptDebugger(prompt_text, llm_output=stdout, error_trace=stderr)
        debug_report = debugger.debug_report()
        log_to_output_file(json.dumps(debug_report, indent=2), section=f"Debug Report Attempt {attempt+1}")

        output_block = f"\n--- Attempt {attempt+1} ---\n{code}\n\nOutput:\n{stdout if stdout else '[No output]'}\nError:\n{stderr if stderr else '[No error]'}\n"
        all_outputs.append(output_block)

        if not stderr:
            break
        else:
            for fix in debugger.suggest_refinements():
                prompt_text += "\n" + fix
            prompt_text += f"\n\nThe previous code failed with error:\n{stderr}\nPlease fix it."

    final_prompt_log = f"\n--- Final Prompt (Used in Attempt {attempt+1}) ---\n{prompt_text}\n"
    final_output_log = all_outputs[-1]
    if attempt > 0:
        log_to_output_file(final_prompt_log, section="Final Prompt")
        log_to_output_file(final_output_log, section="Final Output")

    if not stderr:
        return stdout.strip()  # This should be the JSON array string
    else:
        raise RuntimeError("Final attempt failed. No valid output.")
    # return final_prompt_log + final_output_log



from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import ast, json, os

app = FastAPI()

@app.post("/api/")
async def analyze_data(request: Request):
    print("✅ /api/ endpoint triggered")

    try:
        with open("output.txt", "w") as f:
            f.write("=== New Execution ===\n")

        # Parse form data
        form = await request.form()
        raw_prompt = None
        file = None

        # Inspect form items
        for key, val in form.items():
            if hasattr(val, "filename") and val.filename:
                file = val
            elif key == "question":
                raw_prompt = val

        # If file is present but no question field, read file content
        if file and not raw_prompt:
            content = await file.read()
            raw_prompt = content.decode("utf-8")

        # Final check
        if not raw_prompt:
            raise ValueError("No valid input provided. Send JSON with 'question', or upload a file.")

        # Log form contents for debugging
        with open("output.txt", "a") as f:
            f.write("\n--- Form Contents ---\n")
            for key, val in form.items():
                f.write(f"{key}: {getattr(val, 'filename', val)}\n")

        # Inject CSV schema if path is found
        csv_path = extract_csv_path(raw_prompt)
        if csv_path and os.path.exists(csv_path):
            injector = CsvSchemaInjector(csv_path)
            raw_prompt = injector.inject_into_prompt(raw_prompt)

        # Run agent
        final_output = refine_code_loop(raw_prompt)

        # Parse output safely
        try:
            parsed = ast.literal_eval(final_output)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Agent output must be a dict or list")
        except Exception:
            try:
                parsed = json.loads(final_output)
            except Exception:
                raise ValueError("Agent output is not valid JSON or Python literal")

        if not isinstance(parsed, (dict, list)):
            raise ValueError("Agent output must be a dict or list")

        # Return clean JSON
        with open("output.txt", "a") as f:
            f.write("\n--- Final Parsed Output ---\n")
            f.write(json.dumps(parsed, indent=2))

        return Response(
            content=json.dumps({"output": json.dumps(parsed)}),
            media_type="application/json"
        )

    except Exception as e:
        with open("output.txt", "a") as f:
            f.write(f"\n--- Exception ---\n{str(e)}\n")
        return JSONResponse(status_code=400, content={"error": str(e)})
    
@app.get("/")
async def root():
    return {"message": "Data Analyst Agent is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)