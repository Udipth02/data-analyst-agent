from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini code generation
def generate_code_from_prompt(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt]
    )
    return response.text

# Execute Python code and capture output/errors
# def execute_python_code(code: str) -> tuple[str, str]:
#     with open("temp_code.py", "w") as f:
#         f.write(code)

#     try:
#         result = subprocess.run(
#             # ["python", "temp_code.py"],
#             ["uv run ", "temp_code.py"],
#             capture_output=True,
#             text=True,
#             timeout=60
#         )
#         return result.stdout, result.stderr
#     except subprocess.TimeoutExpired:
#         return "", "Execution timed out"

from execute_with_auto_install import run_code_with_auto_install

def execute_python_code(code: str) -> tuple[str, str]:
    try:
        stdout, stderr = run_code_with_auto_install(code)
        return stdout, stderr
    except Exception as e:
        return "", str(e)


def clean_code(code: str) -> str:
    lines = code.splitlines()
    filtered = [line for line in lines if not line.strip().startswith("*") and not line.strip().startswith("```")]
    return "\n".join(filtered)

# Iterative refinement loop
def refine_code_loop(prompt: str, max_attempts: int = 6) -> str:
    all_outputs = []
    
    SYSTEM_INSTRUCTION = """
    You are a Python coding assistant. Your task is to generate executable Python code only.
    Do not include markdown, explanations, comments, or formatting. 
    Only return valid Python code that can be run directly in a script.
    """

    prompt = SYSTEM_INSTRUCTION + "\n\n" + prompt



    for attempt in range(max_attempts):
        code = clean_code(generate_code_from_prompt(prompt))
        stdout, stderr = execute_python_code(code)

        output_block = f"\n--- Attempt {attempt+1} ---\n{code}\n\nOutput:\n{stdout}\nError:\n{stderr}\n"
        all_outputs.append(output_block)

        if not stderr:
            break  # Success

        # Refine prompt with error feedback
        prompt += f"\nThe previous code failed with error:\n{stderr}\nPlease fix it."

    with open("output.txt", "w") as f:
        f.writelines(all_outputs)

    return all_outputs[-1]  # Return last attempt

@app.post("/api/")
async def analyze_data(file: UploadFile = File(...)):
    try:
        content = await file.read()
        prompt = content.decode("utf-8")

        final_output = refine_code_loop(prompt)
        return {"status": "completed", "result": final_output}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Data Analyst Agent is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)