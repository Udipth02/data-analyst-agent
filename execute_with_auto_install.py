import subprocess
import sys
import traceback

def run_code_with_auto_install(code: str) -> tuple[str, str]:
    temp_file = "temp_code.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout, ""
        else:
            return handle_error(result.stderr, code)
    except Exception as e:
        return "", str(e)


def handle_error(stderr: str, code: str) -> tuple[str, str]:
    if "ModuleNotFoundError" in stderr:
        missing_module = extract_module_name(stderr)
        if missing_module:
            subprocess.run([sys.executable, "-m", "pip", "install", missing_module])
            return run_code_with_auto_install(code)
    return "", stderr


def extract_module_name(error_text: str):
    # Example: "ModuleNotFoundError: No module named 'lxml'"
    for line in error_text.splitlines():
        if "No module named" in line:
            parts = line.split("'")
            if len(parts) >= 2:
                return parts[1]
    return None