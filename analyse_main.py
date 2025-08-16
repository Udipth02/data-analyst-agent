@app.post("/api/")
async def analyze_data(
    request: Request,
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    question: str = Form(None)
):
    # IMMEDIATE logging - this should appear even if everything else fails
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Write to multiple places for debugging
    debug_log = f"=== ENDPOINT HIT at {timestamp} ===\n"
    
    try:
        with open("output.txt", "w") as f:
            f.write(debug_log)
        with open("debug_log.txt", "a") as f:
            f.write(debug_log)
        print(debug_log.strip())  # Also print to console
    except Exception as log_err:
        print(f"LOGGING ERROR: {log_err}")
    
    try:
        # Log EVERYTHING about the request
        headers = dict(request.headers)
        content_type = request.headers.get("content-type", "")
        method = request.method
        url = str(request.url)
        
        debug_info = f"""
REQUEST DEBUG INFO:
- Method: {method}
- URL: {url}
- Content-Type: {content_type}
- Headers: {headers}
- Single file: {file.filename if file else 'None'}
- Multiple files: {[f.filename for f in files if f] if files else 'None'}
- Form question length: {len(question) if question else 0}
"""
        
        with open("output.txt", "a") as f:
            f.write(debug_info)
        print(debug_info)
        
        # Try to read request body directly
        try:
            body_bytes = await request.body()
            body_preview = body_bytes[:500] if body_bytes else b"Empty"
            with open("output.txt", "a") as f:
                f.write(f"\nRAW BODY (first 500 bytes): {body_preview}\n")
        except Exception as body_err:
            with open("output.txt", "a") as f:
                f.write(f"Error reading body: {body_err}\n")

        raw_prompt = None
        
        # Method 1: Handle single file upload (for try.sh compatibility)
        if file and file.filename:
            try:
                content = await file.read()
                raw_prompt = content.decode("utf-8")
                with open("output.txt", "a") as f:
                    f.write(f"SUCCESS: Extracted from single file {file.filename}: {raw_prompt[:100]}...\n")
            except Exception as e:
                with open("output.txt", "a") as f:
                    f.write(f"ERROR reading single file: {e}\n")
        
        # Method 2: Handle JSON body (for promptfoo)
        elif "application/json" in content_type:
            try:
                # Reset request for JSON reading
                body = await request.json()
                with open("output.txt", "a") as f:
                    f.write(f"Received JSON body: {body}\n")
                
                if "question" in body:
                    question_value = body["question"]
                    
                    # Handle file:// references from promptfoo
                    if question_value.startswith("file://"):
                        file_path = question_value.replace("file://", "")
                        with open("output.txt", "a") as f:
                            f.write(f"Attempting to read file: {file_path}\n")
                        
                        # Try multiple possible locations
                        possible_paths = [
                            file_path,
                            os.path.join(".", file_path),
                            os.path.join(os.getcwd(), file_path),
                        ]
                        
                        with open("output.txt", "a") as f:
                            f.write(f"Trying paths: {possible_paths}\n")
                        
                        for path in possible_paths:
                            try:
                                if os.path.exists(path):
                                    with open(path, 'r', encoding='utf-8') as f:
                                        raw_prompt = f.read().strip()
                                    with open("output.txt", "a") as f:
                                        f.write(f"SUCCESS: Read file from {path}, length: {len(raw_prompt)}\n")
                                    break
                            except Exception as file_err:
                                with open("output.txt", "a") as f:
                                    f.write(f"Failed to read {path}: {file_err}\n")
                        
                        if raw_prompt is None:
                            try:
                                current_files = os.listdir(".")[:20]  # First 20 files
                                with open("output.txt", "a") as f:
                                    f.write(f"Current directory files: {current_files}\n")
                                current_dir = os.getcwd()
                                with open("output.txt", "a") as f:
                                    f.write(f"Current working directory: {current_dir}\n")
                            except Exception as dir_err:
                                with open("output.txt", "a") as f:
                                    f.write(f"Error listing directory: {dir_err}\n")
                            
                            error_msg = f"Could not find file: {file_path}"
                            with open("output.txt", "a") as f:
                                f.write(f"ERROR: {error_msg}\n")
                            
                            # Return a simple test response for debugging
                            return JSONResponse(
                                status_code=400,
                                content={"error": error_msg, "debug": "file_not_found"}
                            )
                    else:
                        raw_prompt = question_value
                        with open("output.txt", "a") as f:
                            f.write(f"SUCCESS: Using direct JSON question, length: {len(raw_prompt)}\n")
                else:
                    error_msg = "JSON body must contain 'question' field"
                    with open("output.txt", "a") as f:
                        f.write(f"ERROR: {error_msg}\n")
                    return JSONResponse(
                        status_code=400,
                        content={"error": error_msg, "debug": "missing_question_field"}
                    )
                    
            except Exception as e:
                with open("output.txt", "a") as f:
                    f.write(f"ERROR parsing JSON: {e}\n")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"JSON parsing failed: {str(e)}", "debug": "json_parse_error"}
                )
        
        # Method 3: Handle multiple files upload
        elif files and any(f for f in files):
            with open("output.txt", "a") as f:
                f.write(f"Trying multiple files: {[f.filename for f in files if f]}\n")
            
            for uploaded_file in files:
                if uploaded_file and uploaded_file.filename:
                    try:
                        content = await uploaded_file.read()
                        raw_prompt = content.decode("utf-8")
                        with open("output.txt", "a") as f:
                            f.write(f"SUCCESS: Extracted from {uploaded_file.filename}, length: {len(raw_prompt)}\n")
                        break
                    except Exception as e:
                        with open("output.txt", "a") as f:
                            f.write(f"ERROR reading {uploaded_file.filename}: {e}\n")
        
        # Method 4: Handle form field
        elif question:
            raw_prompt = question
            with open("output.txt", "a") as f:
                f.write(f"SUCCESS: Using form question, length: {len(raw_prompt)}\n")
        
        # Final validation
        if raw_prompt is None:
            error_msg = "No valid input found after trying all methods"
            with open("output.txt", "a") as f:
                f.write(f"FINAL ERROR: {error_msg}\n")
            
            # Return a simple test response to see if this gets through
            return JSONResponse(
                content={
                    "error": error_msg,
                    "debug": "no_input_found",
                    "timestamp": timestamp,
                    "test_response": True
                }
            )
        
        # If we got here, we have raw_prompt
        with open("output.txt", "a") as f:
            f.write(f"PROCESSING: Starting with prompt length {len(raw_prompt)}\n")
            f.write(f"Prompt preview: {raw_prompt[:200]}...\n")
        
        # For debugging, let's return a simple test response first
        # Comment out the actual processing temporarily
        test_response = {
            "debug": "endpoint_working",
            "prompt_length": len(raw_prompt),
            "timestamp": timestamp,
            "message": "This is a test response to verify the endpoint is working"
        }
        
        with open("output.txt", "a") as f:
            f.write(f"RETURNING TEST RESPONSE: {test_response}\n")
        
        return JSONResponse(
            content=test_response,
            headers={"Content-Type": "application/json"}
        )
        
        # TODO: Uncomment this section once we confirm the endpoint is working
        """
        # Process CSV schema injection if needed
        csv_path = extract_csv_path(raw_prompt)
        if csv_path and os.path.exists(csv_path):
            injector = CsvSchemaInjector(csv_path)
            raw_prompt = injector.inject_into_prompt(raw_prompt)

        final_output = refine_code_loop(raw_prompt)
        
        # Parse and return the result
        try:
            parsed = ast.literal_eval(final_output)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Agent output must be a valid dict or list")
            
            return JSONResponse(
                content=parsed,
                headers={"Content-Type": "application/json"}
            )
            
        except (ValueError, SyntaxError) as e:
            try:
                parsed = json.loads(final_output)
                return JSONResponse(
                    content=parsed,
                    headers={"Content-Type": "application/json"}
                )
            except json.JSONDecodeError as json_err:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Failed to parse agent output",
                        "raw_output": final_output[:1000],
                        "parse_errors": {
                            "literal_eval": str(e),
                            "json_loads": str(json_err)
                        }
                    }
                )
        """

    except Exception as e:
        error_msg = str(e)
        try:
            with open("output.txt", "a") as f:
                f.write(f"FINAL EXCEPTION: {error_msg}\n")
        except:
            pass
        
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "timestamp": timestamp,
                "debug": "final_exception"
            }
        )
@app.get("/")
async def root():
    return {"message": "Data Analyst Agent is running!"}

# Add this middleware to your main.py to log ALL requests
from fastapi import FastAPI, Request, Response
import time
import datetime

@app.middleware("http")
async def log_requests(request: Request, call_next):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Log every single request
    log_entry = f"""
=== REQUEST at {timestamp} ===
Method: {request.method}
URL: {str(request.url)}
Path: {request.url.path}
Headers: {dict(request.headers)}
Client: {request.client}
"""
    
    # Write to multiple places
    try:
        with open("all_requests.log", "a") as f:
            f.write(log_entry)
        print(log_entry)  # Console output
    except Exception as e:
        print(f"Logging error: {e}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Log the response too
        response_log = f"Response Status: {response.status_code}\n"
        with open("all_requests.log", "a") as f:
            f.write(response_log)
        print(response_log)
        
        return response
    except Exception as e:
        error_log = f"Request processing error: {e}\n"
        with open("all_requests.log", "a") as f:
            f.write(error_log)
        print(error_log)
        raise

# Also add a simple GET endpoint for the root
@app.get("/api/")
async def api_get():
    return {"message": "API is working! Use POST method.", "timestamp": str(datetime.datetime.now())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)