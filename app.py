from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def task_breakdown(task:str):
    """Get a description of the image using Google GenAI."""
    # client = genai.Client(api_key=os.getenv("AIzaSyCiJqFuV_8LidPrwQr8MAoJRS_5vowMNJQ"))

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # ✅ Use environment variable GOOGLE_API_KEY
    model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-1.0-pro / gemini-1.5-pro
    # my_file = client.files.upload(file=image_path)

    # with open(os.path.join('pdsaiitm.github.io', image_path), 'rb') as img_file:
    #     image_data = img_file.read()
    task_breakdown_file =os.path.join('prompts',"task_breakdown.txt")
    with open(task_breakdown_file,'r') as f:
        task_breakdown_prompt = f.read()

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     # contents=[my_file, "Describe the content of this image in detail, focusing on any text, objects, or relevant aspects."]
    #     contents=[task,"Break down the task"]
    # )
    response = model.generate_content([task_breakdown_prompt, task])
    
    with open("breaked_task.txt","w") as f:
        f.write(response.text)

    return response.text

app.get("/")
async def root():
    return{"message":"Hello!"}


@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text=content.decode("utf-8")
        breakdown = task_breakdown(text)
        print(breakdown)
        return{"filename":file.filename,"content":text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error":str(e)})
    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
