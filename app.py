from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.get("/")
async def root():
    return{"message":"Hello!"}


@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text=content.decode("utf-8")
        return{"filename":file.filename,"content":text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error":str(e)})
    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
