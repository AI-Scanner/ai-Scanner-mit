
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Jello World"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    return {"filename": file.filename}

@app.get("/routes")
def get_routes():
    return [{"path": route.path, "methods": route.methods} for route in app.routes]
