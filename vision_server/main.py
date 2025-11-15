from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from vision_processor import VisionProcessor  # Your vision class
from io import BytesIO
from PIL import Image

# ------------------------------
# 1. Initialize FastAPI app
# ------------------------------
app = FastAPI()

# ------------------------------
# 2. CORS FIX — VERY IMPORTANT
# ------------------------------
# This allows your frontend (even from file:/// or localhost) 
# to access your Python server without being blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Allow all origins (development only)
    allow_credentials=True,
    allow_methods=["*"],              # Allow all HTTP methods (POST, GET, ...)
    allow_headers=["*"],              # Allow all request headers
)

# ------------------------------
# 3. Initialize Vision Processor
# ------------------------------
vp = VisionProcessor()

# ------------------------------
# 4. API Endpoint
# ------------------------------
@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Receives an image file, processes it with AI models,
    and returns structured JSON data.
    """
    try:
        # Read image bytes
        contents = await file.read()

        # Load with PIL
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Process with vision models
        results = vp.process_pil(img)

        # Return JSON back to client/browser
        return JSONResponse(results)

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------
# 5. Run Server
# ------------------------------
if __name__ == "__main__":
    print("--- Vision Server Started Successfully ---")
    print("Access it via: http://YOUR_LOCAL_IP:8000/analyze-frame")
    print("Make sure your app/browser is on the same network.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
