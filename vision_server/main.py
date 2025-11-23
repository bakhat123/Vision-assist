from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from vision_processor import VisionProcessor  # Your vision class
from io import BytesIO
from PIL import Image
import os
import socket

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
# 3. Serve Static Files (index.html from parent directory)
# ------------------------------
# Determine the parent directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Mount static files from parent directory
app.mount("/static", StaticFiles(directory=parent_dir), name="static")

# Serve index.html at root
@app.get("/")
async def serve_index():
    """Serve index.html at the root path."""
    index_path = os.path.join(parent_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

# Serve index.html at /index.html
@app.get("/index.html")
async def serve_index_explicit():
    """Serve index.html explicitly."""
    index_path = os.path.join(parent_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

# ------------------------------
# 4. Initialize Vision Processor
# ------------------------------
vp = VisionProcessor()

# ------------------------------
# 5. API Endpoint
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
# 6. Run Server
# ------------------------------
if __name__ == "__main__":
    # Local IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"

    # Check for SSL certificates
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(current_dir, "cert.pem")
    key_file = os.path.join(current_dir, "key.pem")
    
    use_https = os.path.exists(cert_file) and os.path.exists(key_file)
    protocol = "https" if use_https else "http"
    port = 8443 if use_https else 8000

    print("--- Vision Server Started Successfully ---")
    print(f"Access frontend at: {protocol}://10.7.33.86:{port}/")
    print(f"Local access: {protocol}://{local_ip}:{port}/")
    print(f"API endpoint: {protocol}://10.7.33.86:{port}/analyze-frame")
    
    if not use_https:
        print("\n⚠️  Running on HTTP. Camera access may be blocked by browser.")
        print("To enable HTTPS:")
        print("  1. Run: python setup_https.py")
        print("  2. Restart this script")
    else:
        print("\n✓ Running on HTTPS with self-signed certificate")
        print("  Browser will show a warning (normal for self-signed certs)")
        print("  Click 'Advanced' and 'Proceed' to continue")
    
    print("Make sure your app/browser is on the same network.")
    
    if use_https:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
