import uvicorn
import sys
import os

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    print("[VeriPaper Server] Launching FastAPI Backend with Hybrid Ensemble Engine...")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
