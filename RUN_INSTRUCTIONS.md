# VeriPaper AI Forensic Tool — Run Instructions

This guide provides step-by-step instructions on how to set up and run the VeriPaper AI Forensic Tool (FastAPI backend + Next.js frontend). You can run the application either using the pre-built double-clickable Windows Batch (`.bat`) scripts or manually executing commands in your terminal.

---

## 1. Prerequisites
Make sure you have the following installed on your system:
- **Python 3.9 or newer** (with pip). Ensure "Add Python to PATH" is checked during installation.
- **Node.js (LTS version recommended)** which automatically includes `npm`.

---

## 2. First-Time Setup

### Option A: Using the Setup Batch Script (Recommended)
Simply double-click the **`install.bat`** script in the project root directory.
*Alternatively, run it from your command prompt:*
```cmd
install.bat
```
This script automatically:
1. Verifies your Python (3.9+), Node.js, and npm installations.
2. Creates a Python virtual environment (`.venv`) and installs `requirements.txt`.
3. Validates that the local model weights are present.
4. Installs the frontend Node.js packages.

---

### Option B: Running Setup Commands Manually
If you prefer not to use the batch file, run these commands in order in your command prompt:

1. **Configure Backend Virtual Environment**:
   ```cmd
   python -m venv --system-site-packages .venv
   ```
2. **Activate Environment and Install Python Packages**:
   ```cmd
   call .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Install Frontend NPM Packages**:
   ```cmd
   cd frontend
   npm install
   cd ..
   ```

---

## 3. Running the Application

### Option A: Using Batch Scripts (Runs in Background)

#### To Start the Application:
- **Option 1 (Start Everything)**: Double-click **`start_all.bat`**. This starts the backend server first, waits for it to load, then boots up the frontend server.
- **Option 2 (Independent Starts)**:
  - Double-click **`start_backend.bat`** (Starts FastAPI server in background on port `8000`).
  - Double-click **`start_frontend.bat`** (Starts Next.js dev server in background on port `3000`).

#### To Stop the Application:
- **Option 1 (Stop Everything)**: Double-click **`stop_all.bat`**. This safely terminates uvicorn and node child processes, freeing ports `8000` and `3000`.
- **Option 2 (Independent Stops)**:
  - Double-click **`stop_backend.bat`** to stop the backend.
  - Double-click **`stop_frontend.bat`** to stop the frontend.

---

### Option B: Running Manually (Foreground / Interactive)
If you prefer to see live server console logs inside interactive terminal windows:

1. **Terminal 1 — Run Backend Service**:
   ```cmd
   call .venv\Scripts\activate
   python -u -m uvicorn src.app:app --host 127.0.0.1 --port 8000
   ```
   *(Keep this terminal open. Press `Ctrl + C` to stop the backend).*

2. **Terminal 2 — Run Frontend Service**:
   ```cmd
   cd frontend
   npm run dev
   ```
   *(Keep this terminal open. Press `Ctrl + C` to stop the frontend).*

---

## 4. Expected Ports and URLs
Once both servers are running:
- **Web Interface**: Open [http://localhost:3000](http://localhost:3000) in your web browser.
- **Backend API Docs (Swagger UI)**: View [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test API endpoints directly.
- **Backend Console Log File**: Saved to `backend.log` (if started via `.bat` script).
- **Frontend Console Log File**: Saved to `frontend.log` (if started via `.bat` script).
