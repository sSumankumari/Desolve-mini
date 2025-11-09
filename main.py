import os
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai

# --- Local Imports ---
# Make sure clone.py, file_contents.py, and issues.py are in the same directory
from clone import clone_repo
from file_contents import extract_files_to_csv
from issues import extract_issues
from models.gemini_models_rag import (
    load_env_and_configure,
    build_vector_index,
    create_prompt,
    retrieve_relevant_files,
    MODEL_NAME,
    ISSUES_CSV,
    FILES_CSV,
    INDEX_PATH,
)

# --- Configuration & App Setup ---
try:
    # Load API keys and configure Gemini
    load_env_and_configure()
except ValueError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Please create a .env file with your GEMINI_API_KEY.")
    # The server will fail to start, which is correct if no key is present.
    exit(1)


app = FastAPI()

# --- Global State (Server-level cache) ---
# Initialize the Gemini Model
gemini_model = genai.GenerativeModel(MODEL_NAME)
# Cache for active chat sessions, keyed by issue_id
# This allows users to continue conversations
chat_sessions = {}


# --- Pydantic Models (for API Request Validation) ---
class RepoRequest(BaseModel):
    """Request model for processing a new repository."""
    url: str

class AiRequest(BaseModel):
    """Request model for asking the AI a question."""
    issue_id: int
    prompt: str


# --- Helper Function ---
def load_issue_by_id(csv_path: str, issue_id: int) -> dict | None:
    """
    Loads a specific issue from the CSV using its GitHub issue number (id).
    """
    if not os.path.exists(csv_path):
        print(f"❌ Issues CSV not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # 'number' is the GitHub issue number, which the frontend sends as 'id'
    if 'number' not in df.columns:
        print("❌ 'number' column not found in issues CSV.")
        return None

    # Find the issue by its 'number'
    issue_df = df[df['number'] == issue_id]
    
    if issue_df.empty:
        print(f"❌ Issue with ID {issue_id} not found in {csv_path}")
        return None
        
    # Return the first matching issue as a dictionary
    return issue_df.to_dict(orient="records")[0]


# --- API Endpoint 1: Process Repository ---
@app.post("/process-repo")
async def process_repo(request: RepoRequest):
    """
    Clones a repo, extracts files/issues, builds the vector index,
    and returns the list of issues to the frontend.
    """
    print(f"🚀 Received request to process repo: {request.url}")
    
    # Clear old chat sessions from any previous repo
    chat_sessions.clear()
    
    # Define paths
    clone_dir = "cloned_repo"
    repo_name = request.url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)
    token = os.getenv("GITHUB_TOKEN") # For fetching issues

    try:
        # Step 1: Clone repo
        print(f"Cloning {request.url}...")
        clone_repo(request.url, clone_dir)

        # Step 2: Extract files to CSV
        print("📂 Extracting repository files...")
        extract_files_to_csv(repo_path, FILES_CSV)

        # Step 3: Extract issues to CSV
        print("🐞 Extracting repository issues...")
        if not token:
            print("⚠️ GITHUB_TOKEN not set. May be rate-limited or fail on private repos.")
        extract_issues(request.url, output_file=ISSUES_CSV, token=token)

        # Step 4: Build vector index
        print("🧠 Building vector index...")
        build_vector_index() # This will use paths from gemini_models_rag.py

        # Step 5: Read issues CSV and send to frontend
        print("✅ Processing complete. Sending issues to frontend.")
        if not os.path.exists(ISSUES_CSV):
             raise FileNotFoundError(f"Issues CSV not created at {ISSUES_CSV}")
             
        df_issues = pd.read_csv(ISSUES_CSV)
        
        # --- CRITICAL ---
        # The frontend expects a field named 'id' to be the issue number.
        # We map the 'number' column (from GitHub) to 'id'.
        if 'number' not in df_issues.columns:
            raise KeyError("Issues CSV must contain a 'number' column.")
            
        df_issues['id'] = df_issues['number']
        
        # Select only the columns the frontend needs
        issues_list = df_issues[['id', 'title', 'body']].to_dict(orient="records")
        
        return {"issues": issues_list}

    except Exception as e:
        print(f"❌ Error during repo processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )


# --- API Endpoint 2: Chat with AI ---
@app.post("/ask-ai")
async def ask_ai(request: AiRequest):
    """
    Handles a chat message from the user, routing it to the
    correct RAG-powered chat session.
    """
    print(f"🤖 Received chat request for issue {request.issue_id}: {request.prompt}")
    
    try:
        # Check if a chat session for this issue already exists
        if request.issue_id not in chat_sessions:
            print(f"Creating new chat session for issue {request.issue_id}...")
            
            # 1. Load the selected issue's data
            issue = load_issue_by_id(ISSUES_CSV, request.issue_id)
            if issue is None:
                raise HTTPException(status_code=404, detail="Selected issue not found.")
            
            # 2. Retrieve relevant files (RAG)
            # We use the issue body as the *initial* query to find relevant files
            print("Retrieving relevant files...")
            repo_context = retrieve_relevant_files(issue.get("body", ""))
            
            # 3. Create the initial system prompt
            system_prompt = create_prompt(issue, repo_context)
            
            # 4. Start a new chat session with the system prompt
            chat = gemini_model.start_chat(
                history=[{"role": "user", "parts": system_prompt}]
            )
            
            # 5. Store the new chat session
            chat_sessions[request.issue_id] = chat
        
        # Get the chat session (either new or existing)
        chat = chat_sessions[request.issue_id]
        
        # Send the user's *actual* prompt to the chat
        print(f"Sending prompt to Gemini: {request.prompt}")
        response = chat.send_message(request.prompt)
        
        print(f"🤖 Got response from Gemini: {response.text[:50]}...")
        return {"response": response.text}

    except Exception as e:
        print(f"❌ Error during AI chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred with the AI model: {e}"
        )


# --- API Endpoint 3: Serve Frontend ---
# Mount the 'frontend' directory to serve index.html
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def get_index():
    """Serves the main index.html file."""
    return FileResponse("frontend/index.html")


# --- Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    # Make sure your 'frontend' folder is in the same directory as this main.py
    if not os.path.exists("frontend/index.html"):
        print("WARNING: 'frontend/index.html' not found.")
        print("         Please ensure your 'frontend' folder is in the correct location.")
        
    uvicorn.run(app, host="127.0.0.1", port=8000)