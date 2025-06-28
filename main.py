# main.py (Final Stable Version - V3.0)

import uvicorn
import asyncio
import re
from fastapi.responses import StreamingResponse
import agent_core
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, TypedDict
from supabase import Client

# --- Core Application Imports ---
from agent_core import get_agent_response, CHOMOLUNGMA_PERSONA
from db_client import get_supabase_client, AuthenticatedUser

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, messages_to_dict, messages_from_dict

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Chomolungma API - V3 (Stable)",
    description="API for the AI Trekking Companion with User Authentication.",
    version="3.0.0",
)

# --- 2. Pydantic Data Models ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class ImagePayload(BaseModel):
    mime_type: str
    data: str

class QueryRequest(BaseModel):
    query: str
    image: Optional[ImagePayload] = None

class AgentResponse(BaseModel):
    response: str
    user_id: str

# --- 3. Authentication Logic ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthenticatedUser:
    try:
        supabase = get_supabase_client()
        user_session = supabase.auth.get_user(token)
        if not user_session:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return {"user": user_session.user, "jwt": token}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user_data: UserCreate):
    try:
        supabase = get_supabase_client()
        session = supabase.auth.sign_up({"email": user_data.email, "password": user_data.password})
        if session.user:
            return {"id": session.user.id, "email": session.user.email}
        else:
            raise HTTPException(status_code=400, detail="Could not register user. They may already exist.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        supabase = get_supabase_client()
        session = supabase.auth.sign_in_with_password({"email": form_data.username, "password": form_data.password})
        if session.session.access_token:
            return {"access_token": session.session.access_token, "token_type": "bearer"}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

# --- 4. Database Interaction Logic ---
def load_or_create_history(user_id: str, supabase: Client) -> List[Dict]:
    response = supabase.table('conversations').select('history').eq('user_id', user_id).execute()
    if response.data and response.data[0].get('history') is not None:
        return response.data[0]['history']
    else:
        initial_history = [messages_to_dict([SystemMessage(content=CHOMOLUNGMA_PERSONA)])[0]]
        supabase.table('conversations').upsert({'user_id': user_id, 'history': initial_history}).execute()
        return initial_history

def save_history(user_id: str, messages: List[Dict], supabase: Client):
    supabase.table('conversations').update({'history': messages}).eq('user_id', user_id).execute()

# --- 5. The Main Agent Endpoint ---
@app.post("/agent/invoke", tags=["Agent"])
async def invoke_agent(
    request: QueryRequest,
    current_user_data: AuthenticatedUser = Depends(get_current_user)
):
    """Stream events from the LangGraph agent while also emitting incremental
    UI-friendly chunks.  Each chunk is prefixed with either
    "STATUS:" or "CONTENT:" so the frontend can route it correctly.
    """
    user_id = current_user_data["user"].id
    jwt = current_user_data["jwt"]
    supabase = get_supabase_client(jwt)

    # Load existing history and stage the new user message so the graph sees it
    history_dicts = load_or_create_history(user_id, supabase)
    image_data_dict = request.image.model_dump() if request.image and request.image.data else None

    # Convert stored dict history back into message objects and add the new turn
    messages = messages_from_dict(history_dicts)
    messages.append(HumanMessage(content=request.query))

    initial_state = {
        "messages": messages,
        "user_request": None,
        "plan": None,
        "gear_checklist": None,
        "budget": None,
        "image_data": image_data_dict,
    }

    # Human-friendly mapping from node name to status text
    STATUS_MESSAGES = {
        "extract_user_request": "Understanding your request...",
        "get_data_for_plan": "Searching my knowledge base...",
        "synthesize_custom_plan": "Creating itinerary...",
        "generate_gear_checklist": "Drafting gear checklist...",
        "generate_budget": "Calculating budget estimate...",
        "compile_final_response": "Finalising answer...",
    }

    async def event_stream():
        """Async generator that yields STATUS and CONTENT chunks."""
        final_state = None
        # Stream LangGraph events – stream_mode="values" gives incremental state
        async for event in agent_core.app.astream_events(initial_state, version="v1", stream_mode="values"):
            ev_type: str = event["event"]
            node_name: str = event.get("name", "")

            # Emit a STATUS update for **whitelisted** nodes only
            if ev_type.endswith("_start") and node_name:
                allowed_nodes = set(STATUS_MESSAGES.keys()) | {"router"}
                if node_name in allowed_nodes:
                    status_text = STATUS_MESSAGES.get(node_name, node_name.replace('_', ' ').title())
                    yield f"STATUS:{status_text}\n"
                    await asyncio.sleep(0)

            # Emit router decision summary after it finishes


            # Capture the final graph state when it ends so we can stream the content
            if ev_type == "on_chain_end" and "output" in event.get("data", {}):
                final_state = event["data"]["output"]

        # After the graph is finished, stream the assistant reply word-by-word
        if final_state:
            last_msg = final_state["messages"][-1]
            # The message may be an AIMessage instance or a dict depending on LC version
            content_text = getattr(last_msg, "content", None) or last_msg.get("data", {}).get("content", "")
            # Stream while preserving whitespace/newlines for nicer formatting
            # Replace single newlines with double for markdown paragraphs
            prepped_text = content_text.replace("\n", "\n\n")
            tokens = re.split(r'(\s+)', prepped_text)
            for tok in tokens:
                if tok:
                    # Encode newlines so they survive HTTP chunk separation
                    encoded_tok = tok.replace("\n", "\\n")
                    yield f"CONTENT:{encoded_tok}\n"
                    # await asyncio.sleep(0.02)
            yield "\n"  # flush final newline

            # Persist the full history once streaming is complete
            final_history_to_save = messages + [AIMessage(content=content_text)]
            save_history(user_id, messages_to_dict(final_history_to_save), supabase)

    return StreamingResponse(event_stream(), media_type="text/plain")

@app.get("/", tags=["Monitoring"])
def read_root():
    return {"message": "Welcome to the Chomolungma Planner API V3 (Stable)."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)