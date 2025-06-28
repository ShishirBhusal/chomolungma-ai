# agent_core.py - V2.0 "The Pro Planner"

import os
import json
import uuid
import base64
import operator
from dotenv import load_dotenv
from typing import Dict, TypedDict, Annotated, Literal, List, Optional

# --- LangChain/LangGraph Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import (
    HumanMessage, SystemMessage, BaseMessage, ToolMessage, AIMessage,
    messages_to_dict, messages_from_dict
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# --- 1. Environment and Persona Setup ---
load_dotenv()

CHOMOLUNGMA_PERSONA = """
You are "Chomolungma," an expert, safety-conscious AI guide for tourism in Nepal.
Your goal is to help users plan treks, with a strong emphasis on safety and personalization.
When you generate a custom plan, you MUST explain to the user *why* you made the changes, referencing their fitness and safety rules.
Maintain a friendly, encouraging, and expert tone.
You are a full-service pre-trek assistant. After creating an itinerary, you also provide a detailed gear checklist and a budget estimate.
"""
# RAG Database Path - ensure this matches the output of your script
CHROMA_DB_PATH = "AI Assistant/chroma_db_aca"

# --- 2. RAG Tool (Slightly modified to use the new DB path) ---
@tool
def retrieve_trekking_information(topics: List[str]) -> str:
    """
    Retrieves specific information about trekking in Nepal from the knowledge base.
    Provide a list of relevant topics like: ["gear for EBC trek in winter", "costs for Annapurna circuit", "acclimatization rules"].
    """
    print(f"\n--- TOOL: retrieve_trekking_information ---")
    print(f"Searching for topics: {topics}")
    
    try:
        if not os.path.isdir(CHROMA_DB_PATH):
            return f"Error: Knowledge base not found at {CHROMA_DB_PATH}. Please run the ingestion script."
            
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        retriever = vector_db.as_retriever(search_kwargs={"k": 10}) # Get more context for our specialized nodes
        
        all_docs_content = set()
        for topic in topics:
            docs = retriever.invoke(topic)
            for doc in docs:
                all_docs_content.add(doc.page_content)
        
        if not all_docs_content:
            return "Could not find any information on the requested topics in the knowledge base."
            
        return "\n\n---\n\n".join(list(all_docs_content))
    except Exception as e:
        return f"An error occurred while accessing the knowledge base: {e}"

# --- 3. Agent State (Upgraded for V2.0) ---
class PlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_request: Optional[str]
    # Itinerary will be stored here
    plan: Optional[str]
    # V2.0 Features:
    gear_checklist: Optional[str]
    budget: Optional[str]
    image_data: Optional[Dict]

# --- 4. Graph Nodes (Upgraded and New for V2.0) ---

llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# --- Nodes for the Planning Flow ---

def extract_user_request(state: PlannerState) -> dict:
    """Extracts the core user request from the latest message."""
    print("\n--- NODE: extract_user_request ---")
    # This node now only runs once per planning session.
    if not state.get('user_request'):
        user_message = state['messages'][-1].content
        print(f"Captured initial user request: {user_message}")
        return {"user_request": user_message}
    return {}

# In agent_core.py

def get_data_for_plan(state: PlannerState) -> dict:
    """Generates RAG queries and calls the tool to get planning info."""
    print("\n--- NODE: get_data_for_plan ---")

    # ### FIX 1: A much stricter prompt with formatting instructions. ###
    prompt_text = """
    You are an expert at generating search queries for a vector database.
    Based on the user's request, generate a list of 3 to 5 concise and relevant search queries to find information for planning their trek.

    **USER REQUEST:**
    {request}

    **IMPORTANT FORMATTING RULES:**
    1. You MUST output your answer as a valid JSON list of strings.
    2. The list should be enclosed in square brackets `[]`.
    3. Each query in the list must be a string enclosed in double quotes `""`.
    4. Do NOT include any other text, explanations, or markdown formatting before or after the JSON list.

    **EXAMPLE OUTPUT:**
    ["Annapurna Base Camp 10 day itinerary", "gear list for Annapurna trek", "ACAP permit cost Nepal", "teahouse prices Annapurna"]

    **JSON OUTPUT:**
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # ### FIX 2: Create a more robust chain with the JSON parser and retries. ###
    # This tells the chain to automatically retry up to 2 times if the LLM
    # produces output that doesn't parse correctly as JSON.
    rag_query_chain = (prompt | llm_flash | JsonOutputParser()).with_retry(
        stop_after_attempt=3
    )

    try:
        topics = rag_query_chain.invoke({"request": state['user_request']})
        print(f"Generated RAG topics for planning: {topics}")
    except Exception as e:
        print(f"--- ERROR: Failed to generate RAG topics after multiple retries. Error: {e}")
        # If it still fails, we provide a fallback set of topics to avoid a crash.
        topics = ["trekking itinerary", "trekking rules nepal", "trekking safety guidelines"]

    # This part remains the same
    tool_node = ToolNode([retrieve_trekking_information])
    tool_result = tool_node.invoke({"messages": [
        AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_trekking_information", "args": {"topics": topics}}])
    ]})
    return {"messages": tool_result['messages']}


def synthesize_custom_plan(state: PlannerState) -> dict:
    """Uses the RAG context and user request to create a detailed itinerary."""
    print("\n--- NODE: synthesize_custom_plan ---")
    context = state['messages'][-1].content # RAG tool output
    
    prompt = ChatPromptTemplate.from_template(
        """You are Chomolungma. Create a personalized, day-by-day trekking itinerary based on the user's request and the provided context.

        **User's Core Request:**
        {user_request}

        **Retrieved Knowledge (Standard Itineraries, Rules, Safety):**
        {context}

        **INSTRUCTIONS:**
        1. Create a detailed, day-by-day itinerary.
        2. Add a "Guide's Notes" section explaining *why* you made specific modifications (e.g., adding a rest day) by referencing the user's fitness and safety rules.
        3. Output *only* the itinerary and notes. Do not add any conversational text before or after.
        """
    )
    planner_chain = prompt | llm_pro | StrOutputParser()
    final_plan = planner_chain.invoke({
        "context": context,
        "user_request": state['user_request']
    })
    print("--- Itinerary generated. ---")
    return {"plan": final_plan} # Store the plan in the new 'plan' state field

# --- V2.0: NEW NODES for Gear and Budget ---
# In agent_core.py

def generate_gear_checklist(state: PlannerState) -> dict:
    """Generates a personalized gear checklist using the generated plan."""
    print("\n--- NODE: generate_gear_checklist ---")
    
    # ### THE FIX IS HERE ###
    # We must pass the arguments as a single dictionary to the tool's invoke method.
    tool_input = {"topics": ["trekking gear information", "clothing layers system", "winter gear", "first aid"]}
    context = retrieve_trekking_information.invoke(tool_input)
    # ### END OF FIX ###
    
    prompt = ChatPromptTemplate.from_template(
        """You are a gear expert. Based on the following trek itinerary and retrieved gear information, create a personalized and comprehensive gear checklist.
        Pay attention to the maximum altitude, duration, and implied season of the trek.

        **Trek Itinerary:**
        {plan}

        **Retrieved Gear Information:**
        {context}

        **INSTRUCTIONS:**
        1. Create a checklist organized by categories (e.g., Core Gear, Clothing, First Aid).
        2. Add a short "Expert Tip" explaining *why* certain items are critical for this specific trek.
        3. Output *only* the checklist.
        """
    )
    chain = prompt | llm_pro | StrOutputParser()
    checklist = chain.invoke({"plan": state['plan'], "context": context})
    print("--- Gear Checklist generated. ---")
    return {"gear_checklist": checklist}
# In agent_core.py

def generate_budget(state: PlannerState) -> dict:
    """Generates a budget estimate for the trek."""
    print("\n--- NODE: generate_budget ---")
    
    # ### THE FIX IS HERE ###
    # We must pass the arguments as a single dictionary to the tool's invoke method.
    tool_input = {"topics": ["trekking permit costs", "teahouse accommodation prices", "food and water costs", "guide and porter fees"]}
    context = retrieve_trekking_information.invoke(tool_input)
    # ### END OF FIX ###

    prompt = ChatPromptTemplate.from_template(
        """You are a budget analyst for Nepal treks. Based on the following itinerary and cost data, create a detailed budget estimate.

        **Trek Itinerary:**
        {plan}

        **Retrieved Cost Information:**
        {context}

        **INSTRUCTIONS:**
        1. Calculate the estimated costs for Permits, Accommodation, Food, and a recommendation for a Guide/Porter.
        2. Present the budget in a clear, itemized list.
        3. Provide a total estimated cost per person (excluding international flights).
        4. Add a "Money-Saving Tip" section.
        5. Output *only* the budget.
        """
    )
    chain = prompt | llm_pro | StrOutputParser()
    budget = chain.invoke({"plan": state['plan'], "context": context})
    print("--- Budget generated. ---")
    return {"budget": budget}
def compile_final_response(state: PlannerState) -> dict:
    """Compiles the plan, checklist, and budget into a single, formatted response."""
    print("\n--- NODE: compile_final_response ---")
    
    # Using f-strings to assemble the final markdown response
    full_response = (
        "Namaste! I've prepared a comprehensive trekking package for you. Here are the details:\n\n"
        "----------------------------------------\n\n"
        "## 🏔️ Your Personalized Itinerary\n\n"
        f"{state['plan']}\n\n"
        "----------------------------------------\n\n"
        "## 🎒 Personalized Gear Checklist\n\n"
        f"{state['gear_checklist']}\n\n"
        "----------------------------------------\n\n"
        "## 💰 Estimated Budget\n\n"
        f"{state['budget']}\n\n"
        "I hope this helps you prepare for an incredible adventure! Let me know if you have any other questions."
    )
    
    return {"messages": [AIMessage(content=full_response)]}

# --- Nodes for other conversation flows (unchanged) ---
def general_response(state: PlannerState) -> dict:
    print("\n--- NODE: general_response ---")
    response = llm_pro.invoke(state['messages'])
    return {"messages": [response]}

def answer_follow_up(state: PlannerState) -> dict:
    print("\n--- NODE: answer_follow_up ---")
    # This logic remains the same
    # ...

# --- 5. Graph Router (Simplified for this version) ---
# We'll use a simpler router for now and can bring back the more complex one later.
def route_request(state: PlannerState) -> Literal["plan", "vision", "chat"]:
    """
    Routes user requests to the appropriate workflow.
    Now includes a path for vision-related queries.
    """
    print("\n--- NODE: router ---")
    
    # Check if an image was included in the latest request
    if state.get("image_data"):
        print("Decision: Image detected. -> vision")
        return "vision"

    # If no image, use the existing text-based routing logic
    user_input = state['messages'][-1].content.lower()
    planning_keywords = ['plan', 'trek', 'itinerary', 'ebc', 'annapurna', 'help me with']
    
    if any(keyword in user_input for keyword in planning_keywords):
        print("Decision: Planning request detected. -> plan")
        return "plan"
    else:
        print("Decision: General chat. -> chat")
        return "chat"

# ### NEW NODE FOR VISION ###
def identify_image(state: PlannerState) -> dict:
    """
    Identifies the content of an image using the multimodal model.
    This node is specifically for answering "What is this?" questions about an image.
    """
    print("\n--- NODE: identify_image ---")
    image_info = state.get("image_data")
    last_message_content = state["messages"][-1].content

    # Construct the multimodal message for the LLM
    vision_prompt_message = HumanMessage(
        content=[
            # The text part of the prompt
            {
                "type": "text",
                "text": f"""You are Chomolungma, an expert guide to the Himalayas. A user has sent you an image with the following question: '{last_message_content}'.
                
                Analyze the image and provide a helpful, descriptive answer.
                - If it's a mountain, identify it and share an interesting fact.
                - If it's a plant or animal, identify it and mention its significance in the region.
                - If it's a piece of gear, explain what it is and its use in trekking.
                - If you cannot identify the object, say so politely.
                """
            },
            # The image part of the prompt
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_info['mime_type']};base64,{image_info['data']}"
                }
            }
        ]
    )
    
    # Invoke the pro model which has vision capabilities
    response = llm_pro.invoke([vision_prompt_message])
    
    return {"messages": [response]}



# --- 6. Graph Definition and Compilation (V2.0 Structure) ---
workflow = StateGraph(PlannerState)

# Add the nodes
workflow.add_node("router", lambda state: state)
workflow.add_node("extract_user_request", extract_user_request)
workflow.add_node("get_data_for_plan", get_data_for_plan)
workflow.add_node("synthesize_custom_plan", synthesize_custom_plan)
workflow.add_node("generate_gear_checklist", generate_gear_checklist)
workflow.add_node("generate_budget", generate_budget)
workflow.add_node("compile_final_response", compile_final_response)
workflow.add_node("general_response", general_response) # For non-planning chat
workflow.add_node("identify_image", identify_image) 

# Define the edges
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_request,
    {
        "plan": "extract_user_request",
        "vision": "identify_image", # <-- New path for vision queries
        "chat": "general_response",
    },
)

# The main planning flow is now a longer sequence
workflow.add_edge("extract_user_request", "get_data_for_plan")
workflow.add_edge("get_data_for_plan", "synthesize_custom_plan")
workflow.add_edge("synthesize_custom_plan", "generate_gear_checklist")
workflow.add_edge("generate_gear_checklist", "generate_budget")
workflow.add_edge("generate_budget", "compile_final_response")

# Endpoints of the graph
workflow.add_edge("compile_final_response", END)
workflow.add_edge("general_response", END)
workflow.add_edge("identify_image", END)

# Compile the graph
app = workflow.compile()

# --- 7. Public API Function (Updated to reset state) ---
def get_agent_response(user_input: str, history: List[dict], image_data: Optional[Dict] = None) -> dict:
    """
    The main entry point for the agent.
    It now accepts optional image data.
    """
    messages = messages_from_dict(history)
    
    # The user_input is now just the text part of the prompt
    messages.append(HumanMessage(content=user_input))

    # The initial state for the graph run
    initial_state = {
        "messages": messages,
        "user_request": None,
        "plan": None,
        "gear_checklist": None,
        "budget": None,
        "image_data": image_data, # Pass the image data into the state
    }
    
    final_state = app.invoke(initial_state)

    final_state['messages'] = messages_to_dict(final_state['messages'])
    return final_state