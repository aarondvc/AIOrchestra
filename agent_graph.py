import os
import re
import json
import asyncio
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from linkedin_engine import search_and_prep_easy_apply, submit_confirmed_application
from career_tools import fetch_job_details, load_user_profile, save_document_artifact

load_dotenv()

class AgentState(TypedDict):
    task: str
    target_worker: str
    draft_artifact: str
    is_approved: bool
    execution_result: str
    thread_id: str
    screenshot_path: str
    confirmation_screenshot: str
    job_url: str
    company: str
    action_type: str
    target_count: int
    completed_count: int
    applied_job_ids: list[str]
    current_job: dict

def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts).strip()
    return str(content).strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def orchestrator(state: AgentState):
    system_instruction = (
        "You are the orchestrator. Analyze the user request and output a valid JSON object with:\n"
        "1. 'worker': One of 'dev_worker', 'career_worker', or 'media_worker'.\n"
        "   - If the request involves jobs, applying, LinkedIn, hiring, careers, or finding work, choose 'career_worker'.\n"
        "   - If the request involves writing code, scripts, software architecture, or debugging, choose 'dev_worker'.\n"
        "   - If the request involves social media, video scripts, or viral content, choose 'media_worker'.\n"
        "2. 'count': An integer indicating how many jobs to apply to if specified (e.g. 'two' -> 2, '3' -> 3). Default to 1.\n"
        "Output ONLY the JSON object. Example: {\"worker\": \"career_worker\", \"count\": 2}"
    )
    user_task = f"Task: {state['task']}"
    response = llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_task)
    ])
    
    raw_content = extract_text(response.content)
    # Strip markdown code fencing if returned
    clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()

    try:
        data = json.loads(clean_json)
        worker = data.get("worker", "career_worker").lower()
        target_count = int(data.get("count", 1))
    except Exception:
        worker = "career_worker" if any(k in state["task"].lower() for k in ["apply", "job", "linkedin"]) else "dev_worker"
        target_count = 1

    return {
        "target_worker": worker,
        "target_count": target_count,
        "completed_count": state.get("completed_count", 0),
        "applied_job_ids": state.get("applied_job_ids", [])
    }

def route_worker(state: AgentState) -> Literal["dev_worker", "career_worker", "media_worker"]:
    target = state.get("target_worker", "career_worker")
    if target in ["dev_worker", "career_worker", "media_worker"]:
        return target
    return "career_worker"

def clean_job_title(raw_text: str) -> str:
    """Removes counts, digits, and conversational prompt wrappers to isolate the actual job title."""
    pattern = r"(?i)\b(apply to a|apply to an|apply to one|apply to|apply for a|apply for an|apply for|find a job for|search for|jobs on linkedin|on linkedin|jobs|job|role|positions|position|one|two|three|four|five|\d+)\b"
    cleaned = re.sub(pattern, "", raw_text)
    # Strip any stray leading numbers or punctuation
    cleaned = re.sub(r"^[\s\d\-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) > 1 else "Software Engineer"

def career_worker(state: AgentState):
    task_lower = state["task"].lower()

    # Route: Automated Job Application Submission
    if any(k in task_lower for k in ["apply", "linkedin", "easy apply", "search job"]):
        role_target = clean_job_title(state["task"])
        already_applied = state.get("applied_job_ids", [])
        current_step = state.get("completed_count", 0) + 1
        total_steps = state.get("target_count", 1)

        print(f"\n[Career Worker] Processing job {current_step}/{total_steps} for keyword: '{role_target}'")
        print(f"[Career Worker] Excluding already applied IDs: {already_applied}")

        app_result = asyncio.run(
            search_and_prep_easy_apply(role=role_target, excluded_ids=already_applied)
        )

        if app_result.get("error"):
            return {
                "action_type": "error",
                "draft_artifact": f"Error preparing application: {app_result['error']}",
                "screenshot_path": ""
            }

        summary = (
            f"**Platform:** LinkedIn Easy Apply ({current_step}/{total_steps})\n"
            f"**Role:** {app_result['title']}\n"
            f"**Company:** {app_result['company']}\n"
            f"Review the attached screenshot of the filled form. Click Approve to finalize submission."
        )
        return {
            "action_type": "job_application",
            "draft_artifact": summary,
            "screenshot_path": app_result.get("screenshot", ""),
            "job_url": app_result.get("job_url", ""),
            "company": app_result.get("company", ""),
            "current_job": app_result
        }

    # Route: Cover Letter & Resume Tailoring (Default fallback)
    user_profile = load_user_profile()
    job_content = fetch_job_details(state["task"])
    prompt = f"Tailor a cover letter and bullets for: {job_content}\nProfile: {user_profile}"
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "action_type": "document_draft",
        "draft_artifact": extract_text(response.content),
        "screenshot_path": ""
    }

def dev_worker(state: AgentState):
    prompt = f"Draft an architecture and code implementation plan for: {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft_artifact": extract_text(response.content)}

def media_worker(state: AgentState):
    prompt = f"Draft a 30-second viral TikTok script with visual hooks for: {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft_artifact": extract_text(response.content)}

def human_approval(state: AgentState):
    if state.get("action_type") == "error":
        return {"is_approved": False}

    approval_data = interrupt({
        "worker": state["target_worker"],
        "draft": state["draft_artifact"],
        "screenshot_path": state.get("screenshot_path", "")
    })
    return {"is_approved": approval_data.get("approved", False)}

def route_approval(state: AgentState) -> Literal["execute_action", "handle_rejection"]:
    if state.get("action_type") == "error":
        return "handle_rejection"
    return "execute_action" if state.get("is_approved") else "handle_rejection"

def execute_action(state: AgentState):
    worker = state.get("target_worker")
    
    if worker == "career_worker":
        if state.get("action_type") == "job_application":
            job_url = state.get("job_url", "")
            company = state.get("company", "Company")
            
            sub_res = asyncio.run(submit_confirmed_application(job_url, company))
            
            # Extract job_id from current url to avoid applying to it again in this run
            applied_ids = list(state.get("applied_job_ids", []))
            id_match = re.search(r"currentJobId=(\d+)", job_url) or re.search(r"view/(\d+)", job_url)
            if id_match:
                applied_ids.append(id_match.group(1))

            new_completed = state.get("completed_count", 0) + 1

            return {
                "execution_result": sub_res.get("status", "Submitted"),
                "confirmation_screenshot": sub_res.get("confirmation_screenshot", ""),
                "completed_count": new_completed,
                "applied_job_ids": applied_ids
            }
        else:
            out_path = save_document_artifact(state.get("thread_id", "latest"), state["draft_artifact"])
            return {"execution_result": f"Document exported to `{out_path}`."}

    return {"execution_result": "Action executed successfully."}

def handle_rejection(state: AgentState):
    return {"execution_result": "Action rejected by human operator. Batch stopped."}

def route_next_step(state: AgentState) -> Literal["career_worker", "__end__"]:
    """Determines whether to continue applying to more jobs or terminate the graph."""
    if state.get("action_type") == "job_application":
        completed = state.get("completed_count", 0)
        target = state.get("target_count", 1)
        
        if completed < target:
            print(f"[Orchestrator] Batch Progress: {completed}/{target} complete. Routing to next application.")
            return "career_worker"
        
        print(f"[Orchestrator] Batch completed: {completed}/{target} applications finished.")
        return END

    return END

# Build graph
builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("career_worker", career_worker)
builder.add_node("dev_worker", dev_worker)
builder.add_node("media_worker", media_worker)
builder.add_node("human_approval", human_approval)
builder.add_node("execute_action", execute_action)
builder.add_node("handle_rejection", handle_rejection)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", route_worker)
builder.add_edge("career_worker", "human_approval")
builder.add_edge("dev_worker", "human_approval")
builder.add_edge("media_worker", "human_approval")
builder.add_conditional_edges("human_approval", route_approval)

# Conditional loop from execute_action back to career_worker if remaining target > 0
builder.add_conditional_edges("execute_action", route_next_step, {
    "career_worker": "career_worker",
    END: END
})
builder.add_edge("handle_rejection", END)

# PostgreSQL Checkpointer setup
DB_URI = os.getenv("DATABASE_URL")

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    kwargs={
        "autocommit": True,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

agent_app = builder.compile(checkpointer=checkpointer)