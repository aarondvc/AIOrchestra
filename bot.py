import asyncio
import os
import uuid
import discord
from discord.ext import commands
from discord.ui import Button, View
from dotenv import load_dotenv
from agent_graph import agent_app, pool
from langgraph.types import Command

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def run_agent_start(task_text, thread_id):
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Health check pool before starting graph streaming
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        print(f"[Database] Pool connection check failed, reconnecting: {e}")
        pool.open()

    for _ in agent_app.stream({"task": task_text, "thread_id": thread_id}, thread_config):
        pass

    return agent_app.get_state(thread_config)

def run_agent_resume(thread_id: str, approved: bool):
    """Resumes the graph using persisted state until completion or the next interrupt."""
    thread_config = {"configurable": {"thread_id": thread_id}}

    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        print(f"[Database] Pool connection check failed during resume: {e}")
        pool.open()

    for _ in agent_app.stream(Command(resume={"approved": approved}), thread_config):
        pass

    return agent_app.get_state(thread_config)

class TaskApprovalView(View):
    def __init__(self, thread_id: str):
        super().__init__(timeout=None)
        self.add_item(Button(
            label="Approve",
            style=discord.ButtonStyle.success,
            emoji="✅",
            custom_id=f"agent_approve:{thread_id}"
        ))
        self.add_item(Button(
            label="Reject",
            style=discord.ButtonStyle.danger,
            emoji="❌",
            custom_id=f"agent_reject:{thread_id}"
        ))

@bot.event
async def on_interaction(interaction: discord.Interaction):
    # Only process component interactions (buttons)
    if interaction.type != discord.InteractionType.component:
        return

    custom_id = interaction.data.get("custom_id", "")
    if not (custom_id.startswith("agent_approve:") or custom_id.startswith("agent_reject:")):
        return

    # Acknowledge immediately
    await interaction.response.defer()

    action, thread_id = custom_id.split(":", 1)
    is_approved = (action == "agent_approve")

    # Disable buttons on the current prompt message
    if interaction.message:
        view = View.from_message(interaction.message)
        for child in view.children:
            child.disabled = True
        await interaction.message.edit(view=view)

    target_channel = interaction.channel or await bot.fetch_channel(interaction.channel_id)

    # Resume the LangGraph thread in a worker thread
    state = await asyncio.to_thread(run_agent_resume, thread_id, is_approved)
    values = state.values if hasattr(state, "values") else {}

    # 1. Post submission confirmation if available
    confirm_shot = values.get("confirmation_screenshot", "")
    result_text = values.get("execution_result", "Completed")

    if is_approved and confirm_shot and os.path.exists(confirm_shot):
        embed = discord.Embed(
            title="Application Submission Confirmed",
            description=f"**Status:** {result_text}\n**Thread ID:** `{thread_id}`",
            color=discord.Color.green()
        )
        file = discord.File(confirm_shot, filename="confirmation.png")
        embed.set_image(url="attachment://confirmation.png")
        await target_channel.send(embed=embed, file=file)

    # 2. Check if another step is waiting for human approval (Job 2, 3, etc.)
    if state.tasks and state.tasks[0].interrupts:
        payload = state.tasks[0].interrupts[0].value
        worker = payload.get("worker", "career_worker")
        draft = str(payload.get("draft", ""))
        screenshot = payload.get("screenshot_path", "")

        max_draft_len = 3800
        if len(draft) > max_draft_len:
            draft = draft[:max_draft_len] + "\n\n... *(Description truncated for Discord)*"

        embed = discord.Embed(
            title="Human Approval Required",
            description=f"**Worker Assigned:** `{worker}`\n**Thread ID:** `{thread_id}`\n\n{draft}",
            color=discord.Color.blue()
        )

        next_view = TaskApprovalView(thread_id=thread_id)
        if screenshot and os.path.exists(screenshot):
            review_file = discord.File(screenshot, filename="review.png")
            embed.set_image(url="attachment://review.png")
            await target_channel.send(embed=embed, file=review_file, view=next_view)
        else:
            await target_channel.send(embed=embed, view=next_view)

    # 3. All items in the batch are finished
    elif not state.next:
        completed = values.get("completed_count", 0)
        target = values.get("target_count", 1)
        if values.get("action_type") == "job_application":
            await target_channel.send(f"🏁 **Batch Finished:** Successfully processed `{completed}/{target}` applications.")
        else:
            verdict = "Approved" if is_approved else "Rejected"
            await target_channel.send(f"**{verdict}:** {result_text}")

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("Global approval gateway active.")

@bot.command(name="task")
async def handle_task(ctx, *, task_text: str):
    thread_id = f"task_{uuid.uuid4().hex[:6]}"
    status_msg = await ctx.send(f"⚙️ Routing task `{task_text}`...")

    state = await asyncio.to_thread(run_agent_start, task_text, thread_id)

    if state.tasks and state.tasks[0].interrupts:
        payload = state.tasks[0].interrupts[0].value
        worker = payload.get("worker", "Unknown")
        draft = str(payload.get("draft", ""))
        screenshot = payload.get("screenshot_path", "")

        max_draft_len = 3800
        if len(draft) > max_draft_len:
            draft = draft[:max_draft_len] + "\n\n... *(Description truncated for Discord)*"

        embed = discord.Embed(
            title="Human Approval Required",
            description=f"**Worker Assigned:** `{worker}`\n**Thread ID:** `{thread_id}`\n\n{draft}",
            color=discord.Color.green() if screenshot else discord.Color.blue()
        )

        view = TaskApprovalView(thread_id=thread_id)
        await status_msg.delete()

        if screenshot and os.path.exists(screenshot):
            file = discord.File(screenshot, filename="review.png")
            embed.set_image(url="attachment://review.png")
            await ctx.send(embed=embed, file=file, view=view)
        else:
            await ctx.send(embed=embed, view=view)
    else:
        final_values = state.values if hasattr(state, "values") else {}
        draft = final_values.get("draft_artifact", "")
        if "Error" in draft:
            await status_msg.edit(content=f"⚠️ **Execution Stopped:**\n{draft}")
        else:
            await status_msg.edit(content="Task completed with no approval needed.")

if __name__ == "__main__":
    bot.run(TOKEN)