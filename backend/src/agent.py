import logging
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------- Helper to load tutor content ----------


def load_tutor_content() -> list[dict]:
    """
    Load concepts from shared-data/day4_tutor_content.json.
    Falls back to a small default list if file is missing.
    """
    try:
        base_dir = Path(__file__).resolve().parent.parent  # backend/
        content_path = base_dir / "shared-data" / "day4_tutor_content.json"
        with content_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"Could not load day4_tutor_content.json, using defaults. Error: {e}")

    # Fallback content
    return [
        {
            "id": "variables",
            "title": "Variables",
            "summary": "Variables store values so you can reuse or change them later.",
            "sample_question": "What is a variable and why is it useful?",
        },
        {
            "id": "loops",
            "title": "Loops",
            "summary": "Loops let you repeat an action multiple times.",
            "sample_question": "Explain the difference between a for loop and a while loop.",
        },
    ]


class TutorAgent(Agent):
    def __init__(self, tutor_content: list[dict]) -> None:
        self.tutor_content = tutor_content
        self.current_mode: str | None = None
        self.current_concept_id: str | None = None

        concepts_list = ", ".join(c["title"] for c in tutor_content)

        instructions = f"""
You are an Active Recall Programming Tutor for beginners.

Your role:
- Help the user learn basic programming concepts by using the “teach-the-tutor” method.
- You always speak in a friendly, encouraging tone.

Available concepts (from our small course file):
{concepts_list}

There are THREE learning modes:

1) learn mode (voice: Matthew)
   - Explain the chosen concept in simple language.
   - Use the summary from the content file, then expand with examples.
   - Check understanding with 1–2 quick questions.

2) quiz mode (voice: Alicia)
   - You ask the user questions about the chosen concept.
   - Use the sample_question field as a starting point.
   - Ask follow-up questions and give quick feedback on each answer.

3) teach_back mode (voice: Ken)
   - Ask the user to explain the concept back to you in their own words.
   - Listen, then give short, qualitative feedback:
     * what they did well
     * what’s missing, in 1–2 sentences
   - Optionally suggest what to review.

VERY IMPORTANT BEHAVIOR:
- At the beginning of the session:
  1) Greet the user.
  2) Ask which concept they want to work on (e.g., "variables" or "loops").
  3) Ask which mode they want: "learn", "quiz", or "teach_back".
- After the user chooses, call the tool `set_mode_and_voice` with the mode and concept id.
- Then behave according to that mode.
- The user can SWITCH MODES at any time by saying things like:
  - "Quiz me on loops"
  - "Let’s learn variables"
  - "Let me teach back variables"
  When they do, call `set_mode_and_voice` again.

TOOL USAGE:
- Use:
  - `set_mode_and_voice` to switch modes and set the concept.
  - `get_concept_summary` to get the concept explanation.
  - `get_concept_question` to get a question for quiz or teach_back prompts.

RULES:
- Never mention tools, JSON, files, or code to the user.
- Do not output code like `tool_code` or function calls in your spoken response.
- Keep explanations and questions short and clear.
- Stay focused on one concept at a time.
"""

        super().__init__(instructions=instructions)

    # ---------- Internal helpers ----------

    def _find_concept(self, concept_id: str | None) -> dict:
        if concept_id:
            for c in self.tutor_content:
                if c.get("id") == concept_id:
                    return c
        # default to first concept
        return self.tutor_content[0]

    # ---------- Tools ----------

    @function_tool()
    async def set_mode_and_voice(
        self,
        context: RunContext,
        mode: str,
        concept_id: str | None = None,
    ) -> str:
        """
        Set the tutor mode (learn, quiz, teach_back) and picked concept.
        Also updates the Murf Falcon voice depending on the mode.

        Args:
            mode: "learn", "quiz", or "teach_back"
            concept_id: one of the IDs from the content file, e.g. "variables", "loops"
        """
        mode = mode.lower().strip()
        valid_modes = ["learn", "quiz", "teach_back"]
        if mode not in valid_modes:
            return (
                "Invalid mode. Please choose one of: learn, quiz, or teach_back."
            )

        concept = self._find_concept(concept_id)
        self.current_mode = mode
        self.current_concept_id = concept.get("id")

        # Try to switch Murf voice depending on mode.
        # Voice names follow the same pattern as your Day 1 setup.
        try:
            if mode == "learn":
                # Matthew
                context.session.tts.update_options(voice="en-US-matthew")
            elif mode == "quiz":
                # Alicia
                context.session.tts.update_options(voice="en-US-alicia")
            else:
                # teach_back -> Ken
                context.session.tts.update_options(voice="en-US-ken")
        except Exception as e:
            logger.warning(f"Failed to update TTS voice for mode={mode}: {e}")

        return f"Switched to {mode} mode for concept '{concept.get('title', 'Unknown')}'."

    @function_tool()
    async def get_concept_summary(
        self,
        context: RunContext,
        concept_id: str | None = None,
    ) -> str:
        """
        Get a short summary of the given concept to explain in learn mode.
        """
        concept = self._find_concept(concept_id or self.current_concept_id)
        title = concept.get("title", "Unknown concept")
        summary = concept.get("summary", "No summary available.")
        return f"{title}: {summary}"

    @function_tool()
    async def get_concept_question(
        self,
        context: RunContext,
        concept_id: str | None = None,
    ) -> str:
        """
        Get a sample question for quiz or teach_back modes.
        """
        concept = self._find_concept(concept_id or self.current_concept_id)
        title = concept.get("title", "Unknown concept")
        question = concept.get("sample_question", "No question available.")
        return f"For {title}: {question}"


# ----------------------- Session Setup -----------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Load tutor content once per process
    tutor_content = load_tutor_content()
    logger.info(f"Loaded {len(tutor_content)} tutor concepts for Day 4 tutor.")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # default; will switch per mode
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=TutorAgent(tutor_content=tutor_content),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
