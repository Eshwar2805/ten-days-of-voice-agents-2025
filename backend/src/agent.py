import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# ------------------ Helpers: load & save fraud cases ------------------


def load_fraud_cases() -> List[Dict[str, Any]]:
    """
    Load fraud cases from shared-data/day6_fraud_cases.json.
    """
    base_dir = Path(__file__).resolve().parent.parent  # backend/
    path = base_dir / "shared-data" / "day6_fraud_cases.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"Failed to load day6_fraud_cases.json: {e}")
    return []


def save_fraud_cases(cases: List[Dict[str, Any]]) -> None:
    base_dir = Path(__file__).resolve().parent.parent  # backend/
    path = base_dir / "shared-data" / "day6_fraud_cases.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


class FraudAgent(Agent):
    def __init__(self, cases: List[Dict[str, Any]]) -> None:
        self.cases = cases
        self.current_case_index: Optional[int] = None
        self.verified: bool = False
        self.bank_name = "Falcon Bank"  # fictional bank for demo only

        instructions = f"""
You are a calm, professional fraud detection representative for {self.bank_name}.

Goal:
- Handle a fraud alert call about a suspicious card transaction.
- Verify the customer safely.
- Read the suspicious transaction.
- Ask if it was legitimate.
- Mark the case as safe or fraudulent using the available tools.

VERY IMPORTANT SAFETY RULES:
- You MUST NOT ask for full card numbers, CVV, PIN, OTP, passwords, or any sensitive credentials.
- You may only use:
  - The customer's name,
  - A simple security question from the database,
  - Non-sensitive transaction details (merchant, amount, masked card, city).

Call flow:
1. Greet the user.
   Example: "Hello, this is the fraud monitoring team from {self.bank_name}. Am I speaking with Rahul Sharma?"
2. Ask for the customer's first name to identify the case.
   Once you know their name, call the `load_case_for_user` tool.
3. If no matching case is found:
   - Politely say you don't have an active fraud alert under that name and end the call.
4. If a case is found:
   - Explain that you're calling about a suspicious transaction.
   - Ask the security question from the case by calling `get_security_question`.
   - When the user answers, call `verify_security_answer`.
5. If verification fails:
   - Apologize that you cannot proceed without verification.
   - Call `mark_verification_failed` to update the case.
   - End the call politely.
6. If verification passes:
   - Read out the suspicious transaction by using the data from the current case:
     * merchant name,
     * transaction amount,
     * masked card ending,
     * approximate time and location.
   - Ask: "Did you make this transaction? Yes or no?"
   - Based on the user's response, call `mark_transaction_status` with "safe" or "fraudulent".

7. At the end of the call:
   - Summarize the outcome:
     * If safe: say the transaction is confirmed as legitimate and no further action is needed.
     * If fraud: say the card will be blocked and a dispute will be raised (mock).
     * If verification failed: say you couldn't proceed without verification.
   - Thank the user for their time.

Tool usage:
- Use `load_case_for_user` as soon as you are confident of the user's name.
- Use `get_security_question` once the case is loaded to ask verification.
- Use `verify_security_answer` after user answers the security question.
- Use `mark_transaction_status` after the user confirms or denies the transaction.
- Use `mark_verification_failed` if verification fails.

NEVER:
- Mention tools, JSON, databases, or internal code.
- Show or say raw tool calls like `tool_code`.
- Invent real banks, real card numbers, or real customers.
"""
        super().__init__(instructions=instructions)

    # ------------------ Internal helper ------------------

    def _find_case_by_name(self, user_name: str) -> Optional[int]:
        uname = user_name.strip().lower()
        for idx, case in enumerate(self.cases):
            if case.get("userName", "").strip().lower() == uname:
                return idx
        return None

    def _current_case(self) -> Optional[Dict[str, Any]]:
        if self.current_case_index is None:
            return None
        if 0 <= self.current_case_index < len(self.cases):
            return self.cases[self.current_case_index]
        return None

    def _save_current_case(self) -> None:
        if self.current_case_index is None:
            return
        # writes self.cases back to JSON
        save_fraud_cases(self.cases)

    # ------------------ Tools ------------------

    @function_tool()
    async def load_case_for_user(
        self,
        context: RunContext,
        user_name: str,
    ) -> str:
        """
        Load a fraud case for the given user name (fake).
        If found, set it as the active case for this call.
        """
        idx = self._find_case_by_name(user_name)
        if idx is None:
            return (
                "I couldn't find an active fraud alert under that name. "
                "It's possible there is no suspicious activity on this account."
            )
        self.current_case_index = idx
        self.verified = False
        case = self.cases[idx]
        logger.info(f"[Fraud] Loaded case {case.get('caseId')} for user {case.get('userName')}")
        return (
            f"Thank you. I have located your account, {case.get('userName')}. "
            "Before we proceed, I need to verify your identity with a simple security question."
        )

    @function_tool()
    async def get_security_question(
        self,
        context: RunContext,
    ) -> str:
        """
        Return the security question for the current case.
        """
        case = self._current_case()
        if not case:
            return "I don't have an active fraud case loaded yet."
        question = case.get("securityQuestion") or "I don't have a security question on file."
        return f"For security, please answer this question: {question}"

    @function_tool()
    async def verify_security_answer(
        self,
        context: RunContext,
        answer: str,
    ) -> str:
        """
        Verify the answer to the security question.
        """
        case = self._current_case()
        if not case:
            return "I don't have an active fraud case loaded yet."

        expected = (case.get("securityAnswer") or "").strip().lower()
        given = answer.strip().lower()

        if expected and expected == given:
            self.verified = True
            logger.info(f"[Fraud] Verification passed for case {case.get('caseId')}")
            return "Thank you, your identity is verified. I will now read out the suspicious transaction details."
        else:
            self.verified = False
            logger.info(f"[Fraud] Verification FAILED for case {case.get('caseId')}")
            return (
                "I'm sorry, but the answer doesn't match our records. "
                "For your security, I won't be able to discuss this transaction further."
            )

    @function_tool()
    async def mark_verification_failed(
        self,
        context: RunContext,
    ) -> str:
        """
        Mark the current case as verification_failed and persist to database.
        """
        case = self._current_case()
        if not case:
            return "I don't have an active fraud case loaded yet."
        case["status"] = "verification_failed"
        case["outcomeNote"] = "Verification failed. Could not confirm identity."
        case["lastUpdated"] = datetime.utcnow().isoformat()
        self._save_current_case()
        logger.info(f"[Fraud] Case {case.get('caseId')} marked as verification_failed.")
        return "I have recorded that we could not complete verification on this call."

    @function_tool()
    async def mark_transaction_status(
        self,
        context: RunContext,
        status: str,
    ) -> str:
        """
        Mark the current transaction as safe or fraudulent and persist to database.

        Args:
            status: one of "safe" or "fraudulent"
        """
        case = self._current_case()
        if not case:
            return "I don't have an active fraud case loaded yet."

        status = status.strip().lower()
        if status == "safe":
            case["status"] = "confirmed_safe"
            case["outcomeNote"] = "Customer confirmed the transaction as legitimate."
            spoken = (
                "Thank you for confirming. I have marked this transaction as legitimate, "
                "and no further action is needed on your card."
            )
        elif status == "fraudulent":
            case["status"] = "confirmed_fraud"
            case["outcomeNote"] = "Customer denied the transaction. Card should be blocked and dispute initiated (mock)."
            spoken = (
                "Understood. I have marked this transaction as fraudulent. "
                "We will block this card for your safety and raise a dispute for this charge in our system. "
                "Our team may contact you with next steps."
            )
        else:
            return "I can only mark a transaction as safe or fraudulent."

        case["lastUpdated"] = datetime.utcnow().isoformat()
        self._save_current_case()
        logger.info(f"[Fraud] Case {case.get('caseId')} marked as {case['status']}.")
        return spoken

    @function_tool()
    async def describe_current_transaction(
        self,
        context: RunContext,
    ) -> str:
        """
        Return a natural language description of the suspicious transaction for the current case.
        """
        case = self._current_case()
        if not case:
            return "I don't have an active fraud case loaded yet."

        merchant = case.get("merchantName") or "a merchant"
        amount = case.get("transactionAmount") or "an amount"
        currency = case.get("transactionCurrency") or ""
        masked_card = case.get("maskedCard") or "your card ending in XXXX"
        time = case.get("transactionTime") or "recently"
        location = case.get("transactionLocation") or "your region"
        category = case.get("transactionCategory") or "a purchase"

        desc = (
            f"We detected a {category} transaction at {merchant} for {amount} {currency} "
            f"on your card {masked_card}, around {time}, in {location}. "
            "Did you make this transaction?"
        )
        return desc


# ----------------------- Session Setup -----------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    cases = load_fraud_cases()
    logger.info(f"Day 6 Fraud Agent – loaded {len(cases)} fraud case(s).")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
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
        agent=FraudAgent(cases=cases),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
