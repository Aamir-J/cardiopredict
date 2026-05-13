"""
dashboard/chatbot_sql.py
------------------------
DataExplorer — the SQL-aware chatbot for the Analytics page.

This is a sibling of dashboard/chatbot.py but designed for a different
purpose. The Page 1 chatbot reasons about a single patient. THIS chatbot
reasons about the whole dataset by translating natural-language questions
into SQL queries via Claude's tool-use feature.

Architecture:
  user message  -->  Claude (with sql_tool available)
                       |
                       |--- decides whether SQL is needed
                       |--- if yes, generates SQL, calls run_sql_query
                       |--- receives result rows
                       |--- composes natural-language answer with real numbers
                       v
                     final response streamed to UI

Tool-use can take multiple turns (Claude might run several queries to
answer a complex question), which is handled by the agentic loop below.

Cost note: each Q-and-A is roughly 3,000–4,000 tokens because the schema
documentation is large. Approximately $0.02 per question on Sonnet 4.5.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from dashboard.sql_tool import (
    SCHEMA_DOCUMENTATION,
    SQL_TOOL_DEFINITION,
    run_sql_query,
)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are DataExplorer, the data analysis assistant for CardioPredict — a cardiac risk screening system trained on Indian and Western patient data.

# Your role

You help users explore the CardioPredict training dataset by translating their natural-language questions into SQL queries, executing them against the database, and presenting the results in clear English with the actual numbers.

# How you work

When a user asks a data question:
  1. Decide what SQL would answer it.
  2. Call the `run_sql_query` tool with that SQL.
  3. Receive the results.
  4. Compose a clear, concise answer using the actual numbers returned.

For simple definitional or conceptual questions (e.g., "what is recall?"), you don't need to query — just answer directly.

# Database schema available to you

{SCHEMA_DOCUMENTATION}

# Communication style

- Direct, clear, with specific numbers. Avoid hedging.
- When comparing cohorts, lead with the headline finding (e.g., "Indian patients have 33% higher mean cholesterol than the UCI cohort: 329 vs 247 mg/dL").
- Round numbers sensibly (e.g., 328.725 → "329"; percentages to 1 decimal).
- For categorical comparisons, give absolute counts AND percentages.
- Acknowledge limitations honestly — if the data can't answer a question, say so.
- When relevant, mention the Indian-context implication of the finding.

# Formatting rules

- Use bold (**text**) for emphasis and key numbers.
- Use bullet points for lists of findings.
- DO NOT use markdown headings (# or ##) in your responses — they render too large in the chat UI. Use bold inline labels instead.
- Keep responses concise. Most answers fit in 2-4 short paragraphs or a short bullet list.

# Important boundaries

- This database does NOT contain personally identifying information. Patient IDs were stripped during cleaning.
- You can only execute SELECT queries — INSERT/UPDATE/DELETE/DROP etc are blocked at the OS level. If a user asks you to modify data, explain that the database is read-only.
- If a user asks something out of scope (e.g., generating model predictions for a patient — that's Page 1's job), redirect them to the Risk Calculator page.
"""


# ---------------------------------------------------------------------------
# Client (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def _client_singleton() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in environment. Check .env or "
            "Streamlit Cloud secrets configuration."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# The tool-use agentic loop
# ---------------------------------------------------------------------------
def _run_agent_turn(messages: list[dict], placeholder) -> str:
    """
    Run one full user-question turn through the agentic loop.

    This may involve multiple Claude API calls if Claude decides to issue
    several SQL queries before composing its final answer. Each tool call
    is executed via run_sql_query and the result is fed back to Claude.

    Returns the final assistant text once Claude stops calling tools.
    The placeholder is updated progressively to show what's happening.
    """
    client = _client_singleton()

    # The 'messages' list will grow as we add tool_use and tool_result blocks.
    # We loop until Claude returns a non-tool-use response (i.e., stop_reason
    # is "end_turn" rather than "tool_use").
    max_iterations = 6  # guard against runaway tool calling
    iteration = 0

    final_text = ""

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SQL_TOOL_DEFINITION],
            messages=messages,
        )

        # Add Claude's response to the message history for the next iteration
        # (this is required by the API — every assistant turn must be in history
        # before we can append tool_result blocks for the next call).
        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        # Check whether Claude used any tools in this response
        tool_use_blocks = [
            block for block in response.content
            if block.type == "tool_use"
        ]
        text_blocks = [
            block for block in response.content
            if block.type == "text"
        ]

        # Show any text Claude produced this turn
        text_this_turn = "\n".join(b.text for b in text_blocks)
        if text_this_turn:
            final_text += text_this_turn + "\n"
            placeholder.markdown(final_text)

        # If no tools were called, we're done — Claude has its final answer
        if not tool_use_blocks:
            break

        # Otherwise: execute each tool call and add the results to messages
        # so Claude can see them on the next iteration.
        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input

            # Show the user what's happening (transparent reasoning)
            sql_preview = tool_input.get("sql", "")
            placeholder.markdown(
                final_text
                + f"\n\n_Running query:_ `{sql_preview[:120]}`{'...' if len(sql_preview) > 120 else ''}_"
            )

            if tool_name == "run_sql_query":
                result = run_sql_query(tool_input["sql"])
            else:
                result = {"ok": False, "error": f"Unknown tool: {tool_name}"}

            # Encode the result for Claude
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })

        # Feed tool results back as a user message
        messages.append({
            "role": "user",
            "content": tool_results,
        })

    # Clean up the placeholder — show final text without the running-query suffix
    placeholder.markdown(final_text.strip())
    return final_text.strip()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------
def render_sql_chatbot(
    session_key: str = "page2_sql_chat",
    intro_message: str = (
        "Ask me anything about the dataset. I can run SQL queries to "
        "compute averages, distributions, comparisons between the Indian "
        "and UCI cohorts, and more."
    ),
) -> None:
    """
    Render the SQL-aware chatbot. Chat history persists across reruns via
    session_state under the given session_key.
    """
    # Initialise history
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    history = st.session_state[session_key]

    # Empty-state intro
    if not history:
        st.caption(f"💬 {intro_message}")

        # Suggested questions to get the user started
        with st.expander("💡 Try these questions to start", expanded=False):
            st.markdown(
                """
                - What's the average cholesterol in each cohort?
                - How many patients over 60 have heart disease?
                - What percentage of Indian patients have exercise-induced angina?
                - Compare resting blood pressure between Indian and UCI cohorts.
                - What's the disease prevalence by age group?
                - How does the median oldpeak differ between cohorts?
                """
            )

    # Render existing turns
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask a question about the dataset...")

    if user_input:
        # Show the user's message immediately
        history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build the message history for the API call. We keep our local
        # history in clean (role, content-string) form, but the API needs
        # the raw turn structure including any tool_use/tool_result blocks
        # from previous turns. For simplicity and to keep tool-use state
        # ephemeral, we send only the plain user/assistant text history.
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]

        # Run the agent loop and stream output to the chat
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                final_text = _run_agent_turn(api_messages, placeholder)
                history.append({
                    "role": "assistant",
                    "content": final_text or "_(no response)_",
                })
            except Exception as exc:
                error_msg = (
                    f"⚠️ Could not reach the Claude API: {exc}\n\n"
                    f"Check that your ANTHROPIC_API_KEY is set correctly."
                )
                placeholder.error(error_msg)


def reset_sql_chat(session_key: str = "page2_sql_chat") -> None:
    """Clear the SQL chat history for a given session key."""
    if session_key in st.session_state:
        st.session_state[session_key] = []