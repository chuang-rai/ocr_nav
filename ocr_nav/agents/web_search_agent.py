"""Web-search agent using Playwright and the snapshot → ref → act cycle.

This agent browses the web like a human:
1. Navigates to a URL
2. Takes accessibility-tree snapshots (with [ref=eNN] markers)
3. Interacts with elements by ref (click, type, fill, press …)
4. Reads results from subsequent snapshots

The ``BrowserSession`` is kept alive across tool calls inside a single
agent run so the agent can perform multi-step browsing workflows
(e.g. fill a form, submit, read results).

Dependencies
------------
* ``playwright`` – ``pip install playwright && playwright install chromium``

Example
-------
>>> import asyncio
>>> from ocr_nav.agents.web_search_agent import WebSearchDeps, build_agent
>>> agent = build_agent()
>>> deps = WebSearchDeps(query_text="SBB train Oerlikon to Bern")
>>> result = asyncio.run(agent.run("Find the next train from Oerlikon to Bern on sbb.ch", deps=deps))
>>> print(result.output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent, RunContext
from termcolor import cprint

from ocr_nav.tools.browser_tools import BrowserSession

# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are a web-browsing assistant. You interact with web pages through a
headless browser using the **snapshot → ref → act** cycle.

## Available tools

| Tool | Purpose |
|------|---------|
| `navigate` | Open a URL in the browser |
| `snapshot` | Get a text representation of the current page's accessibility tree. \
Elements you can interact with are tagged `[ref=eNN]`. |
| `act` | Interact with an element by its ref: click, type, fill, press, \
select, hover, scroll_down, scroll_up, wait |
| `get_page_text` | Get the full visible text content of the page (useful after snapshot \
shows the page has the info you need) |

## Interaction protocol

1. Call `navigate` to open the target URL.
2. Call `snapshot` to see the page structure.
3. Use `act` with the appropriate `kind` and `ref` from the snapshot to
   interact (type into fields, click buttons, etc.).
4. After each action that changes the page, call `snapshot` again to
   see the updated state.
5. Repeat until you have the information the user asked for.

## Rules

- ALWAYS call `snapshot` after navigating or performing an action that
  changes the page.
- Use `ref` values ONLY from the most recent snapshot – they are
  regenerated on every snapshot.
- When typing into search/autocomplete fields, use `kind: "type"` with
  `slowly=true` semantics (built-in 50 ms delay) to trigger suggestions.
  Then snapshot and click the right suggestion.
- If a page requires scrolling to see more content, use
  `kind: "scroll_down"` or `kind: "scroll_up"`.
- Be concise in your final answer. Extract the specific information the
  user asked for.
- If you encounter cookie banners or popups, dismiss them by clicking
  the appropriate button before proceeding.
"""


# ------------------------------------------------------------------ #
# Deps
# ------------------------------------------------------------------ #


@dataclass
class WebSearchDeps:
    """Runtime dependencies for the web-search agent."""

    query_text: str = ""
    headless: bool = True
    # The browser session is created lazily and shared across tool calls
    _session: BrowserSession | None = field(default=None, repr=False, init=False)

    async def get_session(self) -> BrowserSession:
        """Return the shared browser session, starting it if needed."""
        if self._session is None:
            self._session = BrowserSession(headless=self.headless)
            await self._session.start()
        return self._session

    async def cleanup(self) -> None:
        """Close the browser session."""
        if self._session is not None:
            await self._session.close()
            self._session = None


# ------------------------------------------------------------------ #
# Agent construction
# ------------------------------------------------------------------ #


def build_agent(
    model_name: str = "google-gla:gemini-2.0-flash",
) -> Agent[WebSearchDeps, str]:
    """Build a Pydantic-AI agent that browses the web via Playwright.

    The agent exposes four tools – ``navigate``, ``snapshot``, ``act``,
    and ``get_page_text`` – which together implement the
    **snapshot → ref → act** cycle described in the system prompt.
    """

    agent: Agent[WebSearchDeps, str] = Agent(
        model_name,
        system_prompt=_SYSTEM_PROMPT,
        deps_type=WebSearchDeps,
        retries=3,
    )

    # -- navigate ---------------------------------------------------------- #

    @agent.tool
    async def navigate(ctx: RunContext[WebSearchDeps], url: str) -> str:
        """Navigate the browser to the given URL.

        Args:
            url: The full URL to open (e.g. "https://www.sbb.ch").
        """
        session = await ctx.deps.get_session()
        return await session.navigate(url)

    # -- snapshot ---------------------------------------------------------- #

    @agent.tool
    async def snapshot(ctx: RunContext[WebSearchDeps]) -> str:
        """Capture the current page's accessibility tree.

        Returns a text representation where interactive elements are tagged
        with [ref=eNN]. Use these refs in subsequent `act` calls.
        """
        session = await ctx.deps.get_session()
        text, _ = await session.snapshot()
        # Truncate very large snapshots to stay within context limits
        max_chars = 15_000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
        return text

    # -- act --------------------------------------------------------------- #

    @agent.tool
    async def act(
        ctx: RunContext[WebSearchDeps],
        kind: Literal[
            "click",
            "type",
            "fill",
            "press",
            "select",
            "hover",
            "scroll_down",
            "scroll_up",
            "wait",
        ],
        ref: str = "",
        text: str = "",
        key: str = "",
    ) -> str:
        """Interact with a page element identified by its ref from the last snapshot.

        Args:
            kind: The action to perform. One of: click, type, fill, press,
                  select, hover, scroll_down, scroll_up, wait.
            ref: The element reference from the last snapshot (e.g. "e3").
                 Required for click, type, fill, press, select, hover.
            text: Text to type/fill, or milliseconds to wait.
            key: Key name for press actions (e.g. "Enter", "Tab").
        """
        session = await ctx.deps.get_session()
        return await session.act(
            kind=kind,
            ref=ref or None,
            text=text or None,
            key=key or None,
        )

    # -- get_page_text ----------------------------------------------------- #

    @agent.tool
    async def get_page_text(ctx: RunContext[WebSearchDeps]) -> str:
        """Get the full visible text content of the current page.

        Useful for reading detailed content after you've navigated to the
        right page. Returns up to 20 000 characters.
        """
        session = await ctx.deps.get_session()
        page = session.page
        text = await page.inner_text("body")
        max_chars = 20_000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
        return text

    return agent


# ------------------------------------------------------------------ #
# Convenience runner
# ------------------------------------------------------------------ #


async def search_web(query: str, model_name: str = "google-gla:gemini-2.0-flash") -> str:
    """High-level helper: run a web-search query and return the answer.

    Handles browser lifecycle automatically.
    """
    agent = build_agent(model_name)
    deps = WebSearchDeps(query_text=query)
    try:
        result = await agent.run(query, deps=deps)
        return result.output
    finally:
        await deps.cleanup()
