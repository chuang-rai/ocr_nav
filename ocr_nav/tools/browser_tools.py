"""Playwright-based browser tools for Pydantic-AI agents.

Implements the **snapshot → ref → act** interaction cycle:
1. ``navigate``  – open a URL in a headless Chromium instance.
2. ``snapshot``  – capture the page's accessibility tree as a flat text
   representation with ``[ref=eNN]`` markers.
3. ``act``       – interact with an element identified by its ref
   (click, type, fill, press, select, hover, scroll, wait …).

Dependencies
------------
* ``playwright`` – ``pip install playwright && playwright install chromium``
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from termcolor import cprint

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

_REF_COUNTER: int = 0


def _reset_ref_counter() -> None:
    global _REF_COUNTER
    _REF_COUNTER = 0


def _next_ref() -> str:
    global _REF_COUNTER
    _REF_COUNTER += 1
    return f"e{_REF_COUNTER}"


def _walk_ax_tree(node: dict[str, Any], depth: int = 0) -> list[str]:
    """Recursively walk an accessibility-tree node and produce snapshot lines.

    Each interactive element gets a ``[ref=eNN]`` tag so the agent can
    reference it in subsequent ``act`` calls.
    """
    lines: list[str] = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    # Skip noise nodes that clutter the snapshot
    skip_roles = {"none", "generic", "InlineTextBox", "StaticText"}
    if role in skip_roles:
        # Still recurse into children (they may be interactive)
        for child in node.get("children", []):
            lines.extend(_walk_ax_tree(child, depth))
        return lines

    # Assign refs to interactive roles
    interactive_roles = {
        "link",
        "button",
        "textbox",
        "combobox",
        "searchbox",
        "checkbox",
        "radio",
        "menuitem",
        "tab",
        "option",
        "slider",
        "spinbutton",
        "switch",
    }
    ref_tag = ""
    ref_id = ""
    if role in interactive_roles:
        ref_id = _next_ref()
        ref_tag = f" [ref={ref_id}]"

    indent = "  " * depth
    display = name or value
    if display:
        line = f"{indent}{role}: {display}{ref_tag}"
    elif role:
        line = f"{indent}{role}{ref_tag}"
    else:
        line = ""

    if line:
        lines.append(line)
    # Store ref_id on the dict so callers can build a ref→node map
    if ref_id:
        node["__ref__"] = ref_id

    for child in node.get("children", []):
        lines.extend(_walk_ax_tree(child, depth + 1))
    return lines


def _cdp_nodes_to_tree(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Convert the flat CDP ``Accessibility.getFullAXTree`` node list into
    a nested dict that :func:`_walk_ax_tree` can consume.

    Each CDP node has ``nodeId``, ``role.value``, ``name.value`` and
    ``childIds``.  We rebuild a tree with keys ``role``, ``name``,
    ``value``, and ``children``.
    """
    if not nodes:
        return None

    lookup: dict[str, dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("nodeId", "")
        role_val = n.get("role", {}).get("value", "")
        name_val = n.get("name", {}).get("value", "")
        value_val = n.get("value", {}).get("value", "") if isinstance(n.get("value"), dict) else ""
        # Skip ignored / invisible nodes
        if role_val in ("none", "generic", "InlineTextBox") and not name_val:
            # still register so children can be found
            lookup[nid] = {
                "role": role_val,
                "name": name_val,
                "value": value_val,
                "children": [],
                "_child_ids": [c for c in n.get("childIds", [])],
                "_skip": True,
            }
            continue
        lookup[nid] = {
            "role": role_val,
            "name": name_val,
            "value": value_val,
            "children": [],
            "_child_ids": [c for c in n.get("childIds", [])],
            "_skip": False,
        }

    # Wire up children
    for entry in lookup.values():
        for cid in entry.get("_child_ids", []):
            child = lookup.get(cid)
            if child is not None:
                entry["children"].append(child)

    # Root is the first node
    root = lookup.get(nodes[0].get("nodeId", ""))
    return root


# ---------------------------------------------------------------------------
# BrowserSession – thin wrapper around a single Playwright page
# ---------------------------------------------------------------------------


@dataclass
class BrowserSession:
    """Manages a single Playwright Chromium page with headless mode.

    Typical lifecycle::

        session = BrowserSession()
        await session.start()
        await session.navigate("https://example.com")
        snapshot_text, ref_map = await session.snapshot()
        await session.act("click", "e3")
        ...
        await session.close()
    """

    headless: bool = True
    _pw_context_manager: Any = field(default=None, repr=False, init=False)
    _pw: Any = field(default=None, repr=False, init=False)
    _browser: Any = field(default=None, repr=False, init=False)  # Browser | None
    _context: Any = field(default=None, repr=False, init=False)  # BrowserContext | None
    _page: Any = field(default=None, repr=False, init=False)  # Page | None
    _cdp_session: Any = field(default=None, repr=False, init=False)
    _ref_map: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False, init=False)

    # -- lifecycle --------------------------------------------------------- #

    async def start(self) -> None:
        """Launch browser and create a fresh page."""
        from playwright.async_api import async_playwright

        self._pw_context_manager = async_playwright()
        self._pw = await self._pw_context_manager.__aenter__()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        self._cdp_session = await self._context.new_cdp_session(self._page)

    async def close(self) -> None:
        """Tear down browser."""
        if self._cdp_session:
            await self._cdp_session.detach()
        if self._browser:
            await self._browser.close()
        if self._pw_context_manager:
            await self._pw_context_manager.__aexit__(None, None, None)
        self._browser = None
        self._context = None
        self._page = None
        self._pw = None
        self._pw_context_manager = None

    @property
    def page(self) -> Page:
        assert self._page is not None, "BrowserSession not started – call start() first"
        return self._page

    # -- navigate ---------------------------------------------------------- #

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """Navigate to *url* and return the final URL after redirects."""
        cprint(f"[browser] navigating to {url}", "cyan")
        resp = await self.page.goto(url, wait_until=wait_until, timeout=30_000)
        # Give JS-heavy SPAs a moment to hydrate
        await self.page.wait_for_timeout(1500)
        status = resp.status if resp else "?"
        final_url = self.page.url
        return f"Navigated to {final_url} (status {status})"

    # -- snapshot ---------------------------------------------------------- #

    async def snapshot(self) -> tuple[str, dict[str, dict[str, Any]]]:
        """Return a textual accessibility-tree snapshot with ref tags.

        Returns
        -------
        snapshot_text : str
            Human-readable, indented text representation of the page.
        ref_map : dict[str, dict]
            Mapping from ref id (``"e3"``) to the AX tree node dict.
        """
        _reset_ref_counter()
        # Use CDP to get the full accessibility tree (works on all Playwright versions)
        cdp_result = await self._cdp_session.send("Accessibility.getFullAXTree")
        cdp_nodes = cdp_result.get("nodes", [])
        ax_tree = _cdp_nodes_to_tree(cdp_nodes)
        if ax_tree is None:
            return "(empty page – no accessibility tree)", {}

        lines = _walk_ax_tree(ax_tree)
        # Build ref_map from annotated nodes
        ref_map: dict[str, dict[str, Any]] = {}
        self._collect_refs(ax_tree, ref_map)
        self._ref_map = ref_map

        snapshot_text = "\n".join(lines)
        cprint(f"[browser] snapshot captured ({len(lines)} lines, {len(ref_map)} refs)", "cyan")
        return snapshot_text, ref_map

    @staticmethod
    def _collect_refs(node: dict[str, Any], out: dict[str, dict[str, Any]]) -> None:
        if "__ref__" in node:
            out[node["__ref__"]] = node
        for child in node.get("children", []):
            BrowserSession._collect_refs(child, out)

    # -- act --------------------------------------------------------------- #

    ActKind = Literal[
        "click",
        "type",
        "fill",
        "press",
        "select",
        "hover",
        "scroll_down",
        "scroll_up",
        "wait",
    ]

    async def act(
        self,
        kind: ActKind,
        ref: str | None = None,
        text: str | None = None,
        key: str | None = None,
    ) -> str:
        """Perform an interaction on an element identified by *ref*.

        Parameters
        ----------
        kind : str
            One of click, type, fill, press, select, hover, scroll_down,
            scroll_up, wait.
        ref : str | None
            Element ref from a prior snapshot (e.g. ``"e3"``).
        text : str | None
            Text payload for type / fill / select.
        key : str | None
            Key name for press (e.g. ``"Enter"``).
        """
        cprint(f"[browser] act kind={kind} ref={ref} text={text!r} key={key!r}", "cyan")
        page = self.page

        if kind == "wait":
            await page.wait_for_timeout(int(float(text or "2000")))
            return "Waited."

        if kind in ("scroll_down", "scroll_up"):
            delta = 600 if kind == "scroll_down" else -600
            await page.mouse.wheel(0, delta)
            await page.wait_for_timeout(400)
            return f"Scrolled {'down' if delta > 0 else 'up'}."

        # All remaining kinds require a ref
        if ref is None:
            return "Error: ref is required for this action kind."

        locator = self._ref_to_locator(ref)
        if locator is None:
            return f"Error: ref {ref!r} not found in last snapshot."

        try:
            if kind == "click":
                await locator.click(timeout=5000)
                await page.wait_for_timeout(800)
                return f"Clicked ref={ref}."

            if kind == "type":
                await locator.click(timeout=5000)
                await locator.type(text or "", delay=50)
                await page.wait_for_timeout(500)
                return f"Typed {text!r} into ref={ref}."

            if kind == "fill":
                await locator.fill(text or "", timeout=5000)
                await page.wait_for_timeout(300)
                return f"Filled ref={ref} with {text!r}."

            if kind == "press":
                await locator.press(key or "Enter", timeout=5000)
                await page.wait_for_timeout(500)
                return f"Pressed {key!r} on ref={ref}."

            if kind == "select":
                await locator.select_option(text or "", timeout=5000)
                return f"Selected {text!r} on ref={ref}."

            if kind == "hover":
                await locator.hover(timeout=5000)
                return f"Hovered ref={ref}."

        except Exception as exc:
            return f"Error performing {kind} on ref={ref}: {exc}"

        return f"Unknown action kind: {kind}"

    # -- internal ---------------------------------------------------------- #

    def _ref_to_locator(self, ref: str):
        """Resolve a snapshot ref to a Playwright locator.

        Strategy: use the node's *name* from the accessibility tree to find
        the element via ``get_by_role`` or fall back to ``get_by_text``.
        """
        node = self._ref_map.get(ref)
        if node is None:
            return None

        role = node.get("role", "")
        name = node.get("name", "")
        page = self.page

        # Prefer get_by_role with exact name for precision
        if role and name:
            try:
                loc = page.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
                return loc
            except Exception:
                pass

        # Fallback: text-based
        if name:
            return page.get_by_text(name, exact=True)

        return None
