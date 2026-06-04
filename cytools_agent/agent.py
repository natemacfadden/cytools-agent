# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Generic agent loop over an OpenAI-compatible API (e.g. Ollama).
#               Recovers tool calls when a local model emits malformed or
#               unfenced JSON instead of structured tool_calls.
# -----------------------------------------------------------------------------

# external imports
import ast
import json
import re
import time

# tool-call parsing
# -----------------
def _decode_at(text, i):
    """JSON object at index i (allows trailing junk), else a Python literal."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, i)
        return obj
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text[i:])
        except (ValueError, SyntaxError):
            return None


def extract_tool_call(content, known_tools):
    """
    Recover the first tool call from a model's text content.

    Parameters
    ----------
    content : str
        The assistant message content to scan.
    known_tools : set of str
        The names of the available tools.

    Returns
    -------
    dict or None
        {"name", "arguments"} for a usable call, {"error": reason} if the text
        looked like a call but was malformed, or None if there is no call.
    """
    if not content:
        return None
    text = content.strip().replace("\x00", "")
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    reason = None
    for i in (j for j, ch in enumerate(text) if ch == "{"):
        obj = _decode_at(text, i)
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        if obj["name"] not in known_tools:
            reason = f"unknown tool {obj['name']!r}"
            continue
        args = obj.get("arguments")
        if isinstance(args, str):               # some models double-encode
            args = _decode_at(args, 0)
        if not isinstance(args, dict):
            reason = f"tool {obj['name']!r} was given non-dict arguments"
            continue
        return {"name": obj["name"], "arguments": args}
    return {"error": reason} if reason else None


def _strip_template_tags(text):
    """
    Strip chat-template boundary tags (<tool_call>/<tool_response>) that the
    model wrote out as literal text characters instead of as real special
    tokens.

    These tags are turn/tool "barriers" that normally live in the tokenizer and
    are consumed by the server. When a small model emits them by imitation they
    are just plain text, so they leak into its final answer -- we remove that
    text here.
    """
    if not text:
        return text
    text = re.sub(r"<tool_(call|response)>.*?</tool_\1>", "", text,
                  flags=re.DOTALL)
    text = re.sub(r"</?tool_(call|response)>", "", text)
    return text.strip()


# agent loop (the harness; not exposed to the model)
# ---------------------------------------------------
class Agent:
    """
    A stateful conversation over a tool-calling model.

    .chat(text) appends the message, runs the tool loop until the model
    produces a final answer, and keeps the history so later calls remember
    earlier turns.

    Parameters
    ----------
    client : openai.OpenAI
        An OpenAI-compatible client (e.g. pointed at a local Ollama server).
    model : str
        The model id to call.
    system_prompt : str
        The system message that opens the conversation.
    tools : list of dict
        OpenAI tool schemas.
    tool_impls : dict
        Mapping from tool name to the Python callable.
    max_steps : int, optional
        Maximum tool-loop iterations per .chat call.
    verbosity : int, optional
        0 silent; 1 prints a tag per step; >=2 also prints the payload.
    """
    def __init__(self, client, model, system_prompt, tools, tool_impls,
                 max_steps=20, verbosity=0):
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_impls = tool_impls
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.messages = [{"role": "system", "content": system_prompt}]
        # wall-clock split between model (LLM) calls and tool computations
        self.timing = {"model": 0.0, "tools": 0.0}
        self.tool_secs = {}  # per-tool cumulative seconds

    def chat(self, user_message):
        """Run one turn through the tool loop and return the final answer."""
        self.messages.append({"role": "user", "content": user_message})
        for step in range(self.max_steps):
            _t = time.monotonic()
            msg = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools
            ).choices[0].message
            self.timing["model"] += time.monotonic() - _t
            self.messages.append(msg)

            # parse the message type
            if msg.tool_calls:
                if self.verbosity == 1:
                    print("Agent: Tool call")
                elif self.verbosity >= 2:
                    print(f"Agent: Tool call `{msg.tool_calls}`")

                calls = [(c.id, c.function.name,
                          json.loads(c.function.arguments))
                         for c in msg.tool_calls]
            elif (fb := extract_tool_call(msg.content, set(self.tool_impls))):
                # malformed call -> tell the model what was wrong and retry
                if "error" in fb:
                    if self.verbosity >= 1:
                        print(f"Agent: malformed tool call ({fb['error']})")
                    self.messages.append({"role": "user", "content":
                        f"That wasn't a valid tool call: {fb['error']}. "
                        'Reply with one JSON object: {"name": ..., '
                        '"arguments": {...}}.'})
                    continue

                if self.verbosity == 1:
                    print("Agent: (recovered) tool call")
                elif self.verbosity >= 2:
                    print(f"Agent: (recovered) tool call `{fb}`")

                # the model wrote the tool call as text, not a real tool_call,
                # so rebuild it as one -- else the result below has nothing to
                # attach to and it just re-issues the same call
                call_id = f"fallback_{step}"
                self.messages[-1] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": fb["name"],
                                     "arguments": json.dumps(fb["arguments"])},
                    }],
                }
                calls = [(call_id, fb["name"], fb["arguments"])]
            else:
                if self.verbosity == 1:
                    print("Agent: Text message")
                elif self.verbosity >= 2:
                    print(f"Agent: Text message `{msg}`")

                return _strip_template_tags(msg.content)

            # run the tools
            for call_id, name, args in calls:
                _t = time.monotonic()
                try:
                    result = self.tool_impls[name](**args)
                except Exception as e:
                    result = f"ERROR: {e}"
                dt = time.monotonic() - _t
                self.timing["tools"] += dt
                self.tool_secs[name] = self.tool_secs.get(name, 0.0) + dt
                self.messages.append({"role": "tool", "tool_call_id": call_id,
                                      "content": str(result)})
        return "(max_steps exceeded)"
