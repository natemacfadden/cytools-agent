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
import inspect
import json
import re
import time

# human-read
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


# human-read
def extract_tool_call(content, known_tools):
    """Recover the first tool call from a model's text content.
    Returns {"name", "arguments"}, {"error": reason}, or None."""
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


# human-read
def _strip_template_tags(text):
    """Remove literal <tool_call>/<tool_response> tags that small models
    sometimes emit as plain text instead of as real special tokens."""
    if not text:
        return text
    text = re.sub(r"<tool_(call|response)>.*?</tool_\1>", "", text,
                  flags=re.DOTALL)
    text = re.sub(r"</?tool_(call|response)>", "", text)
    return text.strip()


# human-read
_CTX_PROBED = set()   # (base_url, model) pairs already probed this process


def _warn_if_truncating(client, model):
    """One-time canary against silent context truncation. The OpenAI-compat
    endpoint can't set num_ctx, so a server at Ollama's default (often 4096)
    front-truncates long sessions, dropping the system prompt first. Probe
    with one long prompt and warn if the server capped the tokens seen."""
    key = (str(getattr(client, "base_url", "")), model)
    if key in _CTX_PROBED:
        return
    _CTX_PROBED.add(key)
    try:
        filler = " ".join(f"w{i}" for i in range(5500))   # ~6-7k tokens
        resp = client.chat.completions.create(
            model=model, max_tokens=1,
            messages=[{"role": "user", "content": filler + " Reply: OK"}])
        seen = getattr(resp.usage, "prompt_tokens", None)
        if seen is not None and seen < 5000:
            print(f"WARNING: the Ollama server is truncating prompts to "
                  f"~{seen} tokens (system prompt is lost first; the agent "
                  f"will misbehave on long sessions). Configure the server "
                  f"with OLLAMA_CONTEXT_LENGTH=16384 -- ./setup.sh does this "
                  f"automatically.")
    except Exception:
        pass   # the probe must never break a session


def _fmt_call(name, args):
    """A tool call as one short, readable line (long arg values truncated)."""
    parts = []
    for k, v in args.items():
        s = repr(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return f"{name}({', '.join(parts)})"


# human-read (class + methods, except save_history which is model-read)
class Agent:
    """
    Stateful conversation over a tool-calling model. .chat(text) runs the
    tool loop and returns the final answer; history accumulates across calls.
    verbosity: 0 silent, 1 tags, >=2 full payloads. message_hook, if given, is
    called on each user message and its (string) return is appended to it,
    used to auto-inject glossary context so the model needn't ask for it.
    """
    def __init__(self, client, model, system_prompt, tools, tool_impls,
                 max_steps=20, verbosity=0, message_hook=None):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.message_hook = message_hook
        self.messages = [{"role": "system", "content": system_prompt}]
        self.timing = {"model": 0.0, "tools": 0.0}
        self.tool_secs = {}
        # auto-register save_history so the model can call it
        self.tool_impls = dict(tool_impls)
        self.tool_impls["save_history"] = self.save_history
        self.tools = list(tools) + [self._save_history_schema()]

    def chat(self, user_message):
        """Run one turn through the tool loop and return the final answer."""
        _warn_if_truncating(self.client, self.model)
        if self.message_hook:
            extra = self.message_hook(user_message)
            if extra:
                user_message = f"{user_message}\n\n{extra}"
        self.messages.append({"role": "user", "content": user_message})
        self._empty_nudges = 0
        for step in range(self.max_steps):
            _t = time.monotonic()
            msg = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools
            ).choices[0].message
            self.timing["model"] += time.monotonic() - _t
            # normalize to a plain dict so history holds one type; the SDK
            # object's shape varies across versions
            self.messages.append(
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump") else msg)

            if msg.tool_calls:
                # a structured call can still carry broken JSON arguments (or
                # a non-object); represent that as an error the dispatch loop
                # feeds back as the tool result, instead of crashing the turn
                calls = []
                for c in msg.tool_calls:
                    try:
                        args = json.loads(c.function.arguments)
                        if not isinstance(args, dict):
                            raise ValueError(
                                f"arguments must be a JSON object, got "
                                f"{type(args).__name__}")
                    except (ValueError, TypeError) as e:
                        args = e
                    calls.append((c.id, c.function.name, args))
                if self.verbosity >= 1:
                    for _, name, args in calls:
                        print(f"  -> {_fmt_call(name, args)}"
                              if self.verbosity >= 2 and isinstance(args, dict)
                              else f"  -> {name}")
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

                if self.verbosity >= 1:
                    print(f"  -> (recovered) {_fmt_call(fb['name'], fb['arguments'])}"
                          if self.verbosity >= 2
                          else f"  -> (recovered) {fb['name']}")

                # the model wrote the call as text, not a real tool_call, so
                # rebuild it as one; else the result below can't attach to it
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
                final = _strip_template_tags(msg.content or "")
                if not final.strip():
                    # newer Ollama builds put qwen3's thinking in a separate
                    # `reasoning` field and may leave content empty; an empty
                    # final answer is never right, so nudge instead of
                    # returning it (bounded: two nudges per turn)
                    if self._empty_nudges < 2:
                        self._empty_nudges += 1
                        self.messages.append({"role": "user", "content":
                            "Your last message was empty. Continue: either "
                            "call a tool, or state the final answer with the "
                            "concrete numbers."})
                        continue
                if self.verbosity >= 1:
                    print()   # blank line: separate the trace from the answer
                return final

            for call_id, name, args in calls:
                _t = time.monotonic()
                if isinstance(args, Exception):
                    result = (f"ERROR: tool arguments were not valid JSON "
                              f"({args}). Re-send the call with arguments as "
                              f"one JSON object.")
                    args = {}
                else:
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

    def _save_history_schema(self) -> dict:
        doc = inspect.getdoc(self.save_history)
        return {
            "type": "function",
            "function": {
                "name": "save_history",
                "description": doc or "",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }

    # model-read (this docstring becomes the save_history tool schema)
    def save_history(self, path: str) -> dict:
        """
        Write the session as a standalone, runnable Python script.

        The script replays every tool call (wrapped in print so results are
        visible) and includes the agent's text as comments. Only call this
        when the user has asked for it. ASK THE USER for the path -- do NOT
        invent one.
        """
        header = [
            "# cytools-agent session script",
            "# run with:  python <this_file>.py",
            "import sys; sys.path.insert(0, '.')",
            "import warnings; warnings.filterwarnings('ignore')",
            "from cytools_agent.tools import (",
            "    fetch_polytopes, get_polytope_info, ks_stats,",
            "    get_heights, get_triangulation_info,",
            "    get_cy_info, get_cy_cones, run_python, cytools_help)",
            "",
        ]
        def _get(m, k):
            return m.get(k) if isinstance(m, dict) else getattr(m, k, None)

        body = []
        n_calls = 0
        for msg in self.messages:
            if _get(msg, "role") != "assistant":
                continue
            calls = _get(msg, "tool_calls")
            content = _get(msg, "content")
            if calls:
                for c in calls:
                    if isinstance(c, dict):
                        name = c["function"]["name"]
                        args = json.loads(c["function"]["arguments"])
                    else:
                        name = c.function.name
                        args = json.loads(c.function.arguments)
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    body.append(f"print({name}({arg_str}))")
                    n_calls += 1
            elif content:
                text = _strip_template_tags(content).strip()
                if text:
                    lines = text.splitlines()[:8]
                    for line in lines:
                        body.append(f"# {line}")
                    if len(text.splitlines()) > 8:
                        body.append("# ...")
                    body.append("")

        with open(path, "w") as f:
            f.write("\n".join(header + body) + "\n")
        return {"path": path, "n_calls": n_calls}
