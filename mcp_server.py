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
# Description:  MCP server exposing the CYTools tools to MCP clients (e.g.
#               Claude Code). Registers the same MODEL_TOOLS the in-house agent
#               uses, so FastMCP derives identical schemas.
#
# Register (stdio), from the repo root:
#     claude mcp add cytools -- \
#         /path/to/envs/cytools-agent/bin/python3 mcp_server.py
# or add to .mcp.json:
#     {"mcpServers": {"cytools": {"type": "stdio",
#       "command": "/path/to/envs/cytools-agent/bin/python3",
#       "args": ["/abs/path/to/mcp_server.py"]}}}
# -----------------------------------------------------------------------------

# external imports
import contextlib
import functools
import sys

from mcp.server.fastmcp import FastMCP

# local imports
from cytools_agent.tools import MODEL_TOOLS

mcp = FastMCP("cytools")


def _quiet(fn):
    """Run the tool with stdout redirected to stderr. stdio MCP owns the real
    stdout (it is the protocol channel), so any stray library print must not
    leak there. functools.wraps keeps the signature/docstring/annotations, so
    FastMCP still derives the correct schema."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


for _fn in MODEL_TOOLS:
    mcp.add_tool(_quiet(_fn))


if __name__ == "__main__":
    mcp.run()   # stdio transport
