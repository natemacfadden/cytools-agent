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
# Description:  Generic file IO for agents. read_file peeks at a file's
#               contents, truncating large files to protect the context window.
# -----------------------------------------------------------------------------

# local imports
from cytools_agent.tools.history import logged

_MAX_CHARS = 4000  # cap returned text to protect the context window

# model-facing
# ------------
@logged
def read_file(path: str) -> str:
    """
    Read and return a text file's contents.

    Large files are truncated to the first few KB so they don't flood the
    context; the return notes when this happens.

    Parameters
    ----------
    path : str
        The file to read.

    Returns
    -------
    str
        The file contents, truncated if very long.
    """
    with open(path) as f:
        text = f.read()
    if len(text) > _MAX_CHARS:
        return text[:_MAX_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text
