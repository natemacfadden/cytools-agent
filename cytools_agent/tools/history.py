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
# Description:  Schema stub for the save_history tool. The actual implementation
#               lives in Agent.save_script; the harness wires them together so
#               the model can call save_history(path) to export the session.
# -----------------------------------------------------------------------------


def save_history(path: str) -> dict:
    """
    Write the session as a standalone, runnable Python script.

    The script imports the tools, replays every tool call the agent made
    (wrapped in print so results are visible), and includes the agent's text
    as comments so the reasoning is preserved.

    Only call this when the user has asked for it. ASK THE USER for the file
    path -- do NOT invent one. Writing to disk is irreversible.

    Parameters
    ----------
    path : str
        Destination file path for the script.

    Returns
    -------
    dict
        {"path": ..., "n_calls": ...}
    """
    raise RuntimeError("save_history must be wired to an Agent instance")
