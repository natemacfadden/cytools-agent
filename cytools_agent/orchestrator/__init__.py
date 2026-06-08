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
# Description:  Two-agent orchestration, split by concern:
#                 project_manager.py -- plan + work down the list + summarize
#                 engineer.py        -- the act-protocol task implementer
#                 evidence.py        -- read outputs, guarantee their accuracy
#               This package re-exports the public surface.
# -----------------------------------------------------------------------------

from cytools_agent.orchestrator.evidence import (EVIDENCE_PATH, LOG_DIR,
                                                 SESSION_PATH, export_script,
                                                 read_evidence, read_session,
                                                 render_evidence)
from cytools_agent.orchestrator.project_manager import (ProjectManager,
                                                        run_session)

__all__ = ["run_session", "ProjectManager", "EVIDENCE_PATH", "SESSION_PATH",
           "LOG_DIR", "export_script", "read_evidence", "read_session",
           "render_evidence"]
