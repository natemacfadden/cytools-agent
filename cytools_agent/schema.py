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
# Description:  Build OpenAI-style tool schemas from a function's signature and
#               docstring, so tools need no hand-written schema.
# -----------------------------------------------------------------------------

# external imports
import inspect
import types
import typing

_JSON_TYPES = {
    int:   "integer",
    float: "number",
    str:   "string",
    bool:  "boolean",
    list:  "array",
    tuple: "array",
    dict:  "object",
}


def _json_type(annotation) -> str:
    """
    Map a type annotation to its JSON-schema type name. Unwraps `X | None` to
    X, and treats list/tuple as "array", dict as "object".
    """
    origin = typing.get_origin(annotation)

    if origin in (typing.Union, types.UnionType):
        annotation = next(
            (a for a in typing.get_args(annotation) if a is not type(None)), str
        )
        origin = typing.get_origin(annotation)

    # parameterized generics (e.g. list[float]) -> the bare container type
    if origin in (list, tuple):
        return "array"
    if origin is dict:
        return "object"
    return _JSON_TYPES.get(annotation, "string")


def function_to_schema(fn) -> dict:
    """
    Build an OpenAI-style tool schema from a function.

    Parameter types and which are required come from the signature + type
    hints; the description is the function's full docstring (so the model sees
    each parameter's meaning). Nothing needs to be written by hand.

    Parameters
    ----------
    fn : callable
        The tool function to describe.

    Returns
    -------
    dict
        An OpenAI tool schema: {"type": "function", "function": {...}}.
    """
    hints = typing.get_type_hints(fn)

    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        props[name] = {"type": _json_type(hints.get(name, str))}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }
