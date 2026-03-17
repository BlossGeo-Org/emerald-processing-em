from . import pipeline
import typing
import libaarhusxyz
import libaarhusxyz.export.msgpack
import pydantic

ManualEditUrl = typing.Annotated[
    typing.Any,
    {"json_schema": {
        "x-reference": "manual-edit",
        "anyOf": [
            {"x-url-media-type": "application/x-geophysics-xyz-model"},
            {"type": "object",
             "additionalProperties": False,
             "required": ["url"],
             "properties": {
                 "url": {"x-url-media-type": "application/x-geophysics-xyz-model"},
                 "title": {"type": "string"},
                 "id": {"type": "integer"}
             }}
        ]
    }}]

def apply_diff(processing : pipeline.ProcessingData, 
               diff: ManualEditUrl):

    """
    Apply a manual culling to your dataset.
    
    Parameters
    ----------
    diff : 
        Manual culling to apply. To create manual culling, save culling in plot workspace first and it will appear here.
    """

    if isinstance(diff, dict): diff = diff["url"]

    if diff.endswith(".xyz") or diff.endswith(".xyzd"):
        diffxyz = libaarhusxyz.XYZ(diff, normalize=False)
    elif diff.endswith(".msgpack"):
        diffxyz = libaarhusxyz.export.msgpack.load(diff)
        
    # Only normalize full XYZ files (.xyz/.xyzd), never diffs from msgpack.
    # Diffs have sparse model_dicts with raw values (not DataFrames) that
    # crash normalize_naming, and their column names already match the source.
    if diff.endswith(".xyz") or diff.endswith(".xyzd"):
        if hasattr(diffxyz, 'model_dict') and 'model_info' in diffxyz.model_dict:
            diffxyz.normalize_naming(naming_standard="alc")

    # Ensure diff_dummy is set — frontend manual edit diffs always use -1 as
    # the sentinel for "no change". If model_info is missing or incomplete
    # (e.g. from a load→re-save cycle), default it so apply_diff skips sentinels.
    # Note: model_info is a property, so set via model_dict directly.
    if not diffxyz.model_info.get("diff_dummy"):
        mi = diffxyz.model_dict.get("model_info", {})
        mi["diff_dummy"] = -1
        diffxyz.model_dict["model_info"] = mi

    processing.xyz = processing.xyz.apply_diff(diffxyz)

