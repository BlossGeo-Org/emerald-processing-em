from . import pipeline
import logging
import typing
import libaarhusxyz
import libaarhusxyz.export.msgpack
import numpy as np
import pydantic

logger = logging.getLogger(__name__)

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

    _remap_apply_idx_by_fid(processing.xyz, diffxyz)
    processing.xyz = processing.xyz.apply_diff(diffxyz)


def _remap_apply_idx_by_fid(target_xyz, diffxyz):
    """Remap apply_idx using fid values so diffs survive sort-order changes.

    When a dataset is reimported and libaarhusxyz.normalize() produces a
    different global sort order (e.g. NaN dates becoming valid dates),
    positional apply_idx values point to wrong soundings.  If the diff
    includes a ``fid`` column (added by the frontend since Sprint 08),
    this function resolves each fid to its current position in the target
    dataset and overwrites apply_idx accordingly.

    No-op for old diffs that lack fid (backwards compatible).  Falls back
    to the original apply_idx if any fid cannot be found in the target.
    """
    if "fid" not in diffxyz.flightlines.columns:
        return

    diff_fids = diffxyz.flightlines["fid"].values
    target_fids = target_xyz.flightlines["fid"]

    # Build fid → positional-index lookup from the current dataset
    fid_to_pos = {}
    for pos, fid_val in enumerate(target_fids):
        if fid_val not in fid_to_pos:
            fid_to_pos[fid_val] = pos

    # Remap each diff fid to its current position in the target
    remapped = []
    for fid_val in diff_fids:
        pos = fid_to_pos.get(fid_val)
        if pos is None:
            logger.warning(
                "fid %s from diff not found in target dataset; "
                "falling back to positional apply_idx",
                fid_val,
            )
            return  # Abort remapping — use original apply_idx as-is
        remapped.append(pos)

    diffxyz.flightlines["apply_idx"] = np.array(remapped, dtype=np.int64)
    # Remove fid so df_apply() doesn't overwrite target fid values
    diffxyz.flightlines.drop("fid", axis=1, inplace=True)

