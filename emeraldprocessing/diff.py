from . import pipeline
import typing
import copy
import pydantic
import libaarhusxyz
import libaarhusxyz.export.msgpack
import slugify

ManualEditUrl = typing.Annotated[
    pydantic.AnyUrl,
    {"json_schema": {
        "x-url-media-type": "application/x-geophysics-xyz-model"
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


def save_intermediate_and_apply_diff(processing: pipeline.ProcessingData,
                                     name: str,
                                     diff: ManualEditUrl = None):
    """
    Save an intermediate state of your processing, then apply a manual culling.

    Parameters
    ----------
    name :
        Give the intermediate result a dataset name. This will appear in the plot editor under Dataset.
    diff :
        Manual culling to apply. To create manual culling, save culling in plot workspace first and it will appear here.
    """

    assert processing.outdir is not None

    processing.dump(
        xyzfile = '%s/%s.xyz' % (processing.outdir, name),
        gexfile = '%s/%s.gex' % (processing.outdir, name),
        msgpackfile = '%s/%s.msgpack' % (processing.outdir, name),
        diffmsgpackfile = '%s/%s.diff.msgpack' % (processing.outdir, name),
        summaryfile = '%s/%s.summary.yml' % (processing.outdir, name),
        geojsonfile = '%s/%s.geojson' % (processing.outdir, name))

    for fline, line_data in processing.xyz.split_by_line().items():
        sfline = slugify.slugify(str(fline), separator="_")
        fl_processing = copy.copy(processing)
        fl_processing.xyz = line_data
        fl_processing.orig_xyz = processing.orig_xyz_by_line[fline]
        fl_processing.dump(
            xyzfile = '%s/%s.%s.xyz' % (processing.outdir, name, sfline),
            gexfile = '%s/%s.%s.gex' % (processing.outdir, name, sfline),
            msgpackfile = '%s/%s.%s.msgpack' % (processing.outdir, name, sfline),
            diffmsgpackfile = '%s/%s.%s.diff.msgpack' % (processing.outdir, name, sfline),
            summaryfile = '%s/%s.%s.summary.yml' % (processing.outdir, name, sfline),
            geojsonfile = '%s/%s.%s.geojson' % (processing.outdir, name, sfline))

    if diff is not None:
        if isinstance(diff, dict): diff = diff["url"]

        if diff.endswith(".xyz") or diff.endswith(".xyzd"):
            diffxyz = libaarhusxyz.XYZ(diff, normalize=False)
        elif diff.endswith(".msgpack"):
            diffxyz = libaarhusxyz.export.msgpack.load(diff)

        diffxyz.normalize_naming(naming_standard="alc")

        processing.xyz = processing.xyz.apply_diff(diffxyz)

