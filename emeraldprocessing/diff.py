from . import pipeline
import typing
import copy
import libaarhusxyz
import libaarhusxyz.export.msgpack
import pydantic
import slugify

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
        
    diffxyz.normalize_naming(naming_standard="alc")

    processing.xyz = processing.xyz.apply_diff(diffxyz)


def save_intermediate_and_apply_diff(processing: pipeline.ProcessingData,
                                     name: str,
                                     diff: typing.Optional[ManualEditUrl] = None):
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

