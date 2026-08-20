"""
Tests for the shipped manual-edit entry point.

'Save intermediate results and apply manual edits' is the only diff step
registered in setup.py, so it is the path that has to be correct. It used to
hold a verbatim copy of apply_diff's body, which meant four upstream
robustness fixes landed in apply_diff and never reached it (#78).

These tests drive save_intermediate_and_apply_diff, not apply_diff, so a
future re-divergence fails here rather than in production.

The libaarhusxyz loaders are stubbed: the behavior under test is the guard
logic around them, not the file parsers.
"""
import copy
import pytest

from emeraldprocessing import diff as diff_module


class FakeDiffXYZ:
    """Stands in for a loaded diff.

    model_info mirrors libaarhusxyz in returning a *copy*, so a regression
    that writes to it instead of to model_dict is caught rather than passing.
    """

    def __init__(self, model_dict=None):
        self.model_dict = {} if model_dict is None else model_dict
        self.normalize_calls = []

    @property
    def model_info(self):
        return dict(self.model_dict.get('model_info', {}))

    def normalize_naming(self, naming_standard=None):
        self.normalize_calls.append(naming_standard)


class FakeXYZ:
    def __init__(self, lines=('line_a', 'line_b')):
        self._lines = lines
        self.applied = []

    def split_by_line(self):
        return {name: FakeXYZ(lines=()) for name in self._lines}

    def apply_diff(self, diffxyz):
        self.applied.append(diffxyz)
        return self


class FakeProcessing:
    def __init__(self, outdir):
        self.outdir = str(outdir)
        self.xyz = FakeXYZ()
        self.orig_xyz_by_line = {name: object() for name in self.xyz._lines}
        self.dumps = []

    def dump(self, **kwargs):
        # copy.copy() in the per-line loop shares this list, so it records
        # both the whole-dataset dump and every per-line dump.
        self.dumps.append(kwargs)


@pytest.fixture
def stub_loaders(monkeypatch):
    """Route both loaders to a FakeDiffXYZ the test controls."""
    holder = {}

    def _install(model_dict=None):
        fake = FakeDiffXYZ(model_dict)
        holder['fake'] = fake
        monkeypatch.setattr(diff_module.libaarhusxyz, 'XYZ',
                            lambda path, normalize=True: fake)
        monkeypatch.setattr(diff_module.libaarhusxyz.export.msgpack, 'load',
                            lambda path: fake)
        return fake

    _install.holder = holder
    return _install


def _run(tmp_path, stub_loaders, diff_name, model_dict=None):
    processing = FakeProcessing(tmp_path)
    fake = stub_loaders(model_dict)
    diff_module.save_intermediate_and_apply_diff(
        processing, name='step_01', diff=f'{tmp_path}/{diff_name}')
    return processing, fake


# --------------------------------------------------------------------------
# The four guards, exercised through the shipped entry point
# --------------------------------------------------------------------------

def test_xyz_diff_without_model_info_skips_normalize(tmp_path, stub_loaders):
    """Frontend manual-edit diffs carry minimal model_info; normalizing crashes."""
    processing, fake = _run(tmp_path, stub_loaders, 'edit.xyz', model_dict={})

    assert fake.normalize_calls == []
    assert processing.xyz.applied == [fake]


def test_xyz_diff_without_layer_data_still_normalizes(tmp_path, stub_loaders):
    """The layer_data guard was deliberately superseded by the extension gate.

    2fc38c7 required 'layer_data' in model_dict; eccae61 replaced that with a
    file-extension check, because some diffs do have layer_data but hold dict
    values rather than DataFrames. For a .xyz with model_info present the
    correct current behavior is to normalize.
    """
    processing, fake = _run(tmp_path, stub_loaders, 'edit.xyz',
                            model_dict={'model_info': {'diff_dummy': -1}})

    assert fake.normalize_calls == ['alc']


def test_msgpack_diff_never_normalizes(tmp_path, stub_loaders):
    """Msgpack diffs hold raw values that crash normalize_naming."""
    processing, fake = _run(tmp_path, stub_loaders, 'edit.msgpack',
                            model_dict={'model_info': {'diff_dummy': -1},
                                        'layer_data': {}})

    assert fake.normalize_calls == []
    assert processing.xyz.applied == [fake]


def test_missing_diff_dummy_defaults_to_minus_one(tmp_path, stub_loaders):
    """Without this, sentinel values get written as real InUse data."""
    processing, fake = _run(tmp_path, stub_loaders, 'edit.msgpack', model_dict={})

    assert fake.model_dict['model_info']['diff_dummy'] == -1


def test_existing_diff_dummy_is_preserved(tmp_path, stub_loaders):
    processing, fake = _run(tmp_path, stub_loaders, 'edit.msgpack',
                            model_dict={'model_info': {'diff_dummy': -999}})

    assert fake.model_dict['model_info']['diff_dummy'] == -999


def test_diff_accepts_dict_with_url_key(tmp_path, stub_loaders):
    """The step runner may hand the parameter through as {'url': ...}."""
    processing = FakeProcessing(tmp_path)
    fake = stub_loaders({})
    diff_module.save_intermediate_and_apply_diff(
        processing, name='step_01', diff={'url': f'{tmp_path}/edit.msgpack'})

    assert processing.xyz.applied == [fake]


# --------------------------------------------------------------------------
# Save-intermediate behavior must be unaffected by the consolidation
# --------------------------------------------------------------------------

def test_saves_intermediate_before_applying_diff(tmp_path, stub_loaders):
    processing, fake = _run(tmp_path, stub_loaders, 'edit.msgpack', model_dict={})

    # One whole-dataset dump plus one per flight line.
    assert len(processing.dumps) == 3

    whole = processing.dumps[0]
    assert whole['xyzfile'] == f'{tmp_path}/step_01.xyz'
    assert whole['gexfile'] == f'{tmp_path}/step_01.gex'
    assert whole['msgpackfile'] == f'{tmp_path}/step_01.msgpack'
    assert whole['summaryfile'] == f'{tmp_path}/step_01.summary.yml'

    per_line = {d['xyzfile'] for d in processing.dumps[1:]}
    assert per_line == {f'{tmp_path}/step_01.line_a.xyz',
                        f'{tmp_path}/step_01.line_b.xyz'}


def test_saves_intermediate_when_no_diff_given(tmp_path):
    """diff is optional; the save half must still run on its own."""
    processing = FakeProcessing(tmp_path)
    diff_module.save_intermediate_and_apply_diff(processing, name='step_01')

    assert len(processing.dumps) == 3
    assert processing.xyz.applied == []


def test_requires_an_outdir(tmp_path):
    processing = FakeProcessing(tmp_path)
    processing.outdir = None

    with pytest.raises(AssertionError):
        diff_module.save_intermediate_and_apply_diff(processing, name='step_01')


# --------------------------------------------------------------------------
# The consolidation itself
# --------------------------------------------------------------------------

def test_delegates_to_apply_diff(tmp_path, monkeypatch):
    """Locks #78: one implementation, and the shipped function calls it.

    If someone re-inlines the load-and-normalize block, this fails.
    """
    calls = []
    monkeypatch.setattr(diff_module, 'apply_diff',
                        lambda processing, diff: calls.append((processing, diff)))

    processing = FakeProcessing(tmp_path)
    diff_module.save_intermediate_and_apply_diff(
        processing, name='step_01', diff=f'{tmp_path}/edit.msgpack')

    assert len(calls) == 1
    assert calls[0][1] == f'{tmp_path}/edit.msgpack'


def test_apply_diff_is_not_called_when_diff_is_none(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(diff_module, 'apply_diff',
                        lambda processing, diff: calls.append(diff))

    processing = FakeProcessing(tmp_path)
    diff_module.save_intermediate_and_apply_diff(processing, name='step_01')

    assert calls == []
