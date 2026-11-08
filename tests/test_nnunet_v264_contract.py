"""Optional exact-source contract check, without importing Torch or nnU-Net.

Use NNUNET_V264_SOURCE=/path/to/nnUNet for a cached 2.6.4 checkout. Otherwise
inspect an installed distribution only if metadata identifies exactly 2.6.4.
No dependencies are installed, no network is used, and no weights are loaded.
"""

import ast
import copy
from importlib import metadata, util
import inspect
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


def source_file():
    configured = os.environ.get("NNUNET_V264_SOURCE")
    if configured:
        root = Path(configured)
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        if not re.search(r'^version\s*=\s*[\"\']2\.6\.4[\"\']', project, re.MULTILINE):
            raise AssertionError("NNUNET_V264_SOURCE must identify version 2.6.4")
        return root / "nnunetv2/inference/predict_from_raw_data.py"
    try:
        version = metadata.version("nnunetv2")
    except metadata.PackageNotFoundError:
        raise unittest.SkipTest("nnunetv2 2.6.4 is not installed; set NNUNET_V264_SOURCE for an offline cached-source check")
    if version != "2.6.4":
        raise unittest.SkipTest("Installed nnunetv2 {} is not the exact 2.6.4 contract".format(version))
    package = util.find_spec("nnunetv2")
    return Path(next(iter(package.submodule_search_locations))) / "inference/predict_from_raw_data.py"


def strip_types(function):
    function = copy.deepcopy(function)
    function.returns = None
    function.decorator_list = []
    for argument in function.args.posonlyargs + function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    return function


class Nnunet264ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = source_file()
        cls.source = path.read_text(encoding="utf-8")
        predictor = next(node for node in ast.parse(cls.source).body
                         if isinstance(node, ast.ClassDef) and node.name == "nnUNetPredictor")
        cls.methods = {node.name: node for node in predictor.body if isinstance(node, ast.FunctionDef)}

    def test_adapter_calls_bind_to_actual_264_signatures(self):
        root = Path(__file__).resolve().parents[1]
        adapter = "\n".join((root / filename).read_text(encoding="utf-8")
                            for filename in ("benchmark_spine_inference.py", "benchmark_lesion_inference.py"))
        targets = {"nnUNetPredictor": "__init__",
                   "initialize_from_trained_model_folder": "initialize_from_trained_model_folder",
                   "predict_from_files_sequential": "predict_from_files_sequential"}
        checked = set()
        for call in ast.walk(ast.parse(adapter)):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "attr", getattr(call.func, "id", None))
            if name not in targets:
                continue
            method = strip_types(self.methods[targets[name]])
            method.body = [ast.Pass()]
            method.args.defaults = [ast.Constant(value=None) for _ in method.args.defaults]
            method.args.kw_defaults = [ast.Constant(value=None) if value is not None else None for value in method.args.kw_defaults]
            module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
            namespace = {}
            exec(compile(module, "<actual nnunetv2 2.6.4 signature>", "exec"), namespace)
            signature = inspect.signature(namespace[method.name])
            signature.bind(object(), *[object() for _ in call.args], **{keyword.arg: object() for keyword in call.keywords})
            checked.add(name)
        self.assertEqual(checked, set(targets))

    def test_actual_filename_mapping_accepts_explicit_lists_without_discovery(self):
        function = strip_types(self.methods["_manage_input_and_output_lists"])
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        def forbidden_discovery(*args, **kwargs):
            raise AssertionError("Explicit list input must not start filename-discovery workers")
        namespace = {"os": os, "join": os.path.join, "isfile": lambda filename: False,
                     "print": lambda *args, **kwargs: None,
                     "create_lists_from_splitted_dataset_folder": forbidden_discovery}
        exec(compile(module, "<actual nnunetv2 2.6.4 filename handling>", "exec"), namespace)
        inputs = [["/inputs/case_0001_0000.nii.gz"]]
        outputs = ["/outputs/case_0001"]
        result = namespace[function.name](SimpleNamespace(dataset_json={"file_ending": ".nii.gz"}),
                                          inputs, outputs, overwrite=False)
        self.assertEqual(result, (inputs, outputs, [None]))
        self.assertEqual(Path(result[1][0] + ".nii.gz").name, "case_0001.nii.gz")


if __name__ == "__main__":
    unittest.main()
