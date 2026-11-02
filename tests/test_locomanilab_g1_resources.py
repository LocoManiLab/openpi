"""G1 recipe resource checks without importing a model runtime."""

import ast
from pathlib import Path


def test_g1_recipes_keep_local_base_checkpoint_placeholders():
    path = Path(__file__).resolve().parents[1] / "src/openpi/training/config.py"
    tree = ast.parse(path.read_text())
    names = {
        "pi05_g1_sonic", "pi05_bench_heft_openfridge", "pi05_bench_amo_closefridge",
        "pi05_bench_sonic_all31", "pi05_g1_sonic_toilet401", "pi05_g1_sonic_toilet404",
    }
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = {argument.arg: argument.value for argument in node.keywords}
        name = arguments.get("name")
        if not isinstance(name, ast.Constant) or name.value not in names:
            continue
        found.add(name.value)
        loader = arguments["weight_loader"]
        assert isinstance(loader, ast.Call)
        assert isinstance(loader.func, ast.Attribute)
        assert loader.func.attr == "PartialCheckpointWeightLoader"
        assert len(loader.args) == 1
        assert loader.args[0].value.startswith("/path/to/local/")
    assert found == names
