"""模型序列化兼容性测试。"""

import ast
from pathlib import Path


def test_enhanced_mstgat_defines_from_config_for_model_reload():
    source = Path("model.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="model.py")

    target = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "EnhancedMSTGAT":
            target = node
            break

    assert target is not None, "缺少 EnhancedMSTGAT 类"
    method_names = {
        child.name for child in target.body if isinstance(child, ast.FunctionDef)
    }
    assert "from_config" in method_names, "EnhancedMSTGAT 需要 from_config 以支持 load_model()"
