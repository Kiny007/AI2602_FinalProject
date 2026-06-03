"""YAML 配置加载工具。

训练脚本会先读取 YAML 配置文件作为默认超参数，再允许命令行参数覆盖，
这样既保留了配置文件的直观性，也方便临时修改单个选项。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _parse_scalar(value: str) -> Any:
    """解析当前项目配置里常见的标量类型。"""

    text = value.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "Null", "none", "None"}:
        return None
    if text in {'""', "''"}:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _simple_yaml_load(content: str) -> dict[str, Any]:
    """解析当前项目使用的平铺 YAML 键值对。"""

    data: dict[str, Any] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"无法解析 YAML 行: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"配置项键名不能为空: {raw_line}")
        data[key] = _parse_scalar(value)
    return data


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并返回字典。"""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    content = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(content)
    else:
        data = _simple_yaml_load(content)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是键值字典: {path}")
    return data


def apply_config_defaults(parser, config: dict[str, Any]) -> None:
    """把 YAML 中的值写回 argparse 默认值。"""

    valid_dests = {action.dest for action in parser._actions}
    unknown_keys = sorted(key for key in config if key not in valid_dests)
    if unknown_keys:
        raise ValueError(f"配置文件中存在未知字段: {', '.join(unknown_keys)}")
    parser.set_defaults(**config)
