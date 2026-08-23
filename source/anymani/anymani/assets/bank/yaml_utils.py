r"""资产 bank 使用的高速安全 YAML 解析入口。

资产 sidecar 规模较大且每只手都需要读取一次。PyYAML 的 ``safe_load`` 默认绑定
纯 Python ``SafeLoader``；当环境安装 libyaml 时，``CSafeLoader`` 对同一 YAML
语义提供相同的安全标签集合，但解析速度显著更高。这里集中选择 loader，保证
dataset manifest、generation summary 与 hand sidecar 遵循同一解析策略。
"""

from __future__ import annotations

from typing import Any

import yaml


def safe_load(data: bytes | str) -> Any:
    r"""使用 libyaml C loader 解析安全 YAML，并在不可用时回退纯 Python loader。

    ``yaml.load`` 的显式 Loader 参数确保不会意外使用任意对象构造器；
    ``CSafeLoader`` 与 ``SafeLoader`` 都只允许 YAML safe-load 合同。
    """

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(data, Loader=loader)


__all__ = ["safe_load"]
