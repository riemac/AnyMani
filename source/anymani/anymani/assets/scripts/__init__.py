r"""手部资产脚本入口层。

当前职责重新收敛为两类：

1. `assets/config/*.py`：声明式资产生成配置
2. `assets/scripts/generate.py`：统一 runner

本包仅保留 runner helper，供测试与统一 CLI 复用。
"""
