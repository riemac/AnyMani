"""资产 schema 的兼容入口。

历史上，这个子项目曾经把大部分 schema 类都放在单独的
``asset_base.py`` 里。拆分成 ``asset_schema_core.py`` 和
``asset_schema_embodiment.py`` 之后，如果没有这个 façade，很多旧实验、
notebook 或一次性脚本里的导入路径会立刻失效。

因此这个文件的职责很明确：

- 保留旧的导入路径；
- 让调用者逐步迁移到新的 schema 分层；
- 不在这里承载新的算法逻辑。

当前的实际分工是：

- ``asset_schema_core.py``：位姿、材质、几何、惯量和底层规范化辅助。
- ``asset_schema_embodiment.py``：joint / finger / palm / hand 这些 embodiment 结构。

也就是说，`asset_base.py` 现在应该被理解成“兼容入口”，而不是
新的逻辑层。
"""

from .asset_schema_core import *  # noqa: F401,F403
from .asset_schema_embodiment import *  # noqa: F401,F403
