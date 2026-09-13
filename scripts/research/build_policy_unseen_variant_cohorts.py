r"""已发布的未见变体命令入口，实现由资产领域的选择与集合工具提供。"""

from anymani.assets.scripts.unseen_variants import main, select_policy_unseen_variant_splits

__all__ = ["main", "select_policy_unseen_variant_splits"]


if __name__ == "__main__":
    main()
