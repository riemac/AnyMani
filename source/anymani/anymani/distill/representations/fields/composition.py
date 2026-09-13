r"""逐 PALM/JOINT/TIP 归属体 field 的组合与联合诊断边界。

第一版保持 $G=N_E$ 个实体/表面归属体直接同索引，并监督逐归属体
$d_g$、$\rho_{\sigma,g}$、$\kappa_g$ 与 $g_{\sigma,g}$。当前不训练独立 whole-hand
union head，也不增加 union consistency 主损失。若为诊断需要联合场，只能在共同查询网格上
从逐归属体预测解析派生，并明确它不是新的监督目标。

预测端不能对所有 field 一律使用 sum 或 min：

- 若各 group SDF 都采用“体内负、体外正”，实体 union 的解析候选为$s_{union}(x)=\min_g s_g(x)$；

- 对 $o_g(x)\in[0,1]$ 的 soft occupancy，可比较 ``max`` 与$1-\prod_g(1-o_g)$，但后者带有概率式解释，不能未经验证宣称严格物理独立；

- surface density / KDE 的 sum 会在重叠 collision bodies 处重复计数，max 或 smooth-max
  又改变 surface-mass 语义；最终算子必须由 target definition 与 overlap stress test 决定；
- parametric Gaussian field 先按 semantic group 输出，再通过同一 field-level operator
  形成 union prediction，而不是先丢弃 group ownership。

早期 tip-only、union-only 与 decomposed+union 路线只保留为历史/后续消融语义，不能在 trainer
中作为与当前逐归属体合同同优先级的隐藏默认。
"""
