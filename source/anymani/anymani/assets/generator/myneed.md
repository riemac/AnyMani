AssetGenerator 便于用户使用，该层是用户真正易于和主要使用接口，主要用于：
- 生成多元化的资产：给定 “生成空间”，AssetGenerator 可以生成多元化的资产
- 微调：对特定 HandCfg的特定属性，如某个挂载点位置，某个连杆长度，可在线实时微调，直到用户满意

AssetGenerator 是主要，用户基本配置其参数，调用其相关 api 即可，不需要再复杂地调用其他组件

AssetGenerator 涵盖了整套流程，从生产空间、采样策略配置，到 Exporter,Validator 和 urdf。但也能精确控制到某种程度，如指定 post-mutate or pre-made，HandCfg -> URDF 一步到位还是先生成 HandCfg 等轻量产物

pre-made 主要复用 builder 组件，post-mutate 可能要另构建一套组件（类似builder），可能是tools，譬如
- joint delete，删除 finger 中某个joint,但会自动合理串联剩余的 joints。而且 joint delete 不是任意的删，而是和 finger_builder 中的 preset相关。有些关节需要保留，不同的关节数对应了不同的finger配置

AssetGenerator的需求复杂度、配置复杂度已经很难了，我们是否需要像 AnyMani/source/anymani/anymani/tasks/inhand/config/leaphand/agents/rl_games_ppo_cfg.yaml 一样是的配置文件？