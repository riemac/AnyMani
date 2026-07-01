# 事实记录

这里记录一些关于 LeapHand 的事实和数据，以供参考。

## Leaphand 关节索引和名称

1. LeapHand 机器人的刚体连杆组成部分：

- 手掌: palm_lower
- 食指: mcp_joint -> pip -> dip -> fingertip -> index_tip_head 
- 拇指: thumb_temp_base -> thumb_pip -> thumb_dip -> thumb_fingertip -> thumb_tip_head
- 中指: mcp_joint_2 -> pip_2 -> dip_2 -> fingertip_2 -> middle_tip_head
- 无名指: mcp_joint_3 -> pip_3 -> dip_3 -> fingertip_3 -> ring_tip_head
  
2. 关节索引：

- joints = [a_1, a_12, a_5, a_9, a_0, a_13, a_4, a_8, a_2, a_14, a_6, a_10, a_3, a_15, a_7, a_11]
- index finger: a_0~a_3
- middle finger: a_4~a_7
- little finger: a_8~a_11
- thumb: a_12~a_15

3. 关节限位：

## Official LEAP USD 与 URDF 的同名关节对应关系

调查对象：

- USD: `/home/hac/isaac/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_hand_v1_right/leap_hand_right.usd`
- URDF: `/home/hac/isaac/AnyMani/source/anymani/assets/hands/leap_hand/leap_hand_right.urdf`

结论：USD 与 URDF 中的 `a_0` 到 `a_15` 是逐名对应的同一组机械转动关节。
即 URDF 的 `a_i` 对应 USD 的 `a_i`，不是仅仅名字相同。验证依据是直接读取
USD `PhysicsRevoluteJoint` 的 `body0/body1/axis/limit` 与 URDF `joint` 的
`parent/child/axis/limit`：16 个同名关节均唯一存在，且 child link、关节限位和
所属手指链一致。

唯一需要注意的差异是 palm 侧 parent 的表示：URDF 中有 fixed joint
`base_joint: base -> palm_lower`，因此根部关节在 URDF 中写作
`palm_lower -> ...`，在 USD 中写作 `base -> ...`。这不是关节语义差异，
而是 fixed base/palm 表达被 USD 资产折叠后的命名差异。

逐项对应表：

| joint | URDF parent -> child | USD body0 -> body1 |
|---|---|---|
| `a_0` | `mcp_joint -> pip` | `mcp_joint -> pip` |
| `a_1` | `palm_lower -> mcp_joint` | `base -> mcp_joint` |
| `a_2` | `pip -> dip` | `pip -> dip` |
| `a_3` | `dip -> fingertip` | `dip -> fingertip` |
| `a_4` | `mcp_joint_2 -> pip_2` | `mcp_joint_2 -> pip_2` |
| `a_5` | `palm_lower -> mcp_joint_2` | `base -> mcp_joint_2` |
| `a_6` | `pip_2 -> dip_2` | `pip_2 -> dip_2` |
| `a_7` | `dip_2 -> fingertip_2` | `dip_2 -> fingertip_2` |
| `a_8` | `mcp_joint_3 -> pip_3` | `mcp_joint_3 -> pip_3` |
| `a_9` | `palm_lower -> mcp_joint_3` | `base -> mcp_joint_3` |
| `a_10` | `pip_3 -> dip_3` | `pip_3 -> dip_3` |
| `a_11` | `dip_3 -> fingertip_3` | `dip_3 -> fingertip_3` |
| `a_12` | `palm_lower -> thumb_temp_base` | `base -> thumb_temp_base` |
| `a_13` | `thumb_temp_base -> thumb_pip` | `thumb_temp_base -> thumb_pip` |
| `a_14` | `thumb_pip -> thumb_dip` | `thumb_pip -> thumb_dip` |
| `a_15` | `thumb_dip -> thumb_fingertip` | `thumb_dip -> thumb_fingertip` |

这条事实只证明 joint identity 与 kinematic-chain 语义对应；不自动保证两种资产在
collision、inertia、root frame、材质或 importer 运行时行为上完全一致。
