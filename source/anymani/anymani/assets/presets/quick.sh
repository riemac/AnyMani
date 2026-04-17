# 用来记录命令行，便于快速使用

# 手构建快速验证 allegro
python AnyMani/source/anymani/anymani/assets/presets/preview/preview_hand_preset.py \
    --hand-preset single_palm_allegro \
    --output-dir AnyMani/source/anymani/anymani/assets/generated/quick_preview

# 手指构建快速验证 allegro thumb
python AnyMani/source/anymani/anymani/assets/presets/preview/preview_finger_preset.py \
    --preset allegro_thumb_v1 \
    --output-dir AnyMani/source/anymani/anymani/assets/generated/quick_preview

# 手构建快速验证 leap
python AnyMani/source/anymani/anymani/assets/presets/preview/preview_hand_preset.py \
    --hand-preset single_palm_leap \
    --output-dir AnyMani/source/anymani/anymani/assets/generated/quick_preview

# 手指构建快速验证 leap thumb
python AnyMani/source/anymani/anymani/assets/presets/preview/preview_finger_preset.py \
    --preset leap_thumb_v1 \
    --output-dir AnyMani/source/anymani/anymani/assets/generated/quick_preview

# 手掌快速验证 allegro extracted compound palm 
python AnyMani/source/anymani/anymani/assets/presets/preview/preview_palm_preset.py \
    --preset com_allegro \
    --output-dir AnyMani/source/anymani/anymani/assets/generated/quick_preview

# 手快速验证 compound palm allegro hand