from importlib import import_module


def test_assets_modules_import_cleanly():
    modules = [
        "source.anymani.anymani.assets.builder.joint_builders_primitive",
        "source.anymani.anymani.assets.builder.palm_builders",
        "source.anymani.anymani.assets.builder.finger_buiders",
        "source.anymani.anymani.assets.builder.hand_builders",
        "source.anymani.anymani.assets.validator.joint_rules",
        "source.anymani.anymani.assets.validator.finger_rules",
        "source.anymani.anymani.assets.validator.hand_rules",
        "source.anymani.anymani.assets.exporter.urdf_writer",
        "source.anymani.anymani.assets.exporter.sidecar",
        "source.anymani.anymani.assets.exporter.hand_exporter",
        "source.anymani.anymani.assets.generator.hand_generator",
        "source.anymani.anymani.assets.asset_exporters",
        "source.anymani.anymani.assets.asset_validators",
        "source.anymani.anymani.assets.asset_generator",
    ]

    for module in modules:
        import_module(module)
