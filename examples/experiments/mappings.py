import importlib

# exp_name -> (module_path, TrainConfig class name)
_CONFIG_SPECS: dict[str, tuple[str, str]] = {
    "ram_insertion": ("experiments.ram_insertion.config", "TrainConfig"),
    "usb_pickup_insertion": ("experiments.usb_pickup_insertion.config", "TrainConfig"),
    "object_handover": ("experiments.object_handover.config", "TrainConfig"),
    "egg_flip": ("experiments.egg_flip.config", "TrainConfig"),
    "panda_pick_cube_sim": ("experiments.panda_pick_cube_sim.config", "TrainConfig"),
    "aubo_pick_cube_sim": ("experiments.aubo_pick_cube_sim.config", "TrainConfig"),
    "aubo_pick_cube_real": ("experiments.aubo_pick_cube_real.config", "TrainConfig"),
}

CONFIG_MAPPING = _CONFIG_SPECS


def get_train_config(exp_name: str):
    """Lazy-load TrainConfig to avoid importing JAX/TF for unrelated experiments."""
    if exp_name not in _CONFIG_SPECS:
        raise KeyError(f"Unknown exp_name: {exp_name}")
    module_path, class_name = _CONFIG_SPECS[exp_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
