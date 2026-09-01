import os

from pipeline.config.configs import Config, PROJECT_ROOT, MODELS_DIR


def test_project_root_resolves_to_prototype_dir():
    assert os.path.isdir(PROJECT_ROOT)
    assert os.path.basename(PROJECT_ROOT) == "ct-tampering-detector"


def test_models_dir_is_under_project_root():
    assert MODELS_DIR.startswith(PROJECT_ROOT)
    assert os.path.basename(MODELS_DIR) == "models"


def test_config_model_paths_point_under_models_dir():
    for path in [
        Config.REAL_FAKE_MODEL_PATH,
        Config.INJECTED_REMOVED_MODEL_PATH,
        Config.BEST_CHECKPOINT,
        Config.REMOVAL_BEST_DICE_MODEL,
    ]:
        assert path.startswith(MODELS_DIR)
