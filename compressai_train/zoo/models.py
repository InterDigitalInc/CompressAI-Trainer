from compressai_train.registry import MODELS

model_architectures = MODELS


def setup_models():
    import compressai_train.models

    assert compressai_train.models
