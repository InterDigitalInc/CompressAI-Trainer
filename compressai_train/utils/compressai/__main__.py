import importlib
import sys

from compressai_train.zoo import setup_models

setup_models()

if __name__ == "__main__":
    _, util_name, *argv = sys.argv
    if util_name == "update_and_eval_model":
        from . import update_and_eval_model

        main = update_and_eval_model.main
    else:
        module = importlib.import_module(
            f"compressai.utils.{util_name}.__main__",
        )
        main = module.main

    main(argv)
