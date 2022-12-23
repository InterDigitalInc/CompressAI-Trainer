compressai_train.registry
=========================

Register models, runners, etc, to make them accessible via dynamic YAML configuration.

Registering maps a string name to a concrete creation method or class.
This allows us to dynamically create an object depending on the given string name at runtime.

All models and runners should be registered and imported, as described in :ref:`custom-model` and :ref:`custom-runner`.

.. code-block:: python
    :caption: compressai/models/custom.py

    from compressai.registry import register_model
    from .base import CompressionModel

    @register_model("my_custom_model")
    class MyCustomModel(CompressionModel):
        def __init__(self, N, M):
            ...

.. code-block:: python
    :caption: compressai_train/runners/custom.py

    from compressai.registry import register_runner
    from .base import BaseRunner

    @register_runner("CustomImageCompressionRunner")
    class CustomImageCompressionRunner(BaseRunner):
        ...


.. automodule:: compressai_train.registry


torch
-----

.. automodule:: compressai_train.registry.torch
   :members:
   :undoc-members:


catalyst
--------

.. automodule:: compressai_train.registry.catalyst
   :members:
   :undoc-members:
