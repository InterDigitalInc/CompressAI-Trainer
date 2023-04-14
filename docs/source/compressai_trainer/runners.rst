compressai_trainer.runners
==========================

Catalyst :py:class:`~catalyst.runners.runner.Runner`\s describe the process for training a model.

The following functions are called during the training loop:

.. code-block:: python
    :caption: Runner training loop call order.

    on_experiment_start   # Once, at the beginning.
      on_epoch_start      # Beginning of an epoch.
        on_loader_start   # For each loader (train / valid / infer).
          on_batch_start  # Before each batch.
            handle_batch  # For each image batch.
          on_batch_end
        on_loader_end
      on_epoch_end
    on_experiment_end

The training loop is effectively equivalent to:

.. code-block:: python
    :caption: Runner training loop pseudo-code.

    on_experiment_start()

    for epoch in range(1, num_epochs):
        on_epoch_start()

        for loader in ["train", "valid", "infer"]:
            on_loader_start()

            for batch in loader:
                on_batch_start()
                handle_batch(batch)
                on_batch_end()

            on_loader_end()

        on_epoch_end()

    on_experiment_end()

Please see the `Catalyst documentation`_ for more information.

.. _Catalyst documentation: https://catalyst-team.github.io/catalyst/

We provide the following pre-made runners:

- :py:class:`~compressai_trainer.runners.BaseRunner` (base compression class)
- :py:class:`~compressai_trainer.runners.ImageCompressionRunner`
- :py:class:`~compressai_trainer.runners.VideoCompressionRunner` (future release)

For guidance on defining your own runner, see: :ref:`custom-runner`.


.. automodule:: compressai_trainer.runners
   :members:
   :undoc-members:
   :show-inheritance:
