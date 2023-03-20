**Note:** This only highlights "important" changes. For more details, see commit history.


### v0.3.7

- [Feature] CLI utilities now organized as:
  - `compressai_trainer.run.train`
  - `compressai_trainer.run.eval_model` (new! evaluate a trained model to produce bitstreams/images/metrics/etc)
  - `compressai_trainer.run.plot_rd`
  - `compressai_trainer.run.compressai` (wrapper around `compressai.utils`)
- [Refactor] `ImageCompressionRunner.predict_batch` now *only* predicts batches.
- [Chore] Upgrade various dependencies (`poetry update`).


### v0.3.6

- [Feature] CLIC 2020 datasets.
- [Chore] Upgrade to `aim==3.16.0`.


### v0.3.5

- [Feature] RD curves for MS-SSIM.
- [Refactor] Simplify `ImageCompressionRunner` by extracting loggers/etc.
- [Fix] CompressAI adapter for "psnr" being renamed to "psnr-rgb".


### v0.3.4

- [Feature] Users of `ImageCompressionRunner` may now plot featuremaps/images during training by specifying `"debug_outputs"` in the forward/compress/decompress return dicts. For instance,
  ```python
  def forward(self, x):
      ...
      return {
          "likelihoods": {"y": y_likelihoods},
          "debug_outputs": {
              "y_hat": y_hat,
              "means_hat": means_hat,
              "scales_hat": scales_hat,
              "nll": -y_likelihoods.log2(),
          },
      }
  ```
  The featuremaps are outputted in the configured `paths.images` directory.


### v0.3.3

- [Chore] Update license copyright year to 2023.


### v0.3.2

- [Feat] RD plot standard codecs (e.g. VTM).
- [Feat] Save `runs/$RUN_HASH/configs/config.yaml`.
- [Fix] Tensorboard logging.


### v0.3.0

- [Refactor] Rename `compressai_train` -> `compressai_trainer`.


### v0.2.17

- [Docs] Documentation! https://interdigitalinc.github.io/CompressAI-Trainer/


### v0.2.16

- [Feat] Separately configurable optimizers for net/aux (e.g. Adam, etc).
- [Chore] Revert to `aim==3.14.4`.


### v0.2.15

- [Feat] RD plot individual per-sample points. (e.g. kodim01, kodim02, ..., kodim24)
- [Chore] Upgrade to `aim==3.15.1`.

