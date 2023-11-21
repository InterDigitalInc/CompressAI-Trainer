**Note:** This only highlights "important" changes. For more details, see commit history.


### v0.3.11

- [Feature] Users may set `++misc.compile=True` with PyTorch 2 to speed up training by [compiling][torch.compile] the model.
- [Feature] RD plot: allow users to specify `results/**/*.json` paths directly.
- [Feature] Simplify `compressai-eval` usage.<br />
  To evaluate multiple models trained using CompressAI Trainer:
  ```bash
  compressai-eval \
      --config-path="$HOME/data/runs/e4e6d4d5e5c59c69f3bd7be2/configs" \
      --config-path="$HOME/data/runs/d4d5e5c5e4e6bd7be29c69f3/configs" \
      ...
  ```
  To evaluate multiple models from the CompressAI zoo:
  ```bash
  compressai-eval \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=1 ++criterion.lmbda=0.0018 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=2 ++criterion.lmbda=0.0035 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=3 ++criterion.lmbda=0.0067 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=4 ++criterion.lmbda=0.0130 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=5 ++criterion.lmbda=0.0250 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=6 ++criterion.lmbda=0.0483 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=7 ++criterion.lmbda=0.0932 \
      --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=8 ++criterion.lmbda=0.1800
  ```

[torch.compile]: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html


### v0.3.10

- [Feature] `compressai-plot --optimal=none|pareto|convex` to select which points to display on RD curve.
- [Feature] Log git repository versions in [new format][git_version_format], e.g. `v0.3.9-8-g643ce8b-dirty`.
- [Feature] [`GVAEImageCompressionRunner`].

[git_version_format]: https://github.com/InterDigitalInc/CompressAI-Trainer/commit/eba2080a87b4a7d48f068f6d37b45055385d0dc4
[`GVAEImageCompressionRunner`]: ./compressai_trainer/runners/gvae_image_compression.py


### v0.3.9

- [Feature] CLI utilities now launch from:
  - `compressai-train`
  - `compressai-eval` (evaluate a trained model to produce bitstreams/images/metrics/etc)
  - `compressai-plot` (plot RD curves)
  - `compressai_trainer.run.compressai` (wrapper around `compressai.utils`)
- [Feature] Log `x_hat` images and `debug_outputs` to experiment tracker, too. (See v0.3.4 notes.)
- [CI] Automated tests.
- [Refactor] Simplify `ImageCompressionRunner`; expose Hydra configuration `++runner.meters` and `++runner.inference` (`skip_compress`/`skip_decompress`).


### v0.3.8

- [Feature] Plot `EntropyBottleneck` distributions.
- [Fix] Ensure `update(force=True)` is called every epoch to update CDFs.


### v0.3.7

- [Feature] CLI utilities reorganized into `compressai_train.run.*`
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

