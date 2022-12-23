RD curve plotter
================

``examples/plot.py`` is an RD curve plotter that can query metrics from the experiment tracker (Aim).

Users may specify what to plot using groups of the CLI flags
``--query``, ``--curves``, and ``--pareto``.
If desired, one may plot multiple query groups within the same plot.


CLI help
--------

``--aim_repo``
~~~~~~~~~~~~~~

Aim repo to query metrics from.


``--query``
~~~~~~~~~~~

Query selector for relevant runs to aggregate plotting data from.

Default:

.. code-block:: bash

    --query=''  # Gets all runs.

Examples:

.. code-block:: bash

    --query='run.hash == "e4e6d4d5e5c59c69f3bd7be2"'
    --query='run.model.name == "bmshj2018-factorized"'
    --query='run.experiment.startswith("some-prefix-")'
    --query='run.created_at >= datetime(1970, 1, 1)'
    --query='run.criterion.lmbda < 0.02 and run.hp.M == 3 * 2**6'


``--curves``
~~~~~~~~~~~~

For the current query, specify a grouping and format for the
curves. One may specify multiple such groupings for a given
query within a list. Each unique "name" produces a unique curve.
If a key (e.g. ``"name"``, ``"x"``, ``"y"``) is not specified,
its default value is used.
For ``"name"``, one may specify a hparam by key via ``"{hparam}"``.

Default (where ``--x="bpp"`` and ``--y="psnr"``):

.. code-block:: bash

    --curves='[{"name": "{experiment}", "x": args.x, "y": args.y}]'

Examples:

- Show both model name and experiment name:

  .. code-block:: bash

      --curves='[{"name": "{model.name} {experiment}"}]'

- Group by ``hp.M``:

  .. code-block:: bash

      --curves='[{"name": "{experiment} (M={hp.M})"}]'

- Multiple metrics as separate curves:

  .. code-block:: bash

      --curves='[
          {"name": "{experiment} (RGB-PSNR)", "y": "psnr_rgb"},
          {"name": "{experiment} (YUV-PSNR)", "y": "psnr_yuv"},
      ]'

- Multi-rate models (e.g. G-VAE):

  .. code-block:: bash

      --curves='[{
          "name": "{experiment} {run.hash}",
          "x": ["bpp_0", "bpp_1", "bpp_2", "bpp_3"],
          "y": ["psnr_0", "psnr_1", "psnr_2", "psnr_3"],
      }]'


``--pareto``
~~~~~~~~~~~~

Show only pareto-optimal points on curve for respective query.

Default:

.. code-block:: bash

    --pareto=False


..
  ``--show``
  ~~~~~~~~~~

  Show figure in browser.



Examples
--------

- Plot all experiments since date, automatically grouping curves by experiment/model:

  .. code-block:: bash

      python examples/plot.py \
          --aim_repo="./logs/aim/main" \
          --query='run.created_at >= datetime(1970, 1, 1)'

- Plot simple curve for specific run hashes:

  .. code-block:: bash

      python examples/plot.py \
          --aim_repo="./logs/aim/main" \
          --query='run.hash in [
              "e4e6d4d5e5c59c69f3bd7be2",
              "b3d5Bb2c5e3a6f49c69f39f6",
              "d4e6e4c5e5d59c69f3bd7bd3",
              ...
          ]'

- Plot single multi-rate model (e.g. G-VAE):

  .. code-block:: bash

      python examples/plot.py \
          --aim_repo="./logs/aim/main" \
          --query='run.hash == "e4e6d4d5e5c59c69f3bd7be2"' \
          --curves='[{
              "x": ["bpp_0", "bpp_1", "bpp_2", "bpp_3"],
              "y": ["psnr_0", "psnr_1", "psnr_2", "psnr_3"],
          }]'

- Plot multiple metrics (e.g. ``psnr_rgb`` and ``psnr_low``) on the same plot:

  .. code-block:: bash

      python examples/plot.py \
          --aim_repo="./logs/aim/main" \
          --curves='[
              {"name": "{experiment} (RGB-PSNR)", "y": "psnr_rgb"},
              {"name": "{experiment} (YUV-PSNR)", "y": "psnr_yuv"},
          ]'

- Plot multiple metrics (e.g. ``psnr_base`` and ``psnr_enhancement``) on the same plot:

  .. code-block:: bash

      python examples/plot.py \
          --aim_repo="./logs/aim/main" \
          --curves='[
              {"name": "{experiment} (base layer)", "y": "psnr_base"},
              {"name": "{experiment} (enhancement layer)", "y": "psnr_enhancement"},
          ]'

