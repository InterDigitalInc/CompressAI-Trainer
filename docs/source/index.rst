CompressAI Trainer
==================

.. image:: https://img.shields.io/github/license/InterDigitalInc/CompressAI-Trainer?color=blue
   :target: https://github.com/InterDigitalInc/CompressAI-Trainer/blob/master/LICENSE

CompressAI Trainer is a training platform that assists in managing experiments for end-to-end neural network-based compression research.

CompressAI Trainer integrates with:

- `CompressAI`_ (library)
- `Aim`_ (experiment tracker)
- `Catalyst`_ (training engine)
- `Hydra`_ (YAML configuration)

.. _Aim: https://aimstack.io/
.. _CompressAI: https://github.com/InterDigitalInc/CompressAI/#readme
.. _Catalyst: https://catalyst-team.com/
.. _Hydra: https://hydra.cc/

.. _Aim: https://aimstack.io/
.. _CompressAI: https://github.com/InterDigitalInc/CompressAI/#readme
.. _Catalyst: https://catalyst-team.com/
.. _Hydra: https://hydra.cc/

These tools give researchers more flexibility, experiment reproducibility, experiment management, and effortless visualizations and metrics.

.. figure:: media/images/aim-run-figure-rd-curves.png

    *CompressAI Trainer integrates with the Aim experiment tracker to display live visualizations of RD curves during training.*


.. toctree::
	:hidden:

	Home <self>

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden:

   tutorials/installation
   tutorials/full

..
  tutorials/first_run
  tutorials/customization
  tutorials/tips

.. toctree::
   :maxdepth: 1
   :caption: CompressAI Trainer API
   :hidden:

   compressai_train/config
   compressai_train/plot
   compressai_train/registry
   compressai_train/runners
   compressai_train/typing
   compressai_train/utils

.. toctree::
   :maxdepth: 1
   :caption: Tools
   :hidden:

   tools/rd_plotter

.. toctree::
   :caption: Development
   :hidden:

   Github repository <https://github.com/InterDigitalInc/CompressAI-Trainer/>

