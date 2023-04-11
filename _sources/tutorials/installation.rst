Installation
============

**Requirements:** Python 3.8+.

First, clone the repositories:

.. code-block:: bash

    git clone "https://github.com/InterDigitalInc/CompressAI.git" compressai
    git clone "https://github.com/InterDigitalInc/CompressAI-Trainer.git" compressai-trainer


.. _install-virtualenv:

Virtual environment
-------------------

Using venv
~~~~~~~~~~

Create a virtual environment and install as editable:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install --editable ./compressai-trainer
    pip install --editable ./compressai


Using poetry
~~~~~~~~~~~~

Poetry helps manage version-pinned virtual environments. First, `install Poetry`_:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

.. _install Poetry: https://python-poetry.org/docs/#installation

Then, create the virtual environment and install the required Python packages:

.. code-block:: bash

    cd compressai-trainer

    # Install Python packages to new virtual environment.
    poetry install
    echo "Virtual environment created in $(poetry env list --full-path)"

    # Link to local CompressAI source code.
    poetry run pip install --editable ../compressai

To activate the virtual environment, run:

.. code-block:: bash

    poetry shell

