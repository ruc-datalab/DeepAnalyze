=================
Development Guide
=================


Pre-requisities
===============

Follow the :doc:`installation guide <installation>`. Make sure that the installation works with our :doc:`quick start example <quickstart>`.


Modifying the code
==================

- Are you adding a new environment or task? Follow the :doc:`new task tutorial <../tutorials/new_env>`. Your custom code can be placed anywhere - building on top of ``skyrl_train`` as a package - but we recommend structuring it as a folder similar to `skyrl-train/examples/multiply <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply>`_. 

- Are you creating a custom ``Generator`` (ex: porting an agent harness, implementing custom trajectory generation logic, etc.)? Same as the above: your custom code can be placed anywhere and we typically use ``skyrl-train/examples/`` for this.

- Are you modifying the training code (ex: adding a new algorithm, changing the training loop etc)? You would modify the code in :code_link:`skyrl_train`. 

- Are you modifying the existing environment code (ex: adding a custom method for all ``Env`` classes, improving the ``SearchEnv`` implementation)? You would modify the code in  `skyrl-gym <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym/>`_. Note: you do **not** have to modify the ``skyrl-gym`` package for adding a new environment or task. 


Contributing to SkyRL
=====================

We welcome all forms of contributions to SkyRL, including bug reports, feature requests, documentation improvements, code contributions etc. 

Testing
-------

For code contributions, we recommend running unit tests before submitting a pull request. 

SkyRL-train
^^^^^^^^^^^

CPU tests
~~~~~~~~~

.. code-block:: bash

    cd skyrl-train # make sure you are in the correct directory
    uv run --isolated --extra dev pytest tests/cpu

GPU tests
~~~~~~~~~

The GPU tests require a node with atleast 8 GPUs. They have been tested on a 8xH100 node, but should work even on 8xA100 nodes. We are actively working on making these more accessible.

.. code-block:: bash

    cd skyrl-train # make sure you are in the correct directory 

The tests assume that the GSM8K dataset is downloaded to ``~/data/gsm8k``. If you have not downloaded the dataset, you can do so by running the following command:

.. code-block:: bash
    
    uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir ~/data/gsm8k

Finally, you can run the tests by running the following command:

.. code-block:: bash

    uv run --isolated --extra dev --extra vllm pytest tests/gpu


SkyRL-gym
^^^^^^^^^

You can run the tests for the ``skyrl-gym`` package by running the following command:

.. code-block:: bash

    cd skyrl-gym # make sure you are in the correct directory
    uv run --isolated --extra dev pytest tests/



Linting and Formatting 
----------------------

We use pre-commit hooks to automatically lint and format the code. You can install the pre-commit hooks by running the following command:

.. code-block:: bash

    uv pip install pre-commit
    pre-commit install

If you wish to manually run linting and formatting:

.. code-block:: bash

    bash format.sh # run from the root directory

Documentation
-------------

We welcome contributions to the documentation. We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to build the documentation. The documentation is stored in the ``skyrl-train/docs/`` folder. 

Developing locally
^^^^^^^^^^^^^^^^^^

We recommend building the documentation locally to preview your changes.

.. code-block:: bash

    cd skyrl-train # make sure you are in the correct directory
    bash docs/build.sh

This will build the documentation and serve it at `http://127.0.0.1:8000 <http://127.0.0.1:8000>`_.