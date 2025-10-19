Installation
============

Requirements
------------
- CUDA version 12.8
- `uv <https://docs.astral.sh/uv/>`_

We use `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. We also make use of the `uv` and `ray` integration to manage dependencies for ray workers. 

If you're :ref:`running on an existing Ray cluster <running-on-existing-ray-cluster>`, we suggest using Ray 2.48.0 and Python 3.12. However, we support Ray versions >= 2.44.0. 

.. warning::

    ⚠️ We do not recommend using Ray 2.47.0 and 2.47.1 for SkyRL due to known issues in the uv+ray integration.

Docker (recommended)
---------------------

We provide a docker image with the base dependencies ``erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8`` for quick setup. 

1. Make sure to have `NVIDIA Container Runtime <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ installed.

2. You can launch the container using the following command:

.. code-block:: bash

    docker run -it  --runtime=nvidia --gpus all --shm-size=8g --name skyrl-train erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8 /bin/bash

3. Inside the launched container, setup the latest version of the project:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd SkyRL/skyrl-train


That is it! You should now be able to run our :doc:`quick start example <quickstart>`.

Install without Dockerfile
--------------------------

For installation without the Dockerfile, make sure you meet the pre-requisities: 

- CUDA 12.8
- `uv <https://docs.astral.sh/uv/>`_
- `ray <https://docs.ray.io/en/latest/>`_ 2.48.0


System Dependencies
~~~~~~~~~~~~~~~~~~~

The only packages required are `build-essential` and `libnuma <https://github.com/numactl/numactl>`_. You can install them using the following command:

.. code-block:: bash

    sudo apt update && sudo apt-get install build-essential libnuma-dev

This will require sudo privileges. If you are running on a machine without sudo access, we recommend using the Dockerfile.

Installing SkyRL-Train
~~~~~~~~~~~~~~~~~~~~~~

All project dependencies are managed by `uv`.

Clone the repo and `cd` into the `skyrl` directory:

.. code-block:: bash

    git clone https://github.com/novasky-ai/SkyRL.git
    cd SkyRL/skyrl-train 

Base environment
~~~~~~~~~~~~~~~~

We recommend having a base virtual environment for the project.

With ``uv``: 

.. code-block:: bash

    uv venv --python 3.12 <path_to_venv>

If ``<path_to_venv>`` is not specified, the virtual environment will be created in the current directory at ``.venv``.

.. tip::
    Because of how Ray ships content in the `working directory <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html>`_, we recommend that the base environment is created *outside* the package directory. For example, ``~/venvs/skyrl-train``.

Then activate the virtual environment and install the dependencies.

.. code-block:: bash

    source <path_to_venv>/bin/activate
    uv sync --active --extra vllm

With ``conda``: 

.. code-block:: bash

    conda create -n skyrl-train python=3.12
    conda activate skyrl-train

After activating the virtual environment, make sure to configure Ray to use `uv`:

.. code-block:: bash

    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
    # or add to your .bashrc
    # echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc

Initialize Ray cluster
----------------------

Finally, you can initialize a Ray cluster using the following command (for single-node):

.. code-block:: bash

    ray start --head 
    # sanity check
    # ray status


.. note::
    For multi-node clusters, please follow the `Ray documentation <https://docs.ray.io/en/latest/cluster/getting-started.html>`_.

You should now be able to run our :doc:`quick start example <quickstart>`.

.. _running-on-existing-ray-cluster:

Running on an existing Ray cluster
----------------------------------

For running on an existing Ray cluster, you need to first make sure that the python version used is 3.12. 

Ray >= 2.48.0
~~~~~~~~~~~~~

We recommend using Ray version 2.48.0 and above for the best experience. In this case, you can simply use the ``uv run`` command to get training started.

.. code-block:: bash

    uv run ... --with ray==2.xx.yy -m skyrl_train.entrypoints.main_base ...

Ray < 2.48.0
~~~~~~~~~~~~
SkyRL-Train is compatible with any Ray version 2.44.0 and above (except 2.47.0 and 2.47.1 -- which we do not recommend due to an issue in the uv + Ray integration). 
Since we use a uv lockfile to pin dependencies, the best way to run SkyRL-Train on a custom Ray version (say 2.46.0) would be to override the version at runtime with the ``--with`` flag. 
For example, to run with Ray 2.46.0, you can do:

.. code-block:: bash

    uv run .... --with ray==2.46.0 -m skyrl_train.entrypoints.main_base ...

For ray versions >= 2.44.0 but < 2.48.0, you additionally need to install vllm in the base pip environment, and then re-install ray to your desired version to ensure that the uv + Ray integration works as expected. 
We include these dependencies in the legacy Dockerfile: `Dockerfile.ray244 <https://github.com/NovaSky-AI/SkyRL/blob/main/docker/Dockerfile.ray244>`_, or you can install them manually:

.. code-block:: bash

    pip install vllm==0.9.2 --extra-index-url https://download.pytorch.org/whl/cu128
    pip install ray==2.46.0 omegaconf==2.3.0 loguru==0.7.3 jaxtyping==0.3.2 pyarrow==20.0.0


.. warning::
    
    ⚠️ We do not recommend using uv versions 0.8.0, 0.8.1, or 0.8.2 due to a `bug <https://github.com/astral-sh/uv/issues/14860>`_ in the ``--with`` flag behaviour.

Development 
-----------

For development, refer to the :doc:`development guide <development>`.