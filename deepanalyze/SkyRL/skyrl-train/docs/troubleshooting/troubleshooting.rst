Troubleshooting
===============


Multi-node Training
-------------------

For multi-node training, it is helpful to first confirm that your cluster is properly configured. We provide a script at ``scripts/multi_node_nccl_test.py`` to test multi-node communication.

To run the script, you can use the following command:

.. code-block:: bash

   uv run --isolated --env-file .env scripts/multi_node_nccl_test.py --num-nodes 2

.env is optional, but it is recommended to use for configuring environment variables.

Note on ``LD_LIBRARY_PATH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS). We've seen issues with `uv` where the ``LD_LIBRARY_PATH`` is not exported even if it is set in the ``.env`` file. It is recommended to set the ``SKYRL_LD_LIBRARY_PATH_EXPORT=1`` in the ``.env`` file and set ``LD_LIBRARY_PATH`` directly in the current shell.



