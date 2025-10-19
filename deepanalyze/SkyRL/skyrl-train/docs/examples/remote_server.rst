Remote Inference Engine
=========================================
This example shows how to run GRPO training on GSM8K using a standalone inference engine instance behind an HTTP server. 

In this example, we will be using 4 GPUs for training (policy and ref models) and 4 GPUs for inference (inference engine) on a single node.

Prerequisites
----------------------

First, make sure you are familiar with the standard setup process for running GRPO training. See :doc:`../getting-started/quickstart` for more details.


Starting the Remote Inference Engine
------------------------------------

We provide an example remote server implementation at :code_link:`skyrl_train/inference_engines/vllm/vllm_server.py`.
In addition to the standard OpenAI-compatible generation endpoints (``/v1/chat/completions`` and ``/v1/completions``), the remote server simply needs to expose endpoints for the following operations:

- ``/sleep`` and ``/wake_up`` to sleep and wake up the inference engine (only needed if you want to colocate the inference engine with the training process)
- ``/init_weight_update_communicator`` to create a process group for weight updates
- ``/update_weight`` to update the model weights

That's it! We use a custom ``WorkerWrap`` class at :code_link:`skyrl_train/inference_engines/vllm/vllm_engine.py` with VLLM's worker extension API to enable efficient NCCL weight updates.

To launch the server, run the following command (the full script is at :code_link:`examples/remote_inference_engine/run_vllm_server.sh`):

.. code-block:: bash

    uv run --isolated --extra vllm -m skyrl_train.inference_engines.vllm.vllm_server \
        # model and tensor parallel size
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --tensor-parallel-size 4 \

        # host and port for the server
        --host 127.0.0.1 \
        --port 8001 \

        # seed for reproducibility
        --seed 42 \

        # max context length
        --max-model-len 4096 \

        # vllm performance related settings
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.9 \
        --enable-sleep-mode \
        --max-num_batched_tokens 8192 \
        --max-num-seqs 1024 \

        # other misc settings
        --trust-remote-code \
        --distributed-executor-backend ray \

        # worker extension class for handling weight updates
        --worker-extension-cls skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap

Starting Training
----------------------

Now that we've started our remote inference engine (located behind ``127.0.0.1:8001``), we can start a training run!

To start training, we need to set up our training script. You can find a complete example in :code_link:`examples/remote_inference_engine/run_remote.sh`:

.. code-block:: bash

    uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
        # Setup for training with a remote inference engine
        generator.run_engines_locally=False \
        generator.remote_inference_engine_urls="['127.0.0.1:8001']" \
        generator.override_existing_update_group=True \

        # sampling parameters for generation
        generator.sampling_params.temperature=0.6 \
        generator.sampling_params.top_p=0.95 \

        # Data setup
        data.train_data="['$HOME/data/gsm8k/train.parquet']" \
        data.val_data="['$HOME/data/gsm8k/validation.parquet']" \

        # Policy model - make sure this is the same model used to launch the inference engine server
        trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
        trainer.algorithm.advantage_estimator="grpo" \

        # Whether or not to colocate all models on the same set of GPUs - we set it to false here,
        # but you can colocate even with a standalone inference engine!
        trainer.placement.colocate_all=False \

        # Model placement arguments for policy and ref models - make sure that the total number of gpus 
        # used for training and inference is maximized
        trainer.placement.policy_num_gpus_per_node=4 \
        trainer.placement.ref_num_gpus_per_node=4 \

        # Training batch size and mini/micro batch sizes for logprobs + training passes
        trainer.train_batch_size=64 \
        trainer.policy_mini_batch_size=64 \
        trainer.micro_forward_batch_size_per_gpu=20 \
        trainer.micro_train_batch_size_per_gpu=20 \

        # Evaluation
        trainer.eval_batch_size=1024 \
        trainer.eval_before_train=true \
        trainer.eval_interval=5 \

        ... # Other parameters (see `examples/remote_inference_engine/run_remote.sh` for more)

.. tip:: 

With remote servers, there can be non-trivial HTTP overhead during generation. When running training and inference in the same Ray cluster, it is recommended to use `run_engines_locally=True` to maximize throughput

Launching Your Training Run
---------------------------

You're done setting up! Now let's get our training run started!

.. code-block:: bash

   export WANDB_API_KEY=your_wandb_api_key
   bash examples/remote_inference_engine/run_remote.sh

What's Next?
------------

Now that you've set up training with a remote inference engine, you might want to explore ways of speeding up training:

- :doc:`../tutorials/async`: Asynchronous off-by-one training in < 100 lines of code!

