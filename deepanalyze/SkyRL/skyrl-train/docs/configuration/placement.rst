Model Placement and Colocation
===============================

SkyRL provides flexible control over how to distribute models across available GPU resources. You can either colocate models on the same GPUs or disaggregate them across separate GPUs, depending on your setup and requirements.

Model Components Overview
-------------------------

A typical PPO training workflow involves 5 model-based components:

- **Inference Engines** (handle text generation)
- **Policy model** (learns actions to take)
- **Reference model** (tracks the original policy)
- **Reward model** (optional; scores action quality)
- **Critic model** (estimates future rewards)

*Note: GRPO training generally uses the first 2-4 components depending on the setup – no critic model needed.*

Inference Engine Placement
--------------------------

The ``generator.run_engines_locally`` setting controls inference engine placement.

**Colocated Engines (run_engines_locally = true)**

Inference engines share GPUs with training models:

- Generation runs on the same hardware as training
- Engines will ``sleep()`` after generation to free GPU memory
- Engines will ``wake_up()`` before the next generation round

**Disaggregated Engines (run_engines_locally = false)**

Inference engines run on dedicated GPUs:

- Inference engines do not need to sleep/wake_up
- The trainer talks to Inference engines over HTTP (OpenAI API compatible)
- Updated weights are still efficiently synced to Inference engines (via NCCL, RDMA, etc.)

Training Model Placement
------------------------

The highest-level placement configuration for the training models is ``trainer.placement.colocate_all``:


**Full Colocation (colocate_all = true)**

All training models (policy, critic, reward, reference) share the same GPUs.

**Granular Control (colocate_all = false)**

The policy and critic models are not colocated, but fine-grained placement of the reference and reward models can be controlled with two additional parameters:

- ``trainer.placement.colocate_policy_ref``: Colocate policy and reference models (``true``) or place them on separate GPUs (``false``)
- ``trainer.placement.colocate_critic_reward``: Colocate critic and reward models (``true``) or place them on separate GPUs (``false``)

Hardware Configuration
----------------------

Finally, the configuration for specifying node and GPU counts for each model (along with their default values) is as follows:

.. code-block:: yaml

    trainer:
      # Training model resources
      policy_num_nodes: 1
      policy_num_gpus_per_node: 4
      critic_num_nodes: 1
      critic_num_gpus_per_node: 4
      ref_num_nodes: 1
      ref_num_gpus_per_node: 4
      reward_num_nodes: 1
      reward_num_gpus_per_node: 4

    generator:
      # InferenceEngine resources
      num_inference_engines: 1
      inference_engine_tensor_parallel_size: 4

.. note::
   **Resource Allocation Guidelines**
   
   - When ``colocate_all=true``, all training models should have identical node and GPU counts.
   - When ``generator.run_engines_locally=true``, the total number of GPUs used for Inference engines should match the total number of GPUs used for training models.
