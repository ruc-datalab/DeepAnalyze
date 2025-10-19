Configuration Overview
======================

Data Configuration
------------------

.. code-block:: yaml

    data:
    train_data: ["${oc.env:HOME}/data/gsm8k/train.parquet"]
    val_data: ["${oc.env:HOME}/data/gsm8k/validation.parquet"]

- ``data.train_data``: A list of files for the training dataset in parquet format.
- ``data.val_data``: A list of files for evaluation in parquet format.


.. note::
    Currently, all datasets are loaded into memory, so the dataset size is limited by available CPU memory on a worker node.


Model Placement Configuration
-----------------------------

.. code-block:: yaml

  placement:
    colocate_all: true
    colocate_policy_ref: true
    colocate_critic_reward: false
    policy_num_nodes: 1
    policy_num_gpus_per_node: 4
    critic_num_nodes: 1
    critic_num_gpus_per_node: 4
    ref_num_nodes: 1
    ref_num_gpus_per_node: 4
    reward_num_nodes: 1
    reward_num_gpus_per_node: 4

For an in-depth guide on model placement and colocation, please refer to the :doc:`model placement and colocation guide <placement>`.

General Training Configuration
------------------------------

.. code-block:: yaml

    epochs: 1  # Number of passes over the full dataset
    update_epochs_per_batch: 1
    train_batch_size: 1024
    policy_mini_batch_size: 256
    critic_mini_batch_size: 256
    micro_train_batch_size_per_gpu: 1
    micro_forward_batch_size_per_gpu: 1
    update_ref_every_epoch: false
    use_sample_packing: true
    max_prompt_length: 512
    gradient_checkpointing: true
    seed: 42


- ``epochs``: Number of epochs/ passes over the full dataset (similar to SFT)
- ``update_epochs_per_batch``: Number of gradient update passes over each training batch. This is equivalent to the concept of "PPO epochs" where you iterate over the same experience multiple times.
- ``train_batch_size``: Batch size of prompts used for each dataloader step.
- ``policy_mini_batch_size``: Mini batch size used during RL training step. Each mini batch corresponds to one optimizer step. For example, if the ``train_batch_size`` is 4 and ``policy_mini_batch_size`` is 2, then there will be 2 optimizer steps (i.e., model updates) for a given training batch. Note that is this the global mini batch size. The actual size of the mini batch per worker would be ``policy_mini_batch_size/ number of DP ranks``
- ``critic_mini_batch_size``: Similar to ``policy_mini_batch_size`` but for the critic model (if applicable). Note that in general, the critic model can tolerate off-policy updates more than the policy. Thus, you would want to set ``critic_mini_batch_size`` to be lower compared ``policy_mini_batch_size`` (i.e., more critic updates).
- ``micro_train_batch_size_per_gpu``: Micro batch size during training step. This is common for both policy and critic models. Each mini batch is split into micro batches of this size, gradients are computed and accumulated over these micro batches.
- ``micro_forward_batch_size_per_gpu``: Micro batch size during forward pass (i.e., for log probability or value computation). This is common for both policy and critic models. Each mini batch is split into micro batches of this size, model forward pass is performed over these micro batches.
- ``update_ref_every_epoch``: Whether to update the reference model every epoch.
- ``use_sample_packing``: Whether to use sample packing during model forward pass (common for all models).
- ``max_prompt_length``: Maximum prompt length during training. Longer prompts will be truncated.
- ``gradient_checkpointing``: Whether to use gradient checkpointing.
- ``seed``: Random seed for training.


.. tip::
  If you're facing issues with tuning the right values for ``micro_train_batch_size_per_gpu``, ``policy_mini_batch_size`` and ``micro_forward_batch_size_per_gpu``, see ``utils/utils.py::validate_batch_sizes`` for details on constraints.

Evaluation Configuration
------------------------------
.. code-block:: yaml

    eval_batch_size: 1024
    eval_before_train: true
    eval_interval: 5 # Set to -1 to disable evaluation.

- ``eval_batch_size``: Batch size for evaluation.
- ``eval_before_train``: Whether to evaluate the model before training.
- ``eval_interval``: The frequency of evaluating the model with the validation dataset (in terms of number of steps). If set to ``-1``, evaluation will not be performed.

.. note::
  If multiple validation datasets are provided (e.g. ``data.val_data="['$DATA_DIR/validation1.parquet', '$DATA_DIR/validation2.parquet']" \``),
  then the evaluation will be performed on all of them. The metrics for each dataset, and the aggregated metrics, will
  all be logged in WandB. If ``dump_eval_results`` is set to ``true``, the per-dataset and aggregated results will be
  dumped.

Checkpoint Configuration
---------------------------------------

.. code-block:: yaml

    resume_mode: latest # null/"none", "latest", "from_path"
    resume_path: null
    ckpt_path: "${oc.env:HOME}/ckpts/" # Path for resumable training checkpoints (model state, optimizer state, etc.)
    max_ckpts_to_keep: -1 # -1 to keep all checkpoints, N to keep the last N checkpoints
    ckpt_interval: 10  # Save full training checkpoint every `ckpt_interval` steps.
    hf_save_interval: -1  # Save HF format model(s)every `hf_save_interval` steps.
    export_path: "${oc.env:HOME}/exports/" # Path for exported artifacts (HF models, debug dumps, etc.)
    project_name: "skyrl"
    run_name: "test_run"
    logger: "wandb"

For an in-depth guide on checkpointing and resumption, please refer to the :doc:`checkpointing guide <../checkpointing-logging/checkpointing>`.

Logging and Debugging Configuration
-----------------------------------

.. code-block:: yaml

    logger: "wandb"
    project_name: "skyrl"
    run_name: "test_run"
    dump_data_batch: false
    dump_eval_results: true

- ``logger``: Logger to use. Currently, we support ``wandb``, ``mlflow``, and ``console``. ``console`` will simply log metrics to the console.
- ``project_name``: Name of the project in WandB and MLFlow.
- ``run_name``: Name of the run in WandB and MLFlow.
- ``dump_data_batch``: Whether to dump the data batch to a file. This is useful for debugging. When ``true``, the data batch will be dumped to a file in the ``export_path`` directory. The training batch at global step ``N`` is saved to ``self.cfg.trainer.export_path / "dumped_data" / global_step_N_training_input``
- ``dump_eval_results``: Whether to dump the evaluation results to a file. When ``true``, the full evaluation results will be dumped to a file in the ``export_path`` directory. The evaluation results at global step ``N`` is saved to ``self.cfg.trainer.export_path / "dumped_eval" / global_step_N_eval_results``

Training Backends
-----------------

We support three backends: FSDP1, FSDP2 and DeepSpeed. The backend can be chosen with ``trainer.strategy`` field.

.. _fsdp-configurations:

FSDP Configuration
~~~~~~~~~~~~~~~~~~

We use the same configuration group for FSDP1 and FSDP2

.. code-block:: yaml

    fsdp_config:
        cpu_offload: false # offload params + optimizer state to cpu during fwd pass
        reshard_after_forward: true # fsdp2 only, [True, False, int between 1 and fsdp_size]
        fsdp_size: -1

- ``cpu_offload``: Whether to train with CPU offloading (i.e., offload state during forward pass). This corresponds to `cpu_offload <https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel>`_  parameter in FSDP1 and `offload_policy <https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.fully_shard>`_ in FSDP2.
- ``reshard_after_forward``: Whether to re-shard FSDP model after forward pass. This is a FSDP2 specific configuration, please refer to the `FSDP2 docs <https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.fully_shard>`_ for more details. If set to ``false``, this would retain the full model parameters on each worker (similar to DeepSpeed's ZeRO stage 2).
- ``fsdp_size``: The group size within which worker state is sharded with FSDP. This is a parameter to be used for hybrid sharding in multi-node settings. For example, if the number of workers in the actor group is 8, with 4 in each node, and ``fsdp_size`` is 4, then the training state will be fully sharded across 4 ranks in each node, but replicated (DP) across nodes.

.. note::
    ``cpu_offload`` is different from worker state offloading with model colocation.

    In FSDP, ``cpu_offload`` will offload parameter and optimizer state to CPU memory and only copy over model parameters to GPU during model forward pass.

    In `skyrl-train`, we offload worker state in certain colocation settings - however this happens only after the training step/ log probability computation - thus optimizer step and model forward pass happen as usual with sharded parameters on GPU. For more details, refer to the guide on :doc:`model placement and colocation <placement>`

.. _deepspeed-configurations:

DeepSpeed Configuration
~~~~~~~~~~~~~~~~~~~~~~~

For DeepSpeed, please refer to DeepSpeed's `configuration guide <https://www.deepspeed.ai/docs/config-json/>`_ for more details. In general, the user experience with DeepSpeed is better and most parameters can set to ``auto`` for DeepSpeed to automatically configure. Here are a couple of important parameters:

- ``deepspeed_config.zero_optimization.stage``: Which ZeRO stage to use. Currently, we only support stage 3.
- ``deepspeed_config.zero_optimization.zero_hpz_partition_size``: Hierarchical Partitioning size. This is similar (although not equivalent) to hybrid sharding in FSDP.
- ``deepspeed_config.gradient_clipping``: This should not be set during training. We instead provide a common optimizer config ``optimizer_config.max_grad_norm`` that will handle gradient clipping configuration for all training backends.

Optimizer Configuration
-----------------------
For both the critic and policy model, we provide a common optimizer configuration

.. code-block:: yaml

    optimizer_config:
       lr: 1.0e-6
       adam_betas: [0.9, 0.999]
       weight_decay: 1e-2
       max_grad_norm: 1.0
       offload_after_step: true
       num_warmup_steps: 0
       scheduler: "constant_with_warmup"

- ``optimizer_config.lr``: Learning rate for the optimizer
- ``optimizer_config.adam_betas``: Betas for AdamW optimizer.
- ``optimizer_config.weight_decay``: L2 regularization strength for AdamW.
- ``optimizer_config.max_grad_norm``: Gradient clipping parameter. The total L2 norm of the model gradients will be scaled to this value during training.
- ``optimizer_config.offload_after_step``: Whether to offload optimizer state to CPU after step if colocated. When generation and training workers are colocated, we recommend using the default setting of ``true``. In some cases with non-colocation, it can be desirable to leave optimizer state on GPU memory to avoid offloading costs as well as additional CPU memory usage.
- ``optimizer_config.num_warmup_steps``: Number of warmup steps for the learning rate scheduler.
- ``optimizer_config.scheduler``: Which learning rate scheduler to use. Intended to align with ``transformers.SchedulerType`` from ([Hugging Face Docs](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType)).

Policy Configuration
--------------------

This section configures the policy model used for training, including optimizer, FSDP, and sequence parallelism options.

.. code-block:: yaml

   policy:
     model:
       path: "Qwen/Qwen2.5-1.5B-Instruct"  # Hugging Face model path for the policy model
     deepspeed_config: ${deepspeed_config.train}  # Reference to default deepspeed config

     optimizer_config:
       lr: 1.0e-6  # Learning rate
       adam_betas: [0.9, 0.999]  # Betas for Adam optimizer
       weight_decay: 1e-2  # L2 regularization strength
       max_grad_norm: 1.0  # Gradient clipping
       offload_after_step: true  # Offload optimizer state to CPU after step (if colocated)

     fsdp_config:
       cpu_offload: false  # Offload model params to CPU during forward
       reshard_after_forward: true  # Re-shard FSDP model after forward pass
       fsdp_size: -1  # Auto FSDP group sizing

     sequence_parallel_size: 1  # sequence parallel size

     use_torch_compile: false  # Enable torch compile for the entropy calculation
     record_memory: false  # Dump memory snapshot for debugging

- ``policy.deepspeed_config``: To be customized if using ``trainer.strategy='deepspeed'``.
- ``policy.optimizer_config``: Optimizer configuration for the policy model
- ``policy.fsdp_config``: FSDP configuration, applicable if ``trainer.strategy='fsdp'``.
- ``policy.sequence_parallel_size``: Sequence parallel size. We implement `Ulysses sequence parallelism <https://arxiv.org/abs/2309.14509>`_
- ``policy.use_torch_compile``: Whether to enable torch compile for entropy calculation
- ``policy.record_memory``: Whether to record memory usage. If ``True``, this will use PyTorch's `memory snapshotting utility <https://docs.pytorch.org/docs/stable/torch_cuda_memory.html>`_ to record memory usage and dump memory snapshots after each policy model training step.



Critic Configuration
--------------------

We support similar configuration options as the policy model.

.. code-block:: yaml

    critic:
      model:
        path: null
      deepspeed_config: ${deepspeed_config.train}
      optimizer_config:
        lr: 5.0e-6
        adam_betas: [0.9, 0.999]
        weight_decay: 1e-2
        max_grad_norm: 1.0 # gradient clipping
        offload_after_step: true # offload optimizer state to cpu after each step. Applicable only when `colocate_all=true`
      fsdp_config:
        cpu_offload: false
        reshard_after_forward: true
        fsdp_size: -1
      sequence_parallel_size: 1


Reference Model Configuration
-----------------------------


.. code-block:: yaml

    ref:
      deepspeed_config: ${deepspeed_config.eval}
      fsdp_config:
        cpu_offload: true
        reshard_after_forward: true
        fsdp_size: -1
      sequence_parallel_size: 1

- ``ref.deepspeed_config``: To be customized if using ``trainer.strategy='deepspeed'``.
- ``ref.fsdp_config``: FSDP configuration, applicable if ``trainer.strategy='fsdp'``.
- ``ref.sequence_parallel_size``: Sequence parallel size. We implement `Ulysses sequence parallelism <https://arxiv.org/abs/2309.14509>`_

.. note::

  The reference model is used only if the base model log probabilities are required either as a part of the training loss or as a part of the reward. Thus, ``trainer.algorithm.use_kl_in_reward`` or ``trainer.algorithm.use_kl_loss`` should be set to ``true`` to use the reference model. If both are ``false``, then the reference model is not instantiated.


Algorithm Configuration
-----------------------

.. code-block:: yaml

    algorithm:
      advantage_estimator: "grpo"  # "grpo", "gae", or customizable with AdvantageEstimatorRegistry

      # KL Penalty Parameters
      kl_ctrl: # only used if use_kl_in_reward is true (not applied in the case of use_kl_loss=true) - uses kl_loss_coef as the initial KL coefficient
        type: "fixed" # "fixed" or "adaptive"
        kl_target: 0.1 # target KL divergence for adaptive KL controller
        horizon: 10000 # controls the update rate of the adaptive KL controller
  
      kl_estimator_type: "k3" # "k1", "k2", "k3", "abs" - see http://joschu.net/blog/kl-approx.html for details
      use_kl_estimator_k3: false # to be deprecated, use kl_estimator_type="k3" instead
      use_abs_kl: false # to be deprecated, use kl_estimator_type="abs" instead

      # note: use_kl_in_reward and use_kl_loss should be mutually exclusive
      use_kl_in_reward: false # apply kl loss to rewards
      use_kl_loss: true # used in policy model
      kl_loss_coef: 0.001
      # this adds training batch level normalization to advantages
      advantage_batch_normalize: false
      value_head_prefix: "value_head"
      policy_loss_type: "regular" # "regular", "dual_clip", "gspo", or customizable with PolicyLossRegistry
      loss_reduction: "token_mean" # "token_mean", "sequence_mean", "seq_mean_token_sum_norm"
      grpo_norm_by_std: true # set to false to disable normalization by std in GRPO (used in Dr. GRPO)

      # GAE parameters
      lambd: 1.0
      gamma: 1.0

      # PPO parameters
      eps_clip_low: 0.2
      eps_clip_high: 0.2
      # dual clip parameters
      clip_ratio_c: 3.0

      # value loss parameters
      value_clip: 0.2
      normalize_reward: true

      # dynamic sampling parameters
      dynamic_sampling:
        type: null # filter (DAPO), replace (POLARIS/WebSailor), or null
        max_sample_batches: 30 # sample at most this many batches before stopping, -1 to sample forever
        min_replace_ratio: 0.3 # minimum proportion of good samples with which to replace bad samples (for replace strategy only)
      
      # Truncated Importance Sampling as proposed in https://fengyao.notion.site/off-policy-rl 
      use_tis: false 
      tis_imp_ratio_cap: -1.0


- ``algorithm.advantage_estimator``: Advantage estimator to use. We currently implement ``grpo``, ``gae``, ``rloo``, ``reinforce++``, and custom advantage estimators can be registered with the ``AdvantageEstimatorRegistry``.
- ``algorithm.kl_ctrl`` Configuration for the KL controller - only used if ``use_kl_in_reward`` is ``true`` (not applied in the case of ``use_kl_loss`` is ``true``). ``kl_loss_coef`` is used as the initial KL coefficient for both ``fixed`` and ``adaptive`` KL controllers.

 - ``type``: Type of KL controller to use. Options include: ``fixed`` or ``adaptive``. 
 - ``kl_target``: Target KL divergence for adaptive KL controller.
 - ``horizon``: Controls the update rate of the adaptive KL controller.

- ``algorithm.kl_estimator_type``: KL estimator type to use. Options include: ``k1``, ``k2``, ``k3``, ``abs``. See `this blog post <http://joschu.net/blog/kl-approx.html>`_ for details. We use ``k3`` as the default.
- ``algorithm.use_kl_estimator_k3``: Whether to use the k3 estimator for KL divergence calculation. The k3 estimator is the non negative kl approximation in `this blog post <http://joschu.net/blog/kl-approx.html>`_. Besides non negative, it is also unbiased and has lower variance. This flag is to be deprecated, use ``kl_estimator_type="k3"`` instead.
- ``algorithm.use_abs_kl``: Whether to use the absolute KL divergence for KL divergence calculation. This flag is to be deprecated, use ``kl_estimator_type="abs"`` instead.
- ``algorithm.use_kl_in_reward``: Whether to apply KL divergence penalty to rewards. The new rewards will be computed as ``rewards - kl * kl_loss_coef``.
- ``algorithm.use_kl_loss``: Whether to add a KL divergence loss to the policy model. The policy loss will be computed as ``policy_loss + kl * kl_loss_coef``.
- ``algorithm.kl_loss_coef``: Coefficient for the KL divergence loss.
- ``algorithm.advantage_batch_normalize``: Whether to normalize advantages by the (global) batch mean and standard deviation.
- ``algorithm.value_head_prefix``: The name used to identify the value head in the critic model.
- ``algorithm.policy_loss_type``: Type of policy loss to use. Options include:

  - ``regular``: Vanilla PPO loss with token-level importance sampling
  - ``dual_clip``: Dual clip PPO loss proposed in `this paper <https://arxiv.org/pdf/1912.09729>`_
  - ``gspo``: `Group Sequence Policy Optimization <https://arxiv.org/abs/2507.18071>`_ with sequence-level importance sampling for improved training stability. Implements "GSPO-token" variant from the paper.
  - Custom policy losses can be registered with the ``PolicyLossRegistry``

- ``algorithm.loss_reduction``: Type of loss reduction to use. Options include:

  - ``token_mean``: computes average loss over all valid tokens in the batch. Used in `DAPO <https://dapo-sia.github.io/>`_.
  - ``sequence_mean``: computes per-sequence avg token loss, then averages over the batch.
  - ``seq_mean_token_sum_norm``: computes the sum of token losses for each sequence, normalizes by the max sequence length (computed as ``cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length``), and then averages over the batch. This is used in `Dr. GRPO <https://arxiv.org/abs/2503.20783>`_.

- ``algorithm.grpo_norm_by_std``: Whether to normalize advantages by the standard deviation in GRPO. This is set to ``false`` in `Dr. GRPO <https://arxiv.org/abs/2503.20783>`_.
- ``algorithm.lambd``: Lambda parameter for GAE.
- ``algorithm.gamma``: Gamma parameter for GAE.
- ``algorithm.eps_clip_low``: Lower bound for PPO clipping.
- ``algorithm.eps_clip_high``: Upper bound for PPO clipping.
- ``algorithm.clip_ratio_c``: Clip ratio for dual clip PPO loss.
- ``algorithm.value_clip``: Clip value for value loss.
- ``algorithm.normalize_reward``: Whether to normalize critic model output (i.e., values). When ``true``, the critic model learns the mean and standard deviation of the values during training and normalizes the values during forward pass.
- ``algorithm.dynamic_sampling``: Dynamic sampling configuration.
  - ``algorithm.dynamic_sampling.type``: Type of dynamic sampling to use. Currently, we support ``filter`` (`DAPO <https://dapo-sia.github.io/>`_), ``replace`` (`POLARIS <https://hkunlp.github.io/blog/2025/Polaris/>`_ / `WebSailor <https://arxiv.org/abs/2507.02592>`_), or ``null`` for no dynamic sampling.
  - ``algorithm.dynamic_sampling.max_sample_batches``: Maximum number of batches to sample before stopping. Set to ``-1`` to sample forever.
  - ``algorithm.dynamic_sampling.min_replace_ratio``: Minimum proportion of good samples with which to replace bad samples for ``replace`` strategy.
- ``algorithm.use_tis``: Whether to use Truncated Importance Sampling (TIS) as proposed in `this blog <https://fengyao.notion.site/off-policy-rl>`_. 
- ``algorithm.tis_imp_ratio_cap``: Cap parameter for the importance ratio in TIS.


Policy Loss Formulation
~~~~~~~~~~~~~~~~~~~~~~~

It can be helpful to understand the final loss formulation to see how the different configuration options are used. The final loss is computed as below in the ``ppo_policy_loss`` function.

.. code-block:: python

  def ppo_policy_loss(
      log_probs: torch.Tensor,
      old_log_probs: torch.Tensor,
      advantages: torch.Tensor,
      config: DictConfig, # trainer.algorithm config
      loss_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:

      ratio = (log_probs - old_log_probs).exp()
      surr1 = ratio * advantages
      surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
      loss = -torch.min(surr1, surr2)
      clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
      clip_pg_losses1 = loss
      if config.policy_loss_type == "dual_clip":
        pg_losses3 = -advantages * config.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
      loss = reduce_loss(loss, loss_mask, config.loss_reduction)
      return loss, clip_ratio


Generator Configuration
-----------------------

.. code-block:: yaml

  generator:
    model_dtype: "bfloat16" # should match dtype for inference engine
    run_engines_locally: true
    num_inference_engines: 1
    backend: "vllm"
    weight_sync_backend: "nccl"
    inference_engine_tensor_parallel_size: 4
    n_samples_per_prompt: 5
    async_engine: true
    batched: true
    max_input_length: ${trainer.max_prompt_length} # max generator input length used for multi-turn conversations - for single turn set equal to max_prompt_length
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_batched_tokens: 8192
    enforce_eager: false
    gpu_memory_utilization: 0.8
    max_num_seqs: 1024
    remote_inference_engine_urls: ["127.0.0.1:8001"]
    max_turns: 1

    override_existing_update_group: "auto" # "auto", "enable", "disable"
    # sampling params for generation phase
    sampling_params:
      max_generate_length: 1024
      temperature: 1.0
      top_p: 1.0
      min_p: 0.0
      top_k: -1

    use_conversation_multi_turn: true

    # sampling params for evaluation
    eval_sampling_params:
      max_generate_length: ${generator.sampling_params.max_generate_length}
      temperature: 1.0
      top_p: 1.0
      min_p: 0.0
      top_k: -1

    # number of samples per prompt for evaluation
    eval_n_samples_per_prompt: 1

    zero_reward_on_non_stop: false

    apply_overlong_filtering: false


Inference Engine Placement Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.run_engines_locally``: Whether to use local inference engines. If ``true``, the inference engine will be initialized during the training run in the current Ray cluster. We use one Ray actor per inference replica and communication will happen via Ray object store.  If set to ``false``, then the generator expects a list of remote urls and communication will happen over HTTP.
- ``generator.num_inference_engines``: Number of inference engines to use. If ``run_engines_locally`` is ``false``, then this number should match the number of remote urls.
- ``generator.remote_inference_engine_urls``: List of remote urls to use. Applicable only when ``run_engines_locally`` is ``false``.

For more details on how different placement options work, please refer to the :doc:`placement guide <placement>`.

Weight Transfer Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.weight_sync_backend``: Backend to use for weight synchronization. Currently, we support ``nccl`` and ``gloo``.
- ``generator.override_existing_update_group``: Whether to override the existing update group for the inference engine. This is applicable only for remote inference engines. During training, `skyrl-train` forms a custom process group ("update group") with the rank 0 training worker and all the inference engine ranks.  If ``override_existing_update_group=enable``, then during initialization, a previous weight update group will be overriden in the inference engine. For example, if you have a remote server setup and you run training for the same model multiple times, it is helpful to override the previous update group. We recommend leaving this to ``auto`` - since it will automatically determine if the previous update group should be overridden based on ``run_engines_locally``.

Inference Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.backend``: Backend to use for the inference engine. We support ``vllm`` and ``sglang``. ``sglang`` is supported only for remote inference engines at the moment.
- ``generator.model_dtype``: Dtype used for the inference engine. This is also used during weight transfer - the policy model weights are casted to this dtype before being sent to the inference engine during weight transfer.
- ``generator.async_engine``:  Whether to use an asynchronous/ offline inference engine. Applicable only when ``backend="vllm"``.
- ``generator.inference_engine_tensor_parallel_size``: Tensor parallel size for the inference engine.
- ``generator.gpu_memory_utilization``: GPU memory utilization for the inference engine. Applicable only for ``run_engines_locally=true``.
- ``generator.vllm_v1_disable_multiproc``: If ``true``, this will set ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` in the environment, which makes the scheduling deterministic. This is useful for reproducibility.
- ``generator.enable_prefix_caching``: Whether to enable prefix caching for the inference engine. Applicable only when ``backend="vllm"``. This can be left to the default ``true`` in most cases. Note that in the case of remote inference engines, you would need to match the setting used when you initialized the remote servers.
- ``generator.enable_chunked_prefill``: Whether to enable chunked prefill for the inference engine. Applicable only when ``backend="vllm"``. With vLLM, this can be left to the default ``true`` in most cases.
- ``generator.max_num_seqs``: Continous batching parameter for vLLM. Maximum number of sequences to pack into a batch.
- ``generator.max_num_batched_tokens``: Continous batching parameter for vLLM. Maximum number of tokens to pack into a batch.


Generation Parameters
~~~~~~~~~~~~~~~~~~~~~

- ``generator.n_samples_per_prompt``: Number of samples to generate per prompt. Note that the total size of the training batch will be ``trainer.train_batch_size * generator.n_samples_per_prompt``.
- ``generator.batched``: Whether to use batched inference. This is applicable only for single turn generation.
- ``generator.max_input_length``: Maximum input length for the inference engine. For single turn generation, this can be same as ``trainer.max_prompt_length`` (i.e., the initial prompt length). For multi-turn generation, this is the maximum input length used for multi-turn conversations at each turn.
- ``generator.sampling_params``: Sampling parameters for the inference engine during trajectory generation phase.

    - ``generator.sampling_params.max_generate_length``: Maximum length of the generated response.
    - ``generator.sampling_params.temperature``: Temperature for the inference engine.
    - ``generator.sampling_params.top_p``: Top-p sampling parameter for the inference engine.
    - ``generator.sampling_params.min_p``: Min-p sampling parameter for the inference engine, as proposed in `this paper <https://arxiv.org/pdf/2407.01082>`_.
    - ``generator.sampling_params.top_k``: Top-k sampling parameter for the inference engine.
- ``generator.eval_sampling_params``: Sampling parameters for evaluation.
- ``generator.eval_n_samples_per_prompt``: Number of samples to generate per prompt for evaluation.
- ``generator.max_turns``: Maximum number of turns for generation with multi-turn RL.
- ``generator.use_conversation_multi_turn``: Whether to use conversation format for multi-turn generation. If set to ``true`` then observations are appended to the chat history as a new turn. If set to ``false`` then observations are appended as-is to the assistant response in token space and generation is continued  (after removing any EOS token in the response).  We've observed some cases where model can be sensitive to chat history format (ex: in SkyRL-SQL), and thus ``false`` can be used for full control over the exact tokens added after environment interaction.

Misc Configuration
~~~~~~~~~~~~~~~~~~

- ``generator.zero_reward_on_non_stop``: Whether to set the reward to 0 if the `stop_reason` is not `stop`. Cases where this is useful: Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response, we typically don't want to reward it. This is a general setting for all environments.
- ``generator.apply_overlong_filtering``: Whether to apply DAPO Overlong Filtering to the loss masks. For each trajectory that exceeds the max length (i.e., truncated and does not end with an EOS token), this masks out every token in the loss mask.