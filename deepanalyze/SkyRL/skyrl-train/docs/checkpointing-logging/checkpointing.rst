Checkpointing
=============

SkyRL provides checkpointing features to resume training from a previous state. Training state is saved at regular intervals and provides flexible configuration options for checkpoint management.

What State is Saved
-------------------

SkyRL saves several types of state to enable complete training resumption:

**Model States**
  - **Policy Model**: Model parameters, optimizer state, and learning rate scheduler state
  - **Critic Model**: Model parameters, optimizer state, and learning rate scheduler state (if critic is enabled)
  - **Reference Model**: Not checkpointed (recreated from policy model)

**Training State**
  - **Global Step**: Current training step counter
  - **Configuration**: Complete training configuration used
  - **Dataloader State**: Current position in dataset iteration (enables resuming from exact data position)

Directory Structure
-------------------

The checkpointing directory structure depends on the training backend used. 

FSDP checkpoints are organized according to the following directory hierarchy:

.. code-block::

    {ckpt_path}/
    ├── latest_ckpt_global_step.txt          # Holds the global step of the latest checkpoint
    ├── global_step_10/                      # Checkpoint at training step 10
    │   ├── policy/                          # Policy model checkpoint directory
    │   │   ├── fsdp_config.json             # stores fsdp version and world size
    │   │   ├── huggingface/                  # HuggingFace config and tokenizer
    │   │       ├── config.json                 # model config
    │   │       ├── tokenizer_config.json       # tokenizer config
    │   │       ├── generation_config.json      # generation config
    │   │       ├── ...                         # other tokenizer config files
    │   │   ├── model_state.pt               # Model parameters
    │   │   ├── optimizer_state.pt           # Optimizer state
    │   │   └── lr_scheduler_state.pt        # Learning rate scheduler state
    │   ├── critic/                          # Critic model checkpoint (if enabled)
    │   │   ├── fsdp_config.json             
    │   │   ├── huggingface/
    │   │   ├── model_state.pt
    │   │   ├── optimizer_state.pt
    │   │   └── lr_scheduler_state.pt
    │   ├── data.pt                          # Dataloader state
    │   └── trainer_state.pt                 # High-level trainer state
    ├── global_step_20/                      # Checkpoint at training step 20
    │   └── ...
    └── global_step_30/                      # Checkpoint at training step 30
        └── ...

DeepSpeed checkpoints follow a similar directory structure but the model checkpoint files under ``policy`` and ``critic`` are created by the DeepSpeed checkpoint API, and are not explicitly managed by SkyRL.

.. code-block::

    {ckpt_path}/
    ├── latest_ckpt_global_step.txt          # Holds the global step of the latest checkpoint
    ├── global_step_10/                      # Checkpoint at training step 10
    │   ├── policy/                          # Policy model checkpoint directory
    │   │   ├── huggingface/                 # HuggingFace config and tokenizer 
    │   │   ├── global_step10/               # Deepspeed checkpoint directory
    │   │   ├── ...                          # other deepspeed checkpointing files
    │   ├── critic/                          # Critic model checkpoint (if enabled)
    │   │   ├── huggingface/                 
    │   │   ├── global_step10/               
    │   │   ├── ...                          
    ├── global_step_20/                      # Checkpoint at training step 20
    │   └── ...
    └── global_step_30/                      # Checkpoint at training step 30
        └── ...


Key Configuration Parameters
----------------------------

Checkpointing behavior is controlled by several parameters in the YAML configuration (see :doc:`../configuration/config` for the full training config):

**Checkpoint Saving**

``ckpt_interval``
  - **Default**: ``10``
  - **Purpose**: Save checkpoints every N training steps

``ckpt_path``
  - **Default**: ``"${oc.env:HOME}/ckpts/"``
  - **Purpose**: Base directory where all checkpoints are stored

**Checkpoint Cleanup**

``max_ckpts_to_keep``
  - **Default**: ``-1`` (keep all checkpoints)
  - **Purpose**: Limit number of stored checkpoints to save disk space
  - **Options**:

    - ``-1``: Keep all checkpoints indefinitely
    - ``N`` (positive integer): Keep only the last N checkpoints, automatically delete older ones

**Training Resumption**

``resume_mode``
  - **Default**: ``"latest"``
  - **Purpose**: Controls how training resumption works
  - **Options**:
  
    - ``"none"`` or ``null``: Start training from scratch, ignore existing checkpoints
    - ``"latest"``: Automatically resume from the most recent checkpoint
    - ``"from_path"``: Resume from a specific checkpoint (requires ``resume_path``)

``resume_path``
  - **Default**: ``null``
  - **Purpose**: Specific checkpoint directory to resume from (only used when ``resume_mode: "from_path"``)
  - **Format**: Must point to a ``global_step_N`` directory

HuggingFace Model Export
------------------------

In addition to checkpointing, users can optionally save the policy model in HuggingFace safetensors format at regular intervals.

**Configuration Parameters:**

``hf_save_interval``
  - **Default**: ``-1`` (disabled)
  - **Purpose**: Save HuggingFace format policy models every N training steps

``export_path``
  - **Default**: ``"${oc.env:HOME}/exports/"``
  - **Purpose**: Base directory where HuggingFace models and other artifacts are saved
  - **Structure**: Models are saved to ``{export_path}/global_step_{N}/policy/``
