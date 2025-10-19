Async Training: One-off Pipelining
=========================================

This example demonstrates how to implement asynchronous training with one-off pipelining, showcasing the flexibility of the training framework to support different execution plans with minimal code changes.

The complete code from this example is available at `examples/async <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/async>`_.

Overview
--------

The one-off pipelining approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

.. TODO(tgriggs): Add a diagram here.

In the SkyRL framework, only minimal code changes are required to modify the synchronous training pipeline to an asynchronous one.

In particular, we follow three simple steps:

1. Define a new trainer class (``AsyncRayPPOTrainer``) and override a single method (``train()``) from the the base trainer class to separate and parallelize the training and generation phases
2. Create a new training entrypoint (``main_async.py``) that uses the new trainer class
3. Update the training configuration to disable model colocation

Implementation Steps
--------------------

Step 1: New trainer class with modified training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first create a new trainer class ``AsyncRayPPOTrainer`` that inherits from ``RayPPOTrainer`` and overrides the ``train()`` method. 

The original ``train()`` method performs generation and training sequentially, so the async trainer splits the traditional synchronous training loop into two parallel components:

1. **Generation Loop** (``_run_generate_loop``): Continuously generates new samples and places them in a queue
2. **Training Loop** (``_run_training``): Consumes generated samples from the queue and performs training updates

The generation loop passes the training loop completed trajectories via an ``asyncio.Queue`` and coordinates weight synchronization using asyncio events.

We include a minimal version here, please see `examples/async/async_trainer.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/async/async_trainer.py>`_ for full details:

.. code-block:: python

   class AsyncRayPPOTrainer(RayPPOTrainer):
       async def train(self):
           # Assert non-colocated training
           assert not self.cfg.trainer.placement.colocate_all, "colocate_all is not supported for async training"
           
           # Create buffer of size 1 for generated trajectories.
           generation_buffer = asyncio.Queue(maxsize=1)
           
           # Start generator task asynchronously.
           generator_task = asyncio.create_task(self._run_generate_loop(generation_buffer))
           
           # Training loop consumes from buffer
           for idx in range(len(self.train_dataloader)):
               # Training starts when generation buffer is filled.
               status = await self._run_training(generation_buffer)
                # Trainer waits for generation to complete before next training step.
                if idx != len(self.train_dataloader) - 1:
                    await self.generation_ack.wait()
                # Synchronize weights after training.
                await self.weights_manager.async_sync_policy_weights_to_inference_engines()
                # Signal that weight sync is done, ready for next round of generation.
                self.sync_finished.set()
                self.generation_ack.clear()

.. note::
   **Flexible execution plans:** The underlying implementation of training and generation is unchanged -- only the training loop is modified to parallelize the two phases. In general, SkyRL enables flexibile modifications to the execution plan simply by modifying the high-level training loop.

Step 2: Create New Training Entrypoint  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we create a new training entrypoint ``main_async.py`` that uses the new trainer class.

The new entrypoint (``AsyncPPOExp``) overrides a single method (``get_trainer()``) in the base entrypoint class (``BasePPOExp``) that constructs the new trainer class:

.. code-block:: python

   class AsyncPPOExp(BasePPOExp):
       def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, 
                      inference_engine_client, generator, colocate_pg):
           # Construct the new trainer class.
           return AsyncRayPPOTrainer(
               cfg=cfg, tracker=tracker, tokenizer=tokenizer,
               train_dataset=train_dataset, eval_dataset=eval_dataset,
               inference_engine_client=inference_engine_client, generator=generator,
               colocate_pg=colocate_pg
           )

That's it! The rest of the entrypoint logic for launching the training run remains unchanged.

Step 3: Update Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we modify the training configuration to use our new entrypoint and disable model colocation:

.. code-block:: bash

  uv run --isolated --extra vllm -m examples.async.main_async \
    trainer.placement.colocate_all=false \
    trainer.placement.colocate_policy_ref=true \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.num_inference_engines=4 \
    generator.inference_engine_tensor_parallel_size=1

Key configuration changes:

* **examples.async.main_async**: Point the bash script to the new entrypoint
* **colocate_all=false, colocate_policy_ref=true**: Disables colocation of generation and training models (but keeps the policy and reference models colocated).


Now we can train!

.. code-block:: bash

   # Prepare the dataset
   uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

    # Run the training script
   export WANDB_API_KEY=your_wandb_api_key  # or set trainer.logger="console" to print to stdout
   bash examples/async/async_run_gsm8k.sh

