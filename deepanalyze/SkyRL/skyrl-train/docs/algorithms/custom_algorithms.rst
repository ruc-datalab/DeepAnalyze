Implementing Custom Algorithms
==============================

SkyRL-Train provides a registry system for easily implementing custom algorithms (advantage estimators, policy loss) without modifying the core codebase. 
The API for the registry system can be found in the :doc:`registry API <../api/registry>`.
Example scripts of using the registry can be found in at :code_link:`examples/algorithm/`.

Additionally for more control, you can subclass the ``BasePPOExp`` class from :code_link:`skyrl_train/entrypoints/main_base.py` and override the ``BasePPOExp.get_trainer`` method to return a custom trainer class.
This allows you to have full control over the training loop and implementing custom reward functions and output postprocessing.
We provide an example of this for applying custom reward penalties in our :ref:`DAPO example <dapo-custom-trainer>`.

Registering a Custom Advantage Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can register custom advantage estimators using either a decorator or the registry directly:

.. code-block:: python

   from skyrl_train.utils.ppo_utils import register_advantage_estimator, AdvantageEstimatorRegistry
   import torch

   # Using the decorator
   @register_advantage_estimator("simple_baseline")
   def compute_simple_baseline_advantage(
        token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, **kwargs
    ):
        with torch.no_grad():
            response_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)

            # Simple baseline: use the mean reward across the batch
            baseline = response_rewards.mean()
            advantages = (response_rewards - baseline) * response_mask
            returns = advantages.clone()

            return advantages, returns

   # Or register directly
   def another_estimator(**kwargs):
       # Implementation here
       pass

   AdvantageEstimatorRegistry.register("direct_registration", another_estimator)

Registering a Custom Policy Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, you can register custom policy loss functions:

.. code-block:: python

   from skyrl_train.utils.ppo_utils import register_policy_loss, PolicyLossRegistry

   @register_policy_loss("reinforce")
   def compute_reinforce_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_log_probs=None):
       # Your custom policy loss implementation (like REINFORCE)
       loss = (-log_probs * advantages).mean()
       # return loss and clip ratio
       return loss, 0.0

Registry Ray Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The registry system handles Ray actor synchronization when Ray is initialized. Functions registered on one process will be available to all Ray actors:

.. code-block:: python

   import ray
   from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry, sync_registries

   # Register a function on the main process
   def my_function(**kwargs):
       # A dummy function for demonstration
       pass
   AdvantageEstimatorRegistry.register("my_function", my_function)

   # After Ray is initialized, we sync the registries to a named ray actor (in utils/utils.py::initialize_ray)
   ray.init()
   sync_registries()
   
   @ray.remote(num_cpus=1)
   def skyrl_entrypoint(cfg: DictConfig):
        # Function is now available on all Ray processes
        available_functions = AdvantageEstimatorRegistry.list_available() # will include "my_function"

        exp = BasePPOExp(cfg)
        exp.run()

Creating a Custom Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom trainer for full control of your training loop, you can subclass the ``BasePPOExp`` class from :code_link:`skyrl_train/entrypoints/main_base.py` and override the ``BasePPOExp.get_trainer`` method to return a custom trainer class.
We show the outline of creating a custom trainer below, and you can find a full running example in our :ref:`DAPO example <dapo-custom-trainer>`.

.. code-block:: python

    class CustomTrainer(RayPPOTrainer):
        @torch.no_grad()
        def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
            # apply custom reward penalties
            ...
            # use base class impl for metrics and per-token reward conversion
            return super().postprocess_generator_output(generator_output, uids)

   class CustomExp(BasePPOExp):
       def get_trainer(self, *args, **kwargs):
           return CustomTrainer(*args, **kwargs)

    @ray.remote(num_cpus=1)
    def skyrl_entrypoint(cfg: DictConfig):
        exp = CustomExp(cfg)
        exp.run()

    @hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
    def main(cfg: DictConfig) -> None:
        # validate the arguments
        validate_cfg(cfg)

        initialize_ray(cfg)
        ray.get(skyrl_entrypoint.remote(cfg))

    if __name__ == "__main__":
        main()