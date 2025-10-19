Creating a New Environment or Task
=====================================

To demonstrate how to create custom environments in SkyRL-Gym and train with SkyRL, let's build a simple multiplication environment!

We'll walk through the complete process: implementing the environment, registering it, preparing training data, and running your first training session.

**What we're building:** An environment that asks the model to multiply numbers and checks if the answer is correct. The completed code is available at `examples/multiply <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply>`_.

Environment Interface
---------------------

SkyRL-Gym includes a simple text-in/text-out environment interface for LLM tasks, ``BaseTextEnv``, which looks like this:

.. code-block:: python
   :linenos:
   :caption: Base text environment interface at `skyrl_gym/envs/base_text_env.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-gym/skyrl_gym/envs/base_text_env.py>`_

   class BaseTextEnv(Env[str, str]):
      def step(self, action: str) -> BaseTextEnvStepOutput:
         """
         Runs one environment step.

         Args:
            action: The LLM's response as a string

         Returns:
            BaseTextEnvStepOutput containing:
            - observations: New messages from the environment
            - reward: Float reward for the action  
            - done: Whether the episode is finished
            - metadata: Additional info (optional)
         """
         pass

      def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
         return prompt, {}

      def close(self):
         pass

This class inherits from ``Env``, which is a generic environment interface (i.e., not specific to text-based tasks). An API reference for ``Env`` can be found in :doc:`../api/env`.

For our multiplication environment, we only need to implement the ``step`` method above because we don't have any initialization or cleanup to do.


Simple Single-Turn Environment
-------------------------------

Let's start with a basic version that gives the model only one chance to get the answer right. 

The prompt and response format expected by the ``multiply`` environment is as follows:

- The model prompt will be a multiplication problem of 2 n-digit numbers, such as "123 * 456" or "999 * 999". 
- The model output should be in the format of ``\\boxed{answer}``, where ``answer`` is the product of the two numbers. 

So, the environment ``step`` must simply parse the answer out of ``\\boxed{answer}`` and check if it matches the ground truth.

.. code-block:: python
   :linenos:
   :caption: Simple multiplication environment

   class MultiplyEnv(BaseTextEnv):
      def _parse_action(self, action: str) -> str:
         """Extract answer from \\boxed{answer} format"""
         match = re.search(r"\\boxed\{([^}]+)\}", action)
         return match.group(1) if match else None
         
      def step(self, action: str) -> BaseTextEnvStepOutput:
         answer = self._parse_action(action)
         is_correct = answer is not None and answer.strip() == str(self.ground_truth).strip()

         return BaseTextEnvStepOutput(
            observations=[],
            reward=1.0 if is_correct else 0.0,
            done=True,
            metadata={"parsed_answer": answer}
         )

That's it! The environment checks if the model's answer matches the ground truth and gives a reward of 1.0 for correct answers, 0.0 for incorrect ones.

Multi-Turn Environment
----------------------

Want to give the model multiple attempts? Let's extend our environment to allow multiple turns.

We will make a few simple extensions to our ``step()`` method:

- Keep track of the number of turns (``self.turns``) and indicate the trajectory is ``done`` after a configured maximum number of turns (``self.max_turns``)
- If the turns expire or the model provides a correct answer, we indicate the trajectory is ``done`` and return a reward as follows:

  - Correct answer: 1.0.
  - Incorrect answer, but in format of ``\\boxed{...}``: 0.5.
  - Incorrect answer, and not in format of ``\\boxed{...}``: 0.0.
- If the model is incorrect and has more turns remaining, we also provide feedback as a new ``observation``.

.. code-block:: python
   :linenos:
   :caption: Multi-turn multiplication environment in `examples/multiply/env.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/env.py>`_

   def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        answer = self._parse_action(action)
        is_correct = answer is not None and answer.strip() == str(self.ground_truth).strip()
        found_boxed = answer is not None

        # Episode ends if max turns reached or correct answer found
        done = self.turns >= self.max_turns or is_correct
        
        # Reward structure:
        # - Correct answer: 1.0
        # - Wrong answer in correct format: 0.5  
        # - No boxed answer: 0.0
        if is_correct:
            reward = 1.0
        elif found_boxed:
            reward = 0.5
        else:
            reward = 0.0

        if done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"parsed_answer": answer}
            )
            
        # Give feedback for another attempt
        if answer is not None:
            feedback = f"Your answer '{answer}' is incorrect. Please try again."
        else:
            feedback = "Please provide your answer in the format \\boxed{your_answer}."
            
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=0.0,
            done=False,
            metadata={"parsed_answer": answer}
        )

The multi-turn version gives partial credit for formatting the answer correctly, even if it's wrong. This helps the model learn the expected output format.

The final implementation is available in `examples/multiply/env.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/env.py>`_.

Registering Your New Environment
--------------------------------

Finally, we need to ``register`` the new environment so the training stack can find it by name (which we refer to as ``env_class``). We will name this environment ``multiply``.

We will create a new entrypoint for training with the ``multiply`` environment by creating a file at ``examples/multiply/main_multiply.py`` that looks like this:

.. code-block:: python
   :linenos:
   :caption: Environment registration at `examples/multiply/main_multiply.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/main_multiply.py>`_

   @ray.remote(num_cpus=1)
   def skyrl_entrypoint(cfg: DictConfig):
      # Register the multiply environment
      # this needs to be done inside the entrypoint task
      register(
         id="multiply",  # <-- The name of the environment.
         entry_point="examples.multiply.env:MultiplyEnv",  # <-- The path to the environment class.
      )

      # make sure that the training loop is not run on the head node.
      exp = BasePPOExp(cfg)
      exp.run()

   @hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
   def main(cfg: DictConfig) -> None:
      # validate the arguments
      validate_cfg(cfg)

      initialize_ray(cfg)
      ray.get(skyrl_entrypoint.remote(cfg))

   if __name__ == "__main__":
      main()

Now, the training stack can simply build the new environment with ``skyrl_gym.make("multiply")``!

.. note::
   All example code written in this document is *outside* of the ``skyrl-train`` and ``skyrl-gym`` packages. There is no need to fork and edit ``skyrl-train`` or ``skyrl-gym`` code -- just implement and register your environment, and the training stack can find the environment seamlessly!

Preparing Training Data
-----------------------

Before we can train, we need a dataset of problems to train on.

We can generate a dataset of multiplication problems using `examples/multiply/multiply_dataset.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/multiply_dataset.py>`_. See the file for more details, but the core idea is to generate random multiplication problems of n-digit numbers, and ensure the dataset example is in the correct format:

.. code-block:: python
   :linenos:
   :caption: Generating a dataset of random multiplication problems.

   for idx in range(num_examples):
        question, answer = generate_multiplication_problem(num_digits)
        
        data = {
            "data_source": "synthetic_multiply",
            "prompt": [
                system_prompt,
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": "multiply",
            "reward_spec": {
                "method": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "num_digits": num_digits,
                "split": split_name,
            },
        }
        examples.append(data)


Note that the ``env_class`` here should match the name of the environment we registered. In this case, it is ``multiply``. You can optionally omit the ``env_class`` here and instead set it in the training configuration to apply to all training samples, but setting ``env_class`` per-sample allows for multi-environment training so it is the recommended practice.

See the doc on :doc:`../datasets/dataset-preparation` for more details on the required dataset format and how to prepare your own dataset.

Now we can generate the datsaet:

.. code-block:: bash
   :linenos:
   :caption: Generate training data

   uv run --isolated examples/multiply/multiply_dataset.py \
     --output_dir $HOME/data/multiply \
     --num_digits 4 \
     --train_size 10000 \
     --test_size 200

This creates ``train.parquet`` and ``validation.parquet`` files in the ``$HOME/data/multiply`` directory.

Training Your Model
-------------------

Time to train! 🚀

We will use the ``run_multiply.sh`` script to train the model. This script is located in `examples/multiply/run_multiply.sh <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/run_multiply.sh>`_, which sets up the training configuration and calls ``main_multiply.py``.

**Common Configuration Parameters**

First, ensure sure your config matches your available GPUs. You may need to adjust the following parameters to match your GPU count (which we set via an environment variable `NUM_GPUS`):

- ``trainer.placement.policy_num_gpus_per_node``
- ``generator.num_inference_engines``

Then, configure how the environment should be executed. For multi-turn environments, we recommend setting ``generator.batched=false`` and ``generator.async_engine=true`` to ensure that each environment is executed asynchronously. If your environment is single-turn, you may get better performance by reversing these settings.

**Launch Training**

.. code-block:: bash
   :linenos:
   :caption: Run training

   export WANDB_API_KEY=your_wandb_api_key  # or set trainer.logger="console" to print to stdout
   bash examples/multiply/run_multiply.sh

**Next Steps:** Want to make multiplication easier? Try integrating a calculator tool into your environment! Check out the :doc:`tools_guide` documentation for details.

That's it! You've created a custom environment, prepared training data, and started training. The same pattern works for any text-based task you want to train on.

Now watch your model become a multiplication master!