Dataset Preparation
===================

This guide covers:

1. The dataset format that SkyRL expects for training, and
2. How to prepare and format a new dataset


Format Requirements
-------------------

Each dataset entry is a dictionary with the following required (and some optional) values:

.. code-block:: python

   data = {
       "data_source": data_source,     # String: Name/identifier of the data source
       "prompt": [                     # List: Conversation format
           {
               "role": "user",            
               "content": question,       
           }
       ],
       "env_class": env_class,         # String: Environment class identifier
       "reward_spec": {
           "method": "rule",           # String: Either "rule" or "reward_model"
           "ground_truth": solution,   # Expected solution
       },
       "extra_info": {                 # Dict: Optional additional metadata
           # ... add your own fields here
       },
   }

We load the dataset as a huggingface `DatasetDict <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict>`_.

**Key Requirements:**

- **data_source**: String identifier for the dataset origin (e.g., "gsm8k", "AIME24", etc.)
- **prompt**: List of dictionaries following standard OpenAI chat format
- **env_class**: Name of environment that the data sample corresponds to. This is used to tell the Generator which environment to instantiate for this prompt.

  - Note: **env_class** can also be specified in the training configuration to apply to all dataset entries.
- **reward_spec**: Dictionary containing the reward specification for the dataset entry (ie, how to get rewards).

  - **method**: Must be either ``"rule"`` or ``"reward_model"``
  - **ground_truth**: If ``method`` is ``"rule"``, this is the expected solution.

- **extra_info**: Extensible dictionary for additional metadata - you can add custom fields as needed.


Data Preparation Scripts
------------------------

We provide several example scripts to help you prepare your dataset, including for gsm8k, LiveCodeBench, SearchR1, and the SynSQL text-to-SQL dataset. 

**To use a new dataset for training, you can use the provided scripts as a template to create your own.**

Generally, only a single method (`make_map_fn`) must be implemented to convert the new dataset into the required format. Below is an example of converting the SynSQL text-to-SQL dataset into the required format:

.. code-block:: python

  def make_map_fn(split):
        def process_fn(example, idx):
            """Transform each dataset example into the required format"""
            if split == "train":
                user_content = ("{db_details}:" + example["schema"] + 
                              ";\n {external_knowledge}: " + example["external_knowledge"] + 
                              ";\n {question}: " + example["question"])
            else:
                user_content = ("{db_details}:" + example["schema"] + 
                              "; {question}: " + example["question"])
            
            data = {
                "data_source": "synsql",
                "prompt": [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "env_class": "text2sql",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": example["sql"],
                },
                # Custom fields specific to the SynSQL dataset:
                "db_id": example["db_id"],
                "data": example["data"],
            }
            return data
        
        return process_fn

Then, the mapping function is called on each sample in the dataset, and the final converted dataset is saved to a parquet file:

.. code-block:: python

  train_dataset = input_dataset.map(function=make_map_fn("train"), with_indices=True)
  train_dataset.to_parquet(os.path.join(args.output, "train.parquet"))


Reference Scripts
-----------------

Use the following scripts as a template to prepare your dataset:

- `gsm8k_dataset.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/gsm8k/gsm8k_dataset.py>`_
- `multiply_dataset.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/multiply/multiply_dataset.py>`_
