Data Interface
==============


Trainer APIs
------------
Our interface for training data is modelled after `DataProto <https://verl.readthedocs.io/en/latest/api/data.html#>`_ in VERL but is much simpler.

.. autoclass:: skyrl_train.training_batch.TensorBatch
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.training_batch.TrainingInput
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.training_batch.TrainingInputBatch
   :members:
   :member-order: bysource
   :undoc-members:

.. autoclass:: skyrl_train.training_batch.TrainingOutputBatch
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:


Generator APIs
--------------

.. autoclass:: skyrl_train.generators.GeneratorInput
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.generators.GeneratorOutput
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance: