Trainer API
===========

The Trainer drives the training loop. 


Trainer Class
-------------

.. autoclass:: skyrl_train.trainer.RayPPOTrainer
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:


Dispatch APIs
-------------

.. autoclass:: skyrl_train.distributed.dispatch.Dispatch
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.distributed.dispatch.MeshDispatch
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.distributed.dispatch.PassThroughDispatch
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:


Actor APIs
-----------

The base worker abstraction in SkyRL is the 

.. autoclass:: skyrl_train.workers.worker.Worker
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.workers.worker.PPORayActorGroup
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:


