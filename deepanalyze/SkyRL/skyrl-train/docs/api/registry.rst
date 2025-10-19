Algorithm Registry API
=====================================

The registry system in SkyRL Train provides a way to register and manage custom algorithm functions (like advantage estimators and policy loss functions) across distributed Ray environments. This system allows users to extend the framework with custom implementations without modifying the core codebase.

Base Registry Classes
---------------------

.. autoclass:: skyrl_train.utils.ppo_utils.BaseFunctionRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.ppo_utils.RegistryActor
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.ppo_utils.sync_registries
    
Advantage Estimator Registry
-----------------------------

The advantage estimator registry manages functions that compute advantages and returns for reinforcement learning algorithms.

.. autoclass:: skyrl_train.utils.ppo_utils.AdvantageEstimatorRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.ppo_utils.AdvantageEstimator
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.ppo_utils.register_advantage_estimator


Policy Loss Registry
--------------------

The policy loss registry manages functions that compute policy losses for PPO and related algorithms.

.. autoclass:: skyrl_train.utils.ppo_utils.PolicyLossRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.ppo_utils.PolicyLossType
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.ppo_utils.register_policy_loss
