Using Tools in SkyRL-Gym
==========================

One of the key features of SkyRL-Gym is `reusable tools`. Often, a single tool (such as Python code execution) is useful in multiple environments (such as math or deep research environments). Reusable tools allow developers to implement a tool once, and use it in multiple environments. You can even build an environment from our library of implemented tools!

This document describes SkyRL-gym's tool interface and how to build and use tools in SkyRL-gym environments. You can also find the source code for tools in `skyrl_gym/tools <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym/skyrl_gym/tools>`_.

Core Concepts
-------------

**tool**: A ``tool`` is a single executable function whose inputs and outputs can be flexibly defined.

**ToolGroup**: A ``ToolGroup`` is a collection of related tools that share the same context or states. Tool groups enable all tools within the group to access and modify the shared state, such as a shared database connection or cache.

**Environment**: An ``Environment`` is a class that defines the task for the agent to solve, and can integrate one ore more tool groups for the agent to use. See the following doc for more details on how to build an environment: :doc:`new_env`.


ToolGroup and the @tool Decorator
-----------------------------------

The ``ToolGroup`` class provides utilities for managing tools and tool execution, such as ``get_tool`` and ``get_tool_names``.  To create a new tool group, you can inherit from ``ToolGroup`` and add or modify utilities as needed. For example, you can add a shared state to the tool group, or add a custom tool execution function.

The `@tool` decorator marks methods as executable tools. After creating a custom tool group, you can use the `@tool` decorator to mark methods as tools, which the ``ToolGroup`` will automatically register and be able to execute. For instance, a simple Python code execution tool group might look like the following code block, where the tool group is called ``PythonCodeExecutorToolGroup`` and the tool is called ``python`` marked with the ``@tool`` decorator.

.. code-block:: python
    
    from skyrl_gym.tools.core import tool, ToolGroup
    import subprocess


    # Create a new tool group that inherits from ToolGroup
    class PythonCodeExecutorToolGroup(ToolGroup):
        def __init__(self):
            super().__init__(name="PythonCodeExecutorToolGroup")

        # Mark the python method as a tool
        @tool
        def python(self, code: str) -> str:
            result = subprocess.run(
                ["python", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, text=True
            )
            return result.stdout


See the base interface for tools and tool groups at `skyrl_gym/tools/core.py <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-gym/skyrl_gym/tools/core.py>`_.

Built-in ToolGroups
-------------------

SkyRL-Gym provides pre-built ToolGroups that can be used out of the box in any SkyRL-gym environment:

Python Code Execution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import PythonCodeExecutorToolGroup

    python_tools = PythonCodeExecutorToolGroup(timeout=15.0)
    result = python_tools.execute_tool("python", "print('Hello, World!')")

SQL Code Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import SQLCodeExecutorToolGroup

    sql_tools = SQLCodeExecutorToolGroup(db_file_path="/path/to/databases")
    result = sql_tools.execute_tool("sql", "SELECT * FROM users")

Search ToolGroup
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import SearchToolGroup

    search_tools = SearchToolGroup(
        search_url="http://127.0.0.1:8000/retrieve"
    )
    result = search_tools.execute_tool("search", "Context to search")

Environment Integration
------------------------

Tools groups can be integrated into any environment in SkyGym-RL. The base environment class for text-based environments is ``BaseTextEnv``, which provides simple utilities for managing and using multiple tool groups in a single envrionment.

The following sub-sections walk through integrating and using tools in an environment.

Tool Initialization
~~~~~~~~~~~~~~~~~~~

To incorporate tools into an envrionment, first build and initialize the tool groups during environment construction:


.. code-block:: python

    from skygym.envs.base_text_env import BaseTextEnv

    class MyEnvironment(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            
            # Construct the tool groups
            python_tools = PythonCodeExecutorToolGroup(timeout=10.0)
            search_tools = SearchToolGroup()
            
            # Initialize and register tool groups
            self.init_tool_groups([python_tools, search_tools])

Tool Execution
~~~~~~~~~~~~~~

To use a tool and get the result, you can call the ``_execute_tool`` (provided by ``BaseTextEnv``) method with the tool group name, tool name, and the tool input. Tools are most often used in the envrionment ``step`` method.

.. code-block:: python

    def step(self, action: str):
        # Parse action to extract tool call
        tool_group_name, tool_name, tool_input = self._parse_action(action)
        
        # Execute the tool
        observation = self._execute_tool(tool_group_name, tool_name, tool_input)
        
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": observation}],
            reward=reward,
            done=done,
            metadata=info
        )

Action Parsing
~~~~~~~~~~~~~~

Users can flexibly determine how and when tools are called. The following code block shows a common case, where the model's output (``action``) is parsed to extract the intended tool call and its input.

.. code-block:: python

    import re

    def _parse_action(self, action: str):
        # Parse tool blocks like <tool><tool_name>input</tool_name></tool>
        tool_block_match = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not tool_block_match:
            raise ValueError("No tool block found in action")
        
        tool_content = tool_block_match.group(1).strip()
        inner_tag_match = re.search(r"<(\w+)>(.*?)</\1>", tool_content, re.DOTALL)
        
        # Extract the tool name
        tool_name = inner_tag_match.group(1)
        # Extract the tool input
        tool_input = inner_tag_match.group(2).strip()
        
        tool_group_name = self.tool_to_toolgroup[tool_name]
        
        return tool_group_name, tool_name, [tool_input]

Using Multiple ToolGroups
-------------------------

An arbitrary number of tool groups can be integrated into an environment, creating a more powerful agent. There is no additional configuration required to use multiple tool groups in an environment, simply construct and initialize all desired tool groups.

.. code-block:: python

    class AdvancedEnvironment(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            
            # Different ToolGroups with shared state
            self.db_tools = SQLCodeExecutorToolGroup(db_file_path="/path/to/databases")
            self.python_tools = PythonCodeExecutorToolGroup(timeout=10.0)
            self.search_tools = SearchToolGroup(search_url="http://127.0.0.1:8000/retrieve")
            self.custom_tools = MyCustomToolGroup(shared_config=extras.get("config"))
            
            # Register all tool groups
            self.init_tool_groups([self.db_tools, self.python_tools, self.search_tools, self.custom_tools])


API Reference
-------------

For a reference of the tool APIs, see the following links:

- :doc:`../api/tools`
- :doc:`../api/env`

That's it! You've learned how to use tools in SkyRL-Gym environments. The same pattern works for any tool-based task you want to build. 