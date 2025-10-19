import pandas as pd
import numpy as np
from typing import Any
import os
import json
import tempfile
import time
import re

from openhands.core.main import create_runtime, run_controller
from openhands.utils.shutdown_listener import sleep_if_should_continue
from openhands.core.logger import openhands_logger as logger
from openhands.core.config import SandboxConfig, AppConfig
from openhands.events.action import CmdRunAction, MessageAction, FileReadAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation, FileReadObservation
from openhands.runtime.base import Runtime
from openhands.core.exceptions import (
    AgentRuntimeBuildError,
    AgentRuntimeDisconnectedError,
    AgentRuntimeError,
    AgentRuntimeNotFoundError,
    AgentRuntimeNotReadyError,
    AgentRuntimeTimeoutError,
    AgentRuntimeUnavailableError,
)
from skyagent.tasks.base import BaseTask
from skyagent.dispatcher.async_utils import call_sync_from_async, call_async_from_sync

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

def ensure_serializable(obj):
    """Recursively convert non-serializable objects to JSON serializable formats."""
    if isinstance(obj, np.ndarray):  # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):  # Convert numpy int to Python int
        return int(obj)
    elif isinstance(obj, (np.floating, float)):  # Convert numpy float to Python float
        return float(obj)
    elif isinstance(obj, dict):  # Recursively process dictionaries
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # Recursively process lists
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, tuple):  # Convert tuples to lists
        return tuple(ensure_serializable(item) for item in obj)
    return obj  # Return as is if already serializable

class EvalException(Exception):
    pass

def assert_and_raise(condition: bool, msg: str):
    """Raise an EvalException if the condition is not met.

    This will be used in conjunction with _process_instance_wrapper to handle retries. An EvalException should trigger a retry.
    """
    if not condition:
        raise EvalException(msg)

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

def _get_swebench_workspace_dir_name(instance: pd.Series, dataset: str) -> str:
    if "r2e-gym" in dataset:
        return "/testbed"
    return '/workspace/' + f'{instance.repo}__{getattr(instance, "version", "null")}'.replace('/', '__')

# def get_instruction(instance: pd.Series):
#     workspace_dir_name = _get_swebench_workspace_dir_name(instance)
#     instruction = f"""
# <uploaded_files>
# {workspace_dir_name}
# </uploaded_files>

# I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following issue description:

# <issue_description>
# {instance.problem_statement}
# </issue_description>

# Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
# I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
# Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
# Your task is to make the minimal changes to non-test files in the {workspace_dir_name} directory to ensure the <issue_description> is satisfied.

# Follow these phases to resolve the issue:

# Phase 1. READING: read the problem and reword it in clearer terms
#    1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
#    1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
#    1.3 Explain the problem in clear terms.
#    1.4 Enumerate the steps to reproduce the problem.
#    1.5 Hightlight any best practices to take into account when testing and fixing the issue

# Phase 2. RUNNING: install and run the tests on the repository
#    2.1 Follow the readme
#    2.2 Install the environment and anything needed
#    2.2 Iterate and figure out how to run the tests

# Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
#    3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
#    3.2 Identify all files related to the problem statement.
#    3.3 Propose the methods and files to fix the issue and explain why.
#    3.4 From the possible file locations, select the most likely location to fix the issue.

# Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
#    4.1 Look at existing test files in the repository to understand the test format/structure.
#    4.2 Create a minimal reproduction script that reproduces the located issue.
#    4.3 Run the reproduction script to confirm you are reproducing the issue.
#    4.4 Adjust the reproduction script as necessary.

# Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
#    5.1 State clearly what the problem is.
#    5.2 State clearly where the problem is located.
#    5.3 State clearly how the test reproduces the issue.
#    5.4 State clearly the best practices to take into account in the fix.
#    5.5 State clearly how to fix the problem.

# Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
#    6.1 Make minimal, focused changes to fix the issue.

# Phase 7. VERIFICATION: Test your implementation thoroughly.
#    7.1 Run your reproduction script to verify the fix works.
#    7.2 Add edge cases to your test script to ensure comprehensive coverage.
#    7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

# 8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {instance['base_commit']}.
#    8.1 Ensure you've fully addressed all requirements.
#    8.2 Run any tests in the repository related to:
#      8.2.1 The issue you are fixing
#      8.2.2 The files you modified
#      8.2.3 The functions you changed
#    8.3 If any tests fail, revise your implementation until all tests pass

# Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
# """

#     if RUN_WITH_BROWSING:
#         instruction += (
#             '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
#         )

#     if 'image_assets' in instance:
#         assets = json.loads(instance['image_assets'])
#         assert 'problem_statement' in assets, (
#             'problem_statement is required in image_assets'
#         )
#         image_urls = assets['problem_statement']
#         return MessageAction(content=instruction, image_urls=image_urls)
#     return MessageAction(content=instruction)


def initialize_runtime(runtime: Runtime, instance: pd.Series, dataset: str):
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)
    obs: CmdOutputObservation

    # Set instance id and git configuration
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export SWE_INSTANCE_ID and configure git: {str(obs)}',
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    # inject the init script
    script_dir = os.path.dirname(__file__)

    # inject the instance info
    action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
    )

    swe_instance_json_name = 'swe-bench-instance.json'
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the desired file name within the temporary directory
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        # Write to the file with the desired name within the temporary directory
        with open(temp_file_path, 'w') as f:
            if not isinstance(instance, dict):
                json.dump([instance.to_dict()], f)
            else:
                json.dump([instance], f)

        # Copy the file to the desired location
        runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        if "r2e-gym" in dataset:
            runtime.copy_to(
                str(os.path.join(script_dir, 'scripts/setup/instance_r2e_entry.sh')),
                '/swe_util/',
            )
        else:
            runtime.copy_to(
                str(os.path.join(script_dir, 'scripts/setup/instance_swe_entry.sh')),
                '/swe_util/',
            )

    action = CmdRunAction(command='cat ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

    action = CmdRunAction(command='source ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if isinstance(obs, ErrorObservation):
        logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
    assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

    if "r2e-gym" in dataset:
        action = CmdRunAction(command='source /swe_util/instance_r2e_entry.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/instance_r2e_entry.sh: {str(obs)}',
        )
    else:
        action = CmdRunAction(command='source /swe_util/instance_swe_entry.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/instance_swe_entry.sh: {str(obs)}',
        )

    action = CmdRunAction(command=f'cd {workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to {workspace_dir_name}: {str(obs)}',
    )

    # git fetch and checkout needed by swe-smith
    if dataset == 'swe-smith':
        action = CmdRunAction(command='git fetch')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to git fetch: {str(obs)}')

        action = CmdRunAction(command=f'git checkout {instance["instance_id"]}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to git checkout {instance["instance_id"]}: {str(obs)}',
        )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    if dataset == 'swt-ci':
        # set up repo
        setup_commands = []
        if instance['repo'] in MAP_REPO_TO_INSTALL:
            setup_commands.append(MAP_REPO_TO_INSTALL[instance['repo']])

        # Run pre-install set up if provided
        install = MAP_VERSION_TO_INSTALL.get(instance['repo'], {}).get(
            instance['version'], []
        )
        if 'pre_install' in install:
            for pre_install in install['pre_install']:
                setup_commands.append(pre_install)

        if 'install' in install:
            setup_commands.append(install['install'])

        for command in setup_commands:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if 'multimodal' not in dataset:
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets are not using the testbed environment
        action = CmdRunAction(command='which python')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0 and 'testbed' in obs.metadata.py_interpreter_path,
            f'Expected to find python interpreter from testbed, but got: {str(obs)}',
        )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)

class SWEBenchTask(BaseTask):
    @classmethod
    def get_instruction(cls, instance: pd.Series, dataset: str):
        workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)
        instruction = f"""
    <uploaded_files>
    {workspace_dir_name}
    </uploaded_files>

    I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following issue description:

    <issue_description>
    {instance.problem_statement}
    </issue_description>

    Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
    I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
    Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
    Your task is to make the minimal changes to non-test files in the {workspace_dir_name} directory to ensure the <issue_description> is satisfied.

    Follow these steps to resolve the issue:

    1. EXPLORATION: First, thoroughly explore the repository structure using tools like `find` and `grep`.
    - Identify all files mentioned in the problem statement
    - Locate where the issue occurs in the codebase
    - Understand the surrounding context and dependencies
    - Use `grep` to search for relevant functions, classes, or error messages

    2. Assess whether you can reproduce the issue:
        - Create a script at {workspace_dir_name}/reproduce_issue.py that demonstrates the error.
        - Execute this script to confirm the error behavior.
        - You should reproduce the issue before fixing it.
        - Your reproduction script should also assert the expected behavior for the fixed code. 

    3. IMPLEMENTATION: Edit the source code to implement your chosen solution.
    - Make minimal, focused changes to fix the issue

    4. VERIFICATION: Test your implementation thoroughly.
    - Run your reproduction script to verify the fix works
    - Add edge cases to your test script to ensure comprehensive coverage
    - Run existing tests related to the modified code to ensure you haven't broken anything

    5. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {instance["base_commit"]}.
    - Ensure you've fully addressed all requirements

    Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
    """

        if RUN_WITH_BROWSING:
            instruction += """
    <IMPORTANT!>
    You SHOULD NEVER attempt to browse the web.
    </IMPORTANT!>
    """
        return MessageAction(content=instruction)
    
    @classmethod
    def get_config(cls, instance, data_source, agent_config=None, max_iterations=None) -> AppConfig:
        # Configure sandbox
        RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
        SWE_BENCH_CONTAINER_IMAGE = 'ghcr.io/opendevin/eval-swe-bench:full-v1.2.1'
        
        if os.environ.get('USE_INSTANCE_IMAGE', 'true').lower() == 'true':
            # Use a different instance image for each instance of swe-bench eval
            base_container_image = get_instance_docker_image(instance, data_source)
            logger.info(
                f'Using instance container image: {base_container_image}. '
                f'Please make sure this image exists. '
                f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
            )
        else:
            base_container_image = SWE_BENCH_CONTAINER_IMAGE
            logger.info(f'Using swe-bench container image: {base_container_image}')
        
        sandbox_config = get_default_sandbox_config_for_eval()
        sandbox_config.base_container_image = base_container_image
        sandbox_config.enable_auto_lint = True
        sandbox_config.use_host_network = False
        sandbox_config.platform = 'linux/amd64'
        extra_kwargs = {}
        if max_iterations is not None:
            extra_kwargs['max_iterations'] = max_iterations

        app_config = AppConfig(
            run_as_openhands=False,
            runtime='remote',
            sandbox=sandbox_config,
            workspace_base=None,
            workspace_mount_path=None,
            **extra_kwargs,
        )
        if agent_config is not None:
            app_config.set_agent_config(agent_config)

        return app_config
    
    @classmethod
    async def initialize_runtime(
        cls,
        instance: pd.Series,
        dataset,
        agent_cfg, 
        max_iterations
    ):
        """Initialize the runtime for the agent.

        This function is called before the runtime is used to run the agent.
        """
        app_config = cls.get_config(instance, dataset, agent_cfg, max_iterations)
            
        # Create runtime
        runtime = create_runtime(app_config)

        # Connect runtime
        await runtime.connect()

        await call_sync_from_async(initialize_runtime, runtime, instance, dataset)

        return runtime


    
    @classmethod
    def complete_runtime(
        cls,
        runtime: Runtime,
        instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
        dataset: str,
    ) -> dict[str, Any]:
        """Complete the runtime for the agent.

        This function is called before the runtime is used to run the agent.
        If you need to do something in the sandbox to get the correctness metric after
        the agent has run, modify this function.
        """
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Completion Fn')
        logger.info('-' * 30)
        obs: CmdOutputObservation
        workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)

        action = CmdRunAction(command=f'cd {workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        if obs.exit_code == -1:
            # The previous command is still running
            # We need to kill previous command
            logger.info('The previous command is still running, trying to kill it...')
            action = CmdRunAction(command='C-c', is_input=True)
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

            # Then run the command again
            action = CmdRunAction(command=f'cd {workspace_dir_name}')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        if obs.exit_code == -1:
            # The previous command is still running
            # We need to kill previous command
            logger.info('The previous command is still running, trying to ctrl+z it...')
            action = CmdRunAction(command='C-z', is_input=True)
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

            # Then run the command again
            action = CmdRunAction(command=f'cd {workspace_dir_name}')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to cd to {workspace_dir_name}: {str(obs)}',
        )

        # r2e directly run the tests
        if "r2e-gym" in dataset:
            action = CmdRunAction(
                    command=f'bash /root/run_tests.sh'
                )
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            if isinstance(obs, CmdOutputObservation):
                test_output = obs.content
                output = re.sub(r"\x1b\[[0-9;]*m|\r", "", test_output)
                assert isinstance(output, str)
                from .r2e_utils import parse_log_pytest, get_reward
                results = parse_log_pytest(output)
                reward = get_reward(results, instance)
                return {"reward": reward}
            else:
                return {"reward": 0, "finish_reason": "error_evaluation"}

        
        action = CmdRunAction(command='git config --global core.pager ""')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to git config --global core.pager "": {str(obs)}',
        )

        # First check for any git repositories in subdirectories
        action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to find git repositories: {str(obs)}',
        )

        git_dirs = [p for p in obs.content.strip().split('\n') if p]
        if git_dirs:
            # Remove all .git directories in subdirectories
            for git_dir in git_dirs:
                action = CmdRunAction(command=f'rm -rf "{git_dir}"')
                action.set_hard_timeout(600)
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                assert_and_raise(
                    isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                    f'Failed to remove git directory {git_dir}: {str(obs)}',
                )

        # add all files
        action = CmdRunAction(command='git add -A')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to git add -A: {str(obs)}',
        )

        # Remove binary files from git staging
        action = CmdRunAction(command=remove_binary_files_from_git())
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to remove binary files: {str(obs)}',
        )

        n_retries = 0
        git_patch = None
        while n_retries < 5:
            if "swe-smith" in dataset or "r2e-gym" in dataset:
                # For swe-smith, directly get the git diff without base commit
                action = CmdRunAction(
                    command=f'git diff --no-color --cached > patch.diff'
                )
            else:
                action = CmdRunAction(
                    command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
                )
            action.set_hard_timeout(max(300 + 100 * n_retries, 600))
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            n_retries += 1
            if isinstance(obs, CmdOutputObservation):
                if obs.exit_code == 0:
                    # Read the patch file
                    action = FileReadAction(path='patch.diff')
                    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                    if isinstance(obs, FileReadObservation):
                        git_patch = obs.content
                        break
                    elif isinstance(obs, ErrorObservation):
                        # Fall back to cat "patch.diff" to get the patch
                        assert 'File could not be decoded as utf-8' in obs.content
                        action = CmdRunAction(command='cat patch.diff')
                        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                        logger.info(action, extra={'msg_type': 'ACTION'})
                        obs = runtime.run_action(action)
                        assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0
                        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                        git_patch = obs.content
                        break
                    else:
                        assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
                else:
                    logger.info('Failed to get git diff, retrying...')
                    sleep_if_should_continue(10)
            elif isinstance(obs, ErrorObservation):
                logger.error(f'Error occurred: {obs.content}. Retrying...')
                sleep_if_should_continue(10)
            else:
                assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

        assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

        # Remove binary diffs from the patch
        git_patch = remove_binary_diffs(git_patch)

        logger.info('-' * 30)
        logger.info('END Runtime Completion Fn')
        logger.info('-' * 30)
        return {'git_patch': git_patch}
    
    @classmethod
    async def evaluate_result(cls, instance, run_results, instance_id, trajectory_id, dataset) -> bool:
        app_config = cls.get_config(instance, dataset)
        
        try:
            # Create runtime
            # TODO(csy): some tasks may not need a runtime for evaluation
            runtime = create_runtime(app_config)
            
            # Connect runtime
            await runtime.connect()
            
            eval_results: bool = await call_sync_from_async(evaluate_result, runtime, instance, run_results, instance_id, trajectory_id, dataset)
        finally:
            logger.info(f"Running cleanup for eval agent task for instance {instance_id}, trajectory {trajectory_id}")
            if 'runtime' in locals() and runtime:
                runtime.event_stream.close()
                runtime.close()
        return eval_results




def evaluate_result(runtime, instance, run_results, instance_id, trajectory_id, dataset) -> bool:
        """Apply patch and evaluate the solution."""
        if 'swe-gym' in dataset:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swegym.harness.test_spec import (
                make_test_spec,
            )
        elif 'swe-smith' in dataset:
            from swesmith.harness.grading import get_eval_report
            from swebench.harness.constants import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from .swesmith_utils import make_test_spec
        else:  # Newer version of SWE-Bench have different import paths
            from swebench.harness.grading import get_eval_report
            from swebench.harness.constants import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swebench.harness.test_spec import (
                make_test_spec,
            )
        
        model_patch = run_results.get('git_patch', None)
        if not model_patch:
            raise Exception(f"No git patch found for instance {instance_id}, trajectory {trajectory_id}")
        
        test_spec = make_test_spec(instance=instance)
        model_patch = process_git_patch(model_patch)
        # Get patch and save it to /tmp/patch.diff
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch file
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            runtime.copy_to(patch_file_path, '/tmp')
            # Eval script
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(test_spec.eval_script)
            runtime.copy_to(eval_script_path, '/tmp')

        # Set +x
        action = CmdRunAction(command='chmod +x /tmp/eval.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Apply patch
        if 'swe-smith' in dataset:
            # need to fetch and checkout the branch first
            exec_command = (
                "cd /testbed && "
                "git fetch && "
                f"git checkout {instance['instance_id']} && "
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
        else:
            exec_command = (
                'cd /testbed && '
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
        action = CmdRunAction(command=exec_command)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        apply_patch_output = obs.content
        assert isinstance(apply_patch_output, str)
        # instance['test_result']['apply_patch_output'] = apply_patch_output

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            raise Exception(f"Instance {instance_id}, trajectory {trajectory_id} {APPLY_PATCH_FAIL}:\n{apply_patch_output}")
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}, {trajectory_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}, {trajectory_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1200  # 20 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        raise Exception(
                            f'[{instance_id}, {trajectory_id}] Evaluation timed out after {timeout} seconds'
                        )
                    check_action = CmdRunAction(
                        command=f'ps -p {pid} > /dev/null; echo $?'
                    )
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}, {trajectory_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}, {trajectory_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    # instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}, {trajectory_id}] Grading answer...')
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)
                        try:
                            extra_kwargs = {}
                            if 'swe-smith' in dataset:
                                # SWE-Gym uses a different version of the package, hence a different eval report argument
                                extra_kwargs['test_log_path'] = test_output_path
                            else:
                                extra_kwargs['log_path'] = test_output_path
                            
                            if 'swe-smith' in dataset:
                                extra_kwargs['inst'] = instance
                            else:
                                extra_kwargs['test_spec'] = test_spec
                                extra_kwargs['include_tests_status'] = True
                            
                            _report = get_eval_report(
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                **extra_kwargs,
                            )
                            # in swe-smith, the report is a single dict
                            # in swe-gym and swe-bench, the report is a dict with instance_id
                            report = _report if 'swe-smith' in dataset else _report[instance_id]
                            logger.info(
                                f"[{instance_id}, {trajectory_id}] report: {report}\nResult for [{instance_id}, {trajectory_id}]: resolved: {report['resolved']}"
                            )
                            return report['resolved']
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}, {trajectory_id}] Error when getting eval report: {e}'
                            )
                            return False
            else:
                raise Exception(f'[{instance_id}, {trajectory_id}] Error when starting eval:\n{obs.content}')
        else:
            raise Exception(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )

def get_instance_docker_image(instance, data_source) -> str:
    if 'swe-smith' in data_source:
        image_name = instance['image_name']
        return f"jyangballin/{image_name.replace('__', '_1776_')}"
    if 'r2e-gym' in data_source:
        return instance["instance_id"]

    instance_id = instance['instance_id']
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

# Helper function for sandbox config
def get_default_sandbox_config_for_eval():
    return SandboxConfig(
        use_host_network=False,
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
    )



def remove_binary_diffs(patch_text):
    """
    Remove binary file diffs from a git patch.

    Args:
        patch_text (str): The git patch text

    Returns:
        str: The cleaned patch text with binary diffs removed
    """
    lines = patch_text.splitlines()
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith('diff --git '):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif 'Binary files' in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return '\n'.join(cleaned_lines)

# avoid depend on file cmd
def remove_binary_files_from_git():
    """
    Generate a bash command to remove binary files from git staging.

    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\?\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (test -x "$file" || git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


def is_fatal_evaluation_error(error: str | None) -> bool:
    if not error:
        return False

    FATAL_EXCEPTIONS = [
        AgentRuntimeError,
        AgentRuntimeBuildError,
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeNotReadyError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
        ConnectionError,
    ]

    if any(exception.__name__ in error for exception in FATAL_EXCEPTIONS):
        logger.error(f'Fatal evaluation error detected: {error}')
        return True

    return False

def process_git_patch(patch):
    if not isinstance(patch, str):
        return ''

    if not patch.strip():
        print(f'Skipping empty patch....')
        # skip empty patches
        return ''

    patch = patch.replace('\r\n', '\n')
    # There might be some weird characters at the beginning of the patch
    # due to some OpenHands inference command outputs

    # FOR EXAMPLE:
    # git diff --no-color --cached 895f28f9cbed817c00ab68770433170d83132d90
    # [A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[K0
    # diff --git a/django/db/models/sql/.backup.query.py b/django/db/models/sql/.backup.query.py
    # new file mode 100644
    # index 0000000000..fc13db5948

    # We "find" the first line that starts with "diff" and then we remove lines before it
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break

    patch = patch.rstrip() + '\n'  # Make sure the last line ends with a newline
    return patch
