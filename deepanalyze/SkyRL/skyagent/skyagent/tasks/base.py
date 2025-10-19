from abc import abstractmethod, ABC
from typing import Dict, Any, List


# TODO: See if it makes sense to have an async version
class BaseTask(ABC):

    @classmethod
    @abstractmethod
    async def initialize_runtime(cls, *args, **kwargs) -> Any:
        """Initialize the runtime for the task in an asyncio-compatible way"""
        pass

    @classmethod
    @abstractmethod
    def get_instruction(cls, *args, **kwargs) -> List[Dict[str, str]]:
        """Get the initial instruction for the agent in the OpenAI messages format"""
        pass

    @classmethod
    @abstractmethod
    def complete_runtime(cls, *args, **kwargs) -> Dict[str, Any]:
        """Complete or finalize the runtime for the task

        For example, this can involve extracting the git patch from the runtime for SWEBench.
        """
        pass

    @classmethod
    @abstractmethod
    async def evaluate_result(cls, *args, **kwargs) -> bool:
        """Evaluate model result for the task in an asyncio-compatible way"""
        pass
