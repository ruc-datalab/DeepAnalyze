"""Defines dispatch and collect logic for distributed training"""

from dataclasses import dataclass
from ray.actor import ActorHandle
from typing import List, Tuple, Optional, Dict, Type, Any
import asyncio
from abc import ABC, abstractmethod
import ray
from ray import ObjectRef
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
import inspect


@dataclass
class MeshRank:
    """Represents a rank in the device mesh.

    This is a tuple of (DP, SP, TP, PP) ranks.
    """

    dp: int
    sp: int
    tp: int
    pp: int

    world_size: int
    dp_size: int

    def is_primary_dp_rank(self) -> bool:
        """Check if this rank is the primary DP rank.

        This is the rank with (SP=0, TP=0, PP=0)
        """
        return self.tp == 0 and self.pp == 0 and self.sp == 0

    def __str__(self) -> str:
        return f"MeshRank(dp={self.dp}, sp={self.sp}, tp={self.tp}, world_size={self.world_size}, dp_size={self.dp_size})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ActorInfo:
    """Actor information for distributed training.

    This includes the actor handle and the rank in the device mesh.
    """

    handle: ActorHandle
    rank: MeshRank


class Dispatch(ABC):
    """Base class for dispatch types

    Dispatch types are responsible for:
    - dispatching method calls to actors handling data sharding if necessary
    - collecting results from actors and concatenating results if necessary
    - validating arguments for dispatch
    """

    @classmethod
    @abstractmethod
    def dispatch(
        cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs
    ) -> List[ObjectRef]:
        """Dispatches method calls to the actors with data sharing if necessary."""
        pass

    @classmethod
    @abstractmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        """Collects results from the actors asynchronously in an asyncio-compatible way."""
        pass

    @classmethod
    @abstractmethod
    def sync_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        """Collects results from the actors synchronously and returns a `TrainingOutputBatch`."""
        pass

    @classmethod
    @abstractmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        """Validate and process arguments for dispatch.

        Returns:
            Tuple of (args, kwargs) to be passed to dispatch
        """
        pass


class MeshDispatch(Dispatch):
    """Mesh dispatch type to dispatch data to a group of actors along the device mesh.

    Supports DP (Data Parallel), SP (Sequence Parallel), TP (Tensor Parallel) and PP (Pipeline Parallel) parallelism.
    The actor method should accept a single argument - the data batch.

    For data dispatch:

    * The input data is chunked into `dp_size` equal chunks, where `dp_size` is the size of data parallelism.
    * Each actor with the same DP rank processes the same data chunk in parallel.

    For data collection:

    * Data is collected only from the primary rank of each model/sequence parallel group.
    * The primary rank is defined as the rank with (SP=0, TP=0, PP=0).
    * The collected chunks are concatenated in order of DP rank to reconstruct the full data.

    Example: For a world size of 8, with DP size=2, SP size=2, TP size=2, PP size=1:

    * Data dispatch: The data is chunked into 2 chunks. All actors with DP rank 0 process the first chunk,
      and all actors with DP rank 1 process the second chunk.
    * Data collection: Only two actors contribute to the final output - the primary rank from each DP group:
      (DP=0, SP=0, TP=0, PP=0) and (DP=1, SP=0, TP=0, PP=0). Their chunks are concatenated in order.

    """

    @classmethod
    def dispatch(
        cls, actor_infos: List[ActorInfo], method: str, data: TrainingInputBatch
    ) -> List[ObjectRef]:
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = []
        dp_size = actor_infos[0].rank.dp_size
        assert (
            len(data) % dp_size == 0
        ), "data batch size must be divisible by dp_size, got {} and {}".format(
            len(data), dp_size
        )
        chunk_size = len(data) // dp_size
        data_chunks: List[TrainingInputBatch] = data.chunk(chunk_size)

        for actor_info in actor_infos:
            # index into tensordict to get the correct data to send
            data_to_send = data_chunks[actor_info.rank.dp]
            object_refs.append(getattr(actor_info.handle, method).remote(data_to_send))
        return object_refs

    @classmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        assert len(actor_infos) == len(
            object_refs
        ), "`actor_infos` and `object_refs` must have the same length"
        all_objects = await asyncio.gather(*object_refs)
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        return

    @classmethod
    def sync_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        assert len(actor_infos) == len(
            object_refs
        ), "`actor_infos` and `object_refs` must have the same length"
        all_objects = ray.get(object_refs)
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        # all should be none
        assert all(
            obj is None for obj in all_objects
        ), "Got a mix of `None` and non-`None` objects"
        return

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        sig = inspect.signature(cls.dispatch)
        # pass dummy actor_infos and method_name
        bound_args = sig.bind([], "dummy", *args, **kwargs)
        bound_args.apply_defaults()
        data = bound_args.arguments.get("data")

        # Check if there are any extra arguments
        if len(bound_args.arguments) > 3:  #  data, actor_infos, method_name
            # remove actor_infos and method_name - not added by user
            bound_args.arguments.pop("actor_infos")
            bound_args.arguments.pop("method")
            raise ValueError(
                f"MeshDispatch only accepts 'data' as an argument, got extra args: {bound_args.arguments}"
            )

        data = bound_args.arguments.get("data")
        if not isinstance(data, TrainingInputBatch):
            raise ValueError(
                f"For MeshDispatch, `data` entry should be a `TrainingInput`, got {data}"
            )
        args = (data,)
        kwargs = {}
        return args, kwargs


class PassThroughDispatch(Dispatch):
    """PassThrough dispatch type to dispatch data to a group of actors without any sharding.

    This is useful for cases where we want to run the same method on all the actors.
    Supports methods with any number of arguments.
    """

    @classmethod
    def dispatch(
        cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs
    ) -> List[ObjectRef]:
        return [
            getattr(actor_info.handle, method).remote(*args, **kwargs)
            for actor_info in actor_infos
        ]

    @classmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        all_objects = await asyncio.gather(*object_refs)
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        return

    @classmethod
    def sync_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        data_batches = ray.get(object_refs)
        if len(data_batches) > 0 and data_batches[0] is not None:
            assert isinstance(
                data_batches[0], TrainingOutputBatch
            ), "data_batches must be a list of `TrainingOutputBatch` objects"
            return concatenate_outputs_after_mesh_dispatch(actor_infos, data_batches)
        # all should be none
        assert all(
            obj is None for obj in data_batches
        ), "Got a mix of `None` and non-`None` objects"
        return

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # no validation needed just pass everything
        return args, kwargs


class DispatchRegistry:
    _registry: Dict[str, Type[Dispatch]] = {
        "mesh": MeshDispatch,
        "pass_through": PassThroughDispatch,
    }

    @classmethod
    def register(cls, name: str, dispatch_class: Type[Dispatch]) -> None:
        """Register a new dispatch type."""
        assert issubclass(dispatch_class, Dispatch)
        cls._registry[name] = dispatch_class

    @classmethod
    def get(cls, name: str) -> Type[Dispatch]:
        """Get a registered dispatch type."""
        if name not in cls._registry:
            raise KeyError(f"Dispatch type '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[Dispatch]]:
        """List all registered dispatch types."""
        return cls._registry


def register_dispatch_type(name: str, dispatch_class: Type) -> None:
    DispatchRegistry.register(name, dispatch_class)


def concatenate_outputs_after_mesh_dispatch(
    actor_infos: List[ActorInfo], data_batches: List[TrainingOutputBatch]
) -> TrainingOutputBatch:
    """Concatenate data batches from different ranks after mesh dispatch.

    - Data is collected only from the primary DP rank.
    - The collected chunks are concatenated in order of DP rank to reconstruct the full data.
    """
    assert len(actor_infos) == len(
        data_batches
    ), "`actor_infos` and `data_batches` must have the same length"
    shards = []
    # collect in-order
    dp_rank_to_shard = {}
    for actor_info, data_batch in zip(actor_infos, data_batches):
        if actor_info.rank.is_primary_dp_rank():
            dp_rank = actor_info.rank.dp
            dp_rank_to_shard[dp_rank] = data_batch
    for i in range(actor_infos[0].rank.dp_size):
        shards.append(dp_rank_to_shard[i])
    return TrainingOutputBatch.cat(shards)
