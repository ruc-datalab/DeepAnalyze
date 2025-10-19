from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.dispatch import (
    MeshDispatch,
    PassThroughDispatch,
    MeshRank,
    ActorInfo,
    DispatchRegistry,
    Dispatch,
)
import ray
import torch
from typing import List, Optional, Union
from ray import ObjectRef
import pytest


@ray.remote
class RayActor:
    def __init__(self, rank: int, dp_rank: int):
        self.rank = rank
        self.dp_rank = dp_rank

    def do_work(self, data: TrainingInputBatch):
        # intentionally create different outputs for each rank
        data["a"] += self.rank
        return data

    def dummy(self, a, b):
        return


class RayActorGroup:
    def __init__(self, num_actors: int):
        sp_size = 2
        dp_size = num_actors // sp_size
        self.actors = [RayActor.remote(i, i % dp_size) for i in range(num_actors)]
        self.actor_infos = [
            ActorInfo(
                actor, MeshRank(dp=i % dp_size, sp=i // dp_size, tp=0, pp=0, world_size=num_actors, dp_size=dp_size)
            )
            for i, actor in enumerate(self.actors)
        ]

    def mesh_dispatch_and_collect(self, data: TrainingInputBatch):
        object_refs = MeshDispatch.dispatch(self.actor_infos, "do_work", data)
        ret = MeshDispatch.sync_collect(self.actor_infos, object_refs)
        return ret

    def pass_through_dispatch(self, a, b):
        # just pass values as is
        object_refs = PassThroughDispatch.dispatch(self.actor_infos, "dummy", a, b)
        ret = PassThroughDispatch.sync_collect(self.actor_infos, object_refs)
        return ret


def test_mesh_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    data = TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])})
    databatch = actor_group.mesh_dispatch_and_collect(data)
    # only dp rank 0, 1, 2, 3, sp 0 will have the contributed to the output.
    # In this case, the rank for these are 0, 1, 2, 3.
    assert torch.equal(databatch["a"], torch.tensor([1, 3, 5, 7]))


def test_pass_through_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    ret = actor_group.pass_through_dispatch(1, 2)
    assert ret is None


def test_mesh_dispatch_with_mixed():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    object_refs = MeshDispatch.dispatch(
        actor_group.actor_infos,
        "do_work",
        TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])}),
    )
    object_refs[0] = ray.put(None)
    with pytest.raises(AssertionError):
        MeshDispatch.sync_collect(actor_group.actor_infos, object_refs)


def test_dispatch_registry():
    # add a custom dispatch type
    try:

        class CustomDispatch(Dispatch):
            @classmethod
            def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
                pass

            @classmethod
            def sync_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef], nonblocking: bool = False
            ) -> Union[List[ObjectRef], TrainingInputBatch]:
                pass

            @classmethod
            def async_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
            ) -> Optional[TrainingInputBatch]:
                pass

        DispatchRegistry.register("custom", CustomDispatch)
        assert DispatchRegistry.get("custom") == CustomDispatch
        assert DispatchRegistry.list_registered() == {
            "mesh": MeshDispatch,
            "pass_through": PassThroughDispatch,
            "custom": CustomDispatch,
        }
    finally:
        DispatchRegistry._registry.pop("custom")
