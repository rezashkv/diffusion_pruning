import torch


def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


# The functions below are used to get unwrapped model attributes and methods in the case of DistributedDataParallel
def get_module_attribute(model, attr_name):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return getattr(model.module, attr_name)
    else:
        return getattr(model, attr_name)


def set_module_attribute(model, attr_name, value):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        setattr(model.module, attr_name, value)
    else:
        setattr(model, attr_name, value)


def call_module_method(model, method_name, *args, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        method = getattr(model.module, method_name)
    else:
        method = getattr(model, method_name)
    return method(*args, **kwargs)
