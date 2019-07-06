import torch

from ignite.contrib.handlers import TensorboardLogger


def get_kernel_named_params(model, copy=False):    
    fn = lambda p: p.cpu().detach()
    if copy:
        fn = lambda p: p.cpu().clone().detach()
    return [(n, fn(p)) for n, p in model.named_parameters() 
            if "weight" in n and len(p.shape) == 4]


def layer_rotation(current_named_params, init_named_params):
    ret = []
    for (n1, p1), (n2, p2) in zip(current_named_params, init_named_params):
        assert n1 == n2, "{} vs {}".format(n1, n2)
        ret.append((n1, 1.0 - torch.cosine_similarity(p1.reshape(-1), p2.reshape(-1), dim=0).item()))
    return ret


def layer_rotation_stats(current_named_params, init_named_params):
    ret = layer_rotation(current_named_params, init_named_params)
    values = torch.tensor([v for n, v in ret])
    return {
        "min": torch.min(values),
        "max": torch.max(values),
        "mean": torch.mean(values),
        "std": torch.std(values)
    }


class LayerRotationStatsHandler:
    """Helper handler to log "Layer Rotation" of the model weights.
    Args:
        model (torch.nn.Module): model to log layer rotation statistics (min, max, mean, std)
    """
    def __init__(self, model):        
        self.model = model
        self.init_named_params = get_kernel_named_params(model, copy=True)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        
        current_named_params = get_kernel_named_params(self.model)
        stats = layer_rotation_stats(current_named_params, self.init_named_params)

        # For example:
        # {'min': tensor(9.4175e-06),
        # 'max': tensor(0.2845),
        # 'mean': tensor(0.0147),
        # 'std': tensor(0.0635)}

        for key, val in stats.items():
            logger.writer.add_scalar("layer_rotation/{}".format(key), val, global_step)
