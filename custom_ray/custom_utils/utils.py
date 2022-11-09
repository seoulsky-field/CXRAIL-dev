# Utility functions

def param_override(p_hydra, p_ray):
    if p_hydra is not None:
        return p_ray
    else:
        return p_hydra