# Utility functions

def param_override(p_hydra, p_ray):
    """Make a decision whether a certain hyperparameter will tuned by ray or 
    not based on hydra configuration value.
    Parameters
    ----------
    p_hydra :
        parameter value from hydra configuration.
    p_ray : 
        paramter value from ray tune search space.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    Examples
    --------
    >>> from utils import param_override
    >>> param_override(None, 1)
    1
    >>> param_override('None', 1)
    1
    >>> param_override('nOnE', 1)
    1
    >>> param_override(3, 1)
    3
    """
    if type(p_hydra) == str:
        p_hydra = p_hydra.lower() # Remove capital letters

    if p_hydra in [None, 'none']: # Cover Nonetype None and string type None
        return p_ray
    else:
        return p_hydra