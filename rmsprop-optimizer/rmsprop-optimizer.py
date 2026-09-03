import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w = np.array(w)
    grad = np.array(g)
    s_prev = np.array(s)
    # Write code here
    s = beta * s_prev + (1 - beta) * (grad ** 2)


    w = w - lr * grad / (np.sqrt(s) + eps) 

    return w, s
    