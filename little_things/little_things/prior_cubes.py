from typing import Sequence

def prior_cube(
        cube,
        bounds: Sequence[Sequence[float]],
        return_cube: bool=True):
    """
    Transforming unit cube to have the right bounds.
    """
    for i,b in enumerate(bounds):
        cube[i] = cube[i]*(b[1]-b[0])+b[0]
    return cube if return_cube else None