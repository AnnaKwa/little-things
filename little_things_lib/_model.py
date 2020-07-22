import numpy as np
from typing import Sequence, Tuple


class Model:
    def __init__(
        self,
        parameters: Sequence[str],
        bounds: Mapping[str, Tuple[float, float]]=None,
        log_parameters: Sequence[str]=None,
    ):
        """[summary]

        Args:
            parameters (Sequence[str]): Ordered parameter names
            bounds (Mapping[str, Tuple[float, float]], optional): 
                Bounds for parameters. If not provided, will be (-inf, inf)
            log_parameters (Sequence[str], optional): Parameters to convert to log
                space when fitting MCMC.
        """

        self.parameters = parameters
        self.bounds = bounds or (-np.inf, np.inf)

    

    

