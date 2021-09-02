from . import adcirc_utils as au

class BaseSimulation:
    """A base simulation class representing a single ADCIRC run.

    The class contains methods shared by all simulations.
    """

    def __init__(self, **params):
        """Initialize the simulation.

        """
        self.params = params

