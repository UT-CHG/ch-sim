from . import adcirc_utils as au
from .manager import SimulationManager
import argparse as ap
import os

class BaseSimulator:
    """A base simulation class representing a single ADCIRC run.

    The class contains methods shared by all simulations.
    """

    def __init__(self, system, user=None,
        deps=None, name=None):
        """Initialize the simulator.

        Args:
            system - the HPC system
            user - the user
            deps - a list of directories and files that are required to run the simulator.
                Defaults to just the current file.
            name - a name for the simulation. If not passed, the name is inferred from the current filename.
        """

        self.system = system
        self.user = user
        self.name = name
        self.deps = deps

        import __main__
        self.script_file = script_file = __main__.__file__
        if name is None:
           self.name = os.path.basename(script_file).split(".")[0]

    def _validate_config(self, config):
        """Ensure config is accurate
        """    

        for arg in ['execs_dir', 'inputs_dir']:
            if config.get(arg) is None:
                raise ValueError(f"Missing requred argument '{arg}'")

    def run(self, **config):
        """Either setup the simulation/submit jobs OR run on TACC resources.
        """
        
        # Determine what we need to do
        args = self._get_args()
        action = args.action
        if action == "setup":
            self._validate_config(config)
            self.setup(**config)
            return
        elif action == "run":
            self._run_on_tacc(**config)
        else:
            raise ValueError(f"Unsupported action {action}")

    def setup(self, **config):
        """Setup simulation files on TACC.
        """

        manager = SimulationManager(self.system, user=self.user)
        manager.setup_simulation(self, **config)

    def _run_on_tacc(self, **config):
        """Run on HPC resources.

        This is the entry point for a single job within the simulation.
        """

        print("I'm Running on TACC!!!")

    def _get_args(self):
        parser = ap.ArgumentParser()
        parser.add_argument("--action", default="setup", choices=["setup", "run"])
        return parser.parse_args()

    def test(self):
        """Do a dry run of the simulation.
        
        This mocks outputs of ADCIRC runs and allows for testing of user code.
        """
        pass
    

class EnsembleSimulator(BaseSimulator):
    """A simulation class representing a set of ADCIRC runs.
    """

    def generate():
        pass
