from . import adcirc_utils as au
from .manager import SimulationManager
import argparse as ap
import os
import logging
import json
import math
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class BaseSimulator:
    """A base simulation class representing a single ADCIRC run.

    The class contains methods shared by all simulations.
    """

    def __init__(self, system,
        user=None, psw=None,
        allocation=None, deps=None, name=None):
        """Initialize the simulator.

        Args:
            system - the HPC system
            user - the user
            psw_file - a text file with the password (optional)
            allocation - the allocation ID
            deps - a list of directories and files that are required to run the simulator.
                Defaults to just the current file.
            name - a name for the simulation. If not passed, the name is inferred from the current filename.
        """

        self.system = system
        self.user = user
        self.name = name
        self.deps = deps
        self.allocation = allocation
        self.psw=psw
        
        self._init_from_env()

        import __main__
        self.script_file = script_file = __main__.__file__
        if name is None:
           self.name = os.path.basename(script_file).split(".")[0]

    def _init_from_env(self):
        load_dotenv()
        for attr in ["PSW", "USER", "SYSTEM", "ALLOCATION"]:
            env_val = os.environ.get("CHSIM_" + attr)
            if env_val is not None and getattr(self, attr.lower()) is None:
                setattr(self, attr.lower(), env_val)

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

        manager = SimulationManager(self.system, user=self.user, psw=self.psw)
        manager.setup_simulation(self, **config)

    def _run_on_tacc(self, **simulation_config):
        """Run on HPC resources.

        This is the entry point for a single job within the simulation.
        """

        job_config = self._get_job_config()
        nodeCount = int(job_config.get("nodeCount"))
        procsPerNode = int(job_config.get("processorsPerNode"))
        totalProcs = nodeCount * procsPerNode
        writers = max(1, nodeCount // 2)
        workers = totalProcs - writers
        
        logger.info("Starting first adcprep run. . .")
        subprocess.run(f"printf '{workers}\\n1\\nfort.14\\n' | ./adcprep", shell=True, stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
        logger.info("Starting second adcprep run. . .")
        subprocess.run(f"printf '{workers}\\n2\\n' | ./adcprep", shell=True, stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
        logger.info("Completed second adcprep run. Starting ADCIRC . . .")
        subprocess.run(f"ibrun ./padcirc -W {writers}", shell=True, check=True, stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
        logger.info("Completed ADCIRC run.")

    def _get_args(self):
        parser = ap.ArgumentParser()
        parser.add_argument("--action", default="setup", choices=["setup", "run"])
        return parser.parse_args()

    def test(self):
        """Do a dry run of the simulation.
        
        This mocks outputs of ADCIRC runs and allows for testing of user code.
        """
        pass
    
    def _get_job_config(self):
        """Get config of local job
        """

        with open("job_config.json", "r") as fp:
            return json.load(fp)

    def generate_job_configs(self, manager, **config):
        """Create the job configs for the simulation.

        Args:
            manager - SimulationManager
                The SimulationManager with remote access to TACC. The jobs
                we generate can depend on the current simulation state - so
                this function will sometimes need to access files on TACC.
            config - the simulation config
        """

        return [self._base_job_config(**config)]

    def _base_job_config(self, **config):
        res = {
            "name": self.name,
            "appId": self.name,
            "nodeCount": config.get("nodeCount", 1),
            "queue": config.get("queue", "development"),
            "processesPerNode": config.get("processesPerNode", 48),
            "desc": "",
            "inputs": {},
            "parameters": {},
        }

        if self.allocation is not None:
            res["allocation"] = self.allocation
        if "runtime" in config:
            res["maxRunTime"] = BaseSimulator.hours_to_runtime_str(config["runtime"])
 
        return res

    @staticmethod
    def hours_to_runtime_str(hours):
        days = math.floor(hours / 24)
        hours = hours - 24*days
        minutes = int(60 * (hours - math.floor(hours)))
        hours = int(hours)
        if days:
            return f"{days}-{hours:02}:{minutes:02}:00"
        else:
            return f"{hours:02}:{minutes:02}:00"

class EnsembleSimulator(BaseSimulator):
    """A simulation class representing a set of ADCIRC runs.
    """

    def generate():
        pass
