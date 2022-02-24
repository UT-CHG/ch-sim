from . import adcirc_utils as au
from .manager import SimulationManager
import argparse as ap
import os
import logging
import json
import math
from dotenv import load_dotenv
import subprocess
from .pylauncher4 import IbrunLauncher

logger = logging.getLogger(__name__)


class BaseSimulator:
    """A base simulation class representing a single ADCIRC run.

    The class contains methods shared by all simulations.
    """

    REQUIRED_PARAMS = {"execs_dir": str, "inputs_dir": str}

    def __init__(
        self, system, user=None, psw=None, allocation=None, deps=None, name=None
    ):
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
        self.psw = psw

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

    def _validate_config(self):
        """Ensure config is accurate
        """

        for arg, arg_type in self.REQUIRED_PARAMS.items():
            val = self.config.get(arg)
            if val is None:
                raise ValueError(f"Missing requred argument '{arg}'")

            if type(arg_type) is list:
                type_match = type(val) in arg_type
            else:
                type_match = type(val) is arg_type

            if not type_match:
                raise TypeError(
                    f"Got value {val} of type '{type(val)}' for argument '{arg}',"
                    f" expected type '{arg_type}'"
                )

    def run(self, **config):
        """Either setup the simulation/submit jobs OR run on TACC resources.
        """

        # Determine what we need to do
        args = self._get_args()
        action = args.action
        self.config = config
        if action == "setup":
            self._validate_config()
            self.setup(**config)
            return
        elif action == "run":
            job_config = self._get_job_config()
            self.setup_job(job_config)
            self.run_job(job_config)
        else:
            raise ValueError(f"Unsupported action {action}")

    def setup(self, **config):
        """Setup simulation files on TACC.
        """

        manager = SimulationManager(self.system, user=self.user, psw=self.psw)
        manager.setup_simulation(self, **config)

    def setup_job(self, job_config):
        """Called before the main work is done for a job
        """

        exec_name = self._get_exec_name()
        os.makedirs("inputs", exist_ok=True)
	# if there are directories in the inputs dir, cp will still copy the inputs,
	# but will have a non-zero exit code
        self._run_command(f"cp {self.config['inputs_dir']}/* inputs", check=False)
        self._run_command(
            f"cp {self.config['execs_dir']}/" + "{adcprep," + exec_name + "} ."
        )
        self._run_command(f"chmod +x adcprep {exec_name}")

    def run_job(self, job_config):
        """Run on HPC resources.

        This is the entry point for a single job within the simulation.
        """

        node_count = int(job_config.get("node_count"))
        procsPerNode = int(job_config.get("processors_per_node"))
        writers, workers = BaseSimulator.get_writers_and_workers(
            node_count, procsPerNode
        )

        logger.info("Starting first adcprep run. . .")
        # ADCPREP returns an exit code of 1 on success - it's terrible . . .
        self._run_command(
            f"printf '{workers}\\n1\\nfort.14\\n' | ./adcprep", check=False
        )
        logger.info("Starting second adcprep run. . .")
        self._run_command(f"printf '{workers}\\n2\\n' | ./adcprep", check=False)
        logger.info("Completed second adcprep run. Starting ADCIRC . . .")
        exec_name = self._get_exec_name()
        self._run_command(f"ibrun ./{exec_name} -I inputs -W {writers}")
        logger.info("Completed ADCIRC run.")

    def _run_command(self, command, check=True, **kwargs):
        logger.info(f"Running '{command}'")
        subprocess.run(command, shell=True, check=check, **kwargs)

    def _get_exec_name(self):
        swan = self.config.get("swan", False)
        return "padcswan" if swan else "padcirc"

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

        with open("job.json", "r") as fp:
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
            "app": self.name,
            "node_count": config.get("node_count", 1),
            "queue": config.get("queue", "development"),
            "processors_per_node": config.get("processors_per_node", 48),
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
        hours = hours - 24 * days
        minutes = int(60 * (hours - math.floor(hours)))
        hours = int(hours)
        if days:
            return f"{days}-{hours:02}:{minutes:02}:00"
        else:
            return f"{hours:02}:{minutes:02}:00"

    @staticmethod
    def get_writers_and_workers(node_count, procsPerNode):
        totalProcs = node_count * procsPerNode
        writers = max(1, node_count // 2)
        workers = totalProcs - writers
        return writers, workers


class EnsembleSimulator(BaseSimulator):
    """A simulation class representing a set of ADCIRC runs.
    """

    REQUIRED_PARAMS = {
        # executables directory
        "execs_dir": str,
        # directory for common inputs across all runs OR template input files
        "inputs_dir": str,
        # list of dictionaries describing each run
        "runs": list,
        # per-task runtime
        "runtime": [float, int],
        # per-task node_count
        "node_count": int,
    }

    def _validate_config(self):
        super()._validate_config()

        for r in self.config["runs"]:
            if type(r) is not dict:
                raise TypeError(
                    "Runs should be a list of dictionaries with run information!"
                )

    def generate_job_configs(self, manager, **config):
        maxJobNodes = config.get("maxJobNodes", 30)
        maxJobRuntime = config.get("maxJobRuntime", 24)
        runs = config["runs"]
        # Note that for EnsembleSimulator config["node_count"] is the per-run nodes, NOT for the entire simulation
        nodesPerRun, timePerRun = config["node_count"], config["runtime"]
        numSlots = int(maxJobNodes // nodesPerRun)
        consecRuns = int(maxJobRuntime // timePerRun)
        runsPerJob = numSlots * consecRuns

        if not numSlots:
            raise ValueError(
                f"Nodes per run is {config['node_count']}, but the maximum is {maxJobNodes}."
                f" If you really need that many nodes for a single run, increase the maximum by setting maxJobNodes."
            )
        elif not consecRuns:
            raise ValueError(
                f"Runtime for a single run is {config['runtime']} hours, but the maximum is {maxRuntime} hours."
                f" If you really need a longer runtime, increase the maximum by setting maxJobRuntime."
                " Note that the maximum per-job runtime on TACC is typically 48 hours."
            )

        res = []
        for start in range(0, len(runs), runsPerJob):
            stop = min(len(runs), start + runsPerJob)
            jobRuns = runs[start:stop]
            inds = list(range(start, stop))
            input_config = config.copy()
            numJobRuns = len(jobRuns)
            if numJobRuns == runsPerJob:
                input_config["node_count"] = numSlots * nodesPerRun
                input_config["runtime"] = consecRuns * timePerRun
            else:
                input_config["node_count"] = min(numSlots, numJobRuns) * nodesPerRun
                input_config["runtime"] = math.ceil(numJobRuns / numSlots) * timePerRun
            config = self._base_job_config(**input_config)
            config["jobRuns"] = jobRuns
            config["jobRunInds"] = inds
            res.append(config)
        return res

    def setup_job(self, job_config):
        super().setup_job(job_config)
        os.makedirs("runs", exist_ok=True)
        ndigits = len(str(len(self.config["runs"])))
        self.run_dirs = []
        job_dir = job_config['job_dir']
        for run, ind in zip(job_config["jobRuns"], job_config["jobRunInds"]):
            run_dir = f"{job_dir}/runs/run{ind:0{ndigits}}"
            os.makedirs(run_dir, exist_ok=True)
            self._run_command(f"ln -sf {job_dir}/inputs/* {run_dir}")
            self.setup_run(run, run_dir, job_config)
            self.run_dirs.append(run_dir)

    def setup_run(self, run, run_dir, job_config):
        """Do any setup specific to the run.

        For parameter sweeps, this is where parameters are set.
        """

        if "inputs_dir" in run:
            # add extra inputs/overwrite existing ones
            self._run_command(f"ln -sf {run['inputs_dir']}/* {run_dir}")

        if "parameters" in run:
            # TODO - edit input parameter files, performing copy-on-write
            pass

    def run_job(self, job_config):

        # Step 1 - generate Pylauncher input
        tasks = []

        # We have to be careful to use node_count from config - because that corresponds
        # to the per-run nodes
        writers, workers = BaseSimulator.get_writers_and_workers(
            self.config["node_count"], job_config["processors_per_node"]
        )

        total = workers + writers
        exec_name = self._get_exec_name()
        job_dir = job_config['job_dir']

        for run_dir in self.run_dirs:
            pre_process = ";".join(
                [
                    f"cd {run_dir}",
                    f"printf '{workers}\\n1\\nfort.14\\n' | {job_dir}/adcprep",
                    f"printf '{workers}\\n2\\n' | {job_dir}/adcprep",
                    f"cd {job_dir}",
                ]
            )

            task = {
                "cores": total,
                "pre_process": pre_process,
                "main": f"{job_dir}/{exec_name} -I {run_dir} -O {run_dir} -W {writers}",
            }

            tasks.append(task)

        # step 2 - launch Pylauncher with this input
        outfile = "pylauncher_jobs.json"
        with open(outfile, "w") as fp:
            json.dump(tasks, fp)

        IbrunLauncher(outfile, pre_post_process=True)
