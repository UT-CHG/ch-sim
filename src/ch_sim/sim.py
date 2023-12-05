from . import adcirc_utils as au
from .manager import SimulationManager, RemoteSimulationManager
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
        self, system=None, user=None, psw=None, allocation=None, deps=None, name=None,
        modules=None
    ):
        """Initialize the simulator.

        Args:
            system - the HPC system - if None will run on local system
            user - the user
            psw_file - a text file with the password (optional)
            allocation - the allocation ID
            deps - a list of directories and files that are required to run the simulator.
                Defaults to just the current file.
            name - a name for the simulation. If not passed, the name is inferred from the current filename.
            modules - a list of required modules
        """

        self.system = system
        self.user = user
        self.name = name
        self.deps = deps
        self.allocation = allocation
        self.psw = psw
        self.modules = modules
	
        self._init_from_env()

        import __main__

        self.script_file = script_file = __main__.__file__
        if name is None:
            self.name = os.path.basename(script_file).split(".")[0]

        self.args = self._get_args()

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
        action = self.args.action
        self.config = config
        if action == "setup":
            # add job dependency id
            config["dependency"] = self.args.dependency
            self._validate_config()
            self.setup(**config)
            return
        elif action == "run":
            self.job_config = self._get_job_config()
            # Ensure simulation-level config is identical to what was set at setup time
            self.config.update(self.job_config.get("sim_config", {}))
            self.setup_job()
            self.run_job()
        else:
            raise ValueError(f"Unsupported action {action}")

    def setup(self, **config):
        """Setup simulation files on TACC.
        """

        if self.system is not None:
            manager = RemoteSimulationManager(self.system, user=self.user, psw=self.psw)
        else: # assume local
            manager = SimulationManager()
        manager.setup_simulation(self, **config)

    def setup_job(self):
        """Called before the main work is done for a job
        """

        exec_name = self._get_exec_name()
        os.makedirs("inputs", exist_ok=True)
	# if there are directories in the inputs dir, cp will still copy the inputs,
	# but will have a non-zero exit code
        self._run_command(f"cp {self.config['inputs_dir']}/* inputs", check=False)
        self._run_command("ln -sf inputs/* .")
        self._run_command(
            f"cp {self.config['execs_dir']}/" + "{adcprep," + exec_name + "} ."
        )
        self._run_command(f"chmod +x adcprep {exec_name}")

    def run_job(self):
        """Run on HPC resources.

        This is the entry point for a single job within the simulation.
        """

        run = self.config
        run_dir = self.job_config['job_dir']
        pre_cmd = self.make_preprocess_command(run, run_dir)
        post_cmd = self.make_postprocess_command(run, run_dir)
        if pre_cmd is not None:
            self._run_command(pre_cmd, check=False)

        self._run_command("ibrun " + self.make_main_command(run, run_dir))
        
        if post_cmd is not None:
            self._run_command(post_cmd)

    def make_preprocess_command(self, run, run_dir):
        writers, workers = self.get_writers_and_workers()
        job_dir = self.job_config['job_dir']
        return ";".join(
            [
                f"cd {run_dir}",
                f"printf '{workers}\\n1\\nfort.14\\n' | {job_dir}/adcprep > {run_dir}/adcprep.log",
                f"printf '{workers}\\n2\\n' | {job_dir}/adcprep >> {run_dir}/adcprep.log",
                f"cd {job_dir}",
            ]
        )

    def make_main_command(self, run, run_dir):
        job_dir = self.job_config["job_dir"]
        writers, workers = self.get_writers_and_workers()
        exec_name = self._get_exec_name()
        if job_dir != run_dir:
            # ADCIRC and especially SWAN bug out when not run in working directory
            # This works because of how pylauncher wraps things 
            return f"cd {run_dir}; {job_dir}/{exec_name} -W {writers} &> {run_dir}/exec.log"
        else:
            return f"{job_dir}/{exec_name} -W {writers}"

    def make_postprocess_command(self, run, run_dir):
        outdir = run.get('outputs_dir')
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            command = "cp"
            if run.get("symlink_outputs"):
                command = "ln -sf"
            files = run.get("output_files", ["*.nc"])
            if type(files) is str: files = [files]
            files_str = " ".join([f"{run_dir}/{filename}" for filename in files])
            return f"{command} {files_str} {outdir}"

    def _run_command(self, command, check=True, **kwargs):
        logger.info(f"Running '{command}'")
        subprocess.run(command, shell=True, check=check, **kwargs)

    def _get_exec_name(self):
        swan = self.config.get("swan", False)
        return "padcswan" if swan else "padcirc"

    def _get_args(self):
        action_parser = ap.ArgumentParser(add_help=False)
        action_parser.add_argument("--action", default="setup", choices=["setup", "run"])
        # optional job dependency
        action_parser.add_argument("--dependency", type=str)
        action_args, _ = action_parser.parse_known_args()
        if action_args.action == "setup":
            parser = ap.ArgumentParser(parents=[action_parser])
            self.add_commandline_args(parser)
            args = parser.parse_args()
            args.action = action_args.action
            return args
        else:
            return action_args


    def add_commandline_args(self, parser):
        pass

    def get_arg(self, arg):
        return self.job_config["args"][arg]

    def _format_param(self, param):
        if type(param) is str:
            return param.format(**self.job_config["args"])
        return param

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
            "dependency": config.get("dependency"),
            "sim_config": self.config
        }

        if self.allocation is not None:
            res["allocation"] = self.allocation
        if "runtime" in config:
            res["max_run_time"] = BaseSimulator.hours_to_runtime_str(config["runtime"])

        res["args"] = self.args.__dict__.copy()

        return res

    def config_by_host(self, **kwargs):
        import socket
        hostname = socket.gethostname()
        for pattern, config in kwargs.items():
            if pattern in hostname:
                return config

        raise ValueError("Unsupported hostname '{hostname}', must contain one of {list(kwargs.keys)}.") 


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

  
    def get_writers_and_workers(self):
        node_count = self.config.get("node_count")
        procsPerNode = int(self.job_config.get("processors_per_node"))        
        totalProcs = int(node_count * procsPerNode)
        if self.config.get("no_writers"):
            writers = 0
        else:
            writers = node_count
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
        "node_count": [int, float],
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
                f"Runtime for a single run is {config['runtime']} hours, but the maximum is {maxJobRuntime} hours."
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
            input_config["node_count"] = int(math.ceil(input_config["node_count"]))
            config = self._base_job_config(**input_config)
            for i in range(len(jobRuns)):
                jobRuns[i]["index"] = i
                jobRuns[i]["global_index"] = inds[i]
            config["jobRuns"] = jobRuns
            config["jobRunInds"] = inds
            res.append(config)
        return res

    def setup_job(self):
        super().setup_job()
        os.makedirs("runs", exist_ok=True)
        ndigits = len(str(len(self.config["runs"])))
        self.run_dirs = []
        job_dir = self.job_config['job_dir']
        for run, ind in zip(self.job_config["jobRuns"], self.job_config["jobRunInds"]):
            run_dir = f"{job_dir}/runs/run{ind:0{ndigits}}"
            os.makedirs(run_dir, exist_ok=True)
            self._run_command(f"ln -sf {job_dir}/inputs/* {run_dir}")
            self.setup_run(run, run_dir)
            self.run_dirs.append(run_dir)

    def setup_run(self, run, run_dir):
        """Do any setup specific to the run.

        For parameter sweeps, this is where parameters are set.
        """

        for k, v in list(run.items()):
            run[k] = self._format_param(v)

        if "inputs_dir" in run:
            # add extra inputs/overwrite existing ones
            command = "cp" if run.get("copy_inputs") else "ln -sf"
            self._run_command(f"{command} {run['inputs_dir']}/* {run_dir}")

        if "parameters" in run:
            # TODO - edit input parameter files, performing copy-on-write
            pass

    def run_job(self):

        # Step 1 - generate Pylauncher input
        tasks = []
        job_dir = self.job_config['job_dir']

        for run, run_dir in zip(self.job_config['jobRuns'], self.run_dirs):
            # We need to redirect the output of adcprep to a file - because with subprocess.Popen
            # if a process has too much output it can cause a deadlock

            pre_process = self.make_preprocess_command(run, run_dir)
            postprocess_cmd = self.make_postprocess_command(run, run_dir)
            task = {
                    "cores": int(self.config["node_count"] * self.job_config["processors_per_node"]),
                    # the command executed in parallel
                    "main": self.make_main_command(run, run_dir),
            }

            if pre_process is not None:
                task["pre_process"] = pre_process

            if postprocess_cmd is not None:
                task["post_process"] = postprocess_cmd 

            tasks.append(task)

        # step 2 - launch Pylauncher with this input
        outfile = "pylauncher_jobs.json"
        with open(outfile, "w") as fp:
            json.dump(tasks, fp)

        IbrunLauncher(outfile, pre_post_process=True)


