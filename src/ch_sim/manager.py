from taccjm import TACCJobManager
from pathlib import Path
import shutil
import json
import logging

logger = logging.getLogger(__name__)


class SimulationManager(TACCJobManager):
    """An wrapper around TACCJobManager meant for working with simulations - which can be collections of jobs.
    """

    def __init__(self, system, user=None, psw=None):
        super().__init__(system, user=user, psw=psw)

    def _setup_app_dir(self, simulator, dirname, **config):
        """Setup a local app dir for a simulation
        """

        assets_dir = dirname + "/assets"
        p = Path(assets_dir)
        if p.exists():
            shutil.rmtree(p)

        p.mkdir(exist_ok=True, parents=True)

        shutil.copy(simulator.script_file, assets_dir + "/sim.py")
        if simulator.deps is not None:
            for d in simulator.deps:
                shutil.copy(d, assets_dir)

        # now make the app.json
        app_config = {
            "name": simulator.name,
            "shortDescription": "",
            "defaultQueue": config.get("queue", "development"),
            "defaultNodeCount": config.get("nodes", 1),
            "defaultProcessorsPerNode": 48,
            "defaultMaxRunTime": "0:30:00",
            "templatePath": "run.sh",
            "inputs": [],
            "parameters": [],
            "outputs": [],
        }

        with open(dirname + "/app.json", "w") as fp:
            json.dump(app_config, fp)

        with open(dirname + "/project.ini", "w") as fp:
            pass

        with open(assets_dir + "/run.sh", "w") as fp:
            fp.write(
                "#!/bin/bash\n\n"
                'eval "$(conda shell.`basename -- $SHELL` hook)"'
                "\nconda activate ch-sim\n"
                "python3 sim.py --action=run"
            )

    def prompt(self, question):
        return input(question + " [y/n]").strip().lower() == "y"

    def setup_simulation(self, simulator, **config):
        """Setup the simulation on TACC
        """

        logger.info("Setting up simulation.")
        # setup a TACCJM application
        tmpdir = ".tmp_chsim_app"
        self._setup_app_dir(simulator, tmpdir, **config)
        app_config = self.deploy_app(local_app_dir=tmpdir, overwrite=True)
        shutil.rmtree(tmpdir)

        job_configs = []
        for job_config in simulator.generate_job_configs(self, **config):
            config = self.setup_job(job_config)
            job_configs.append(config)

        njobs = len(job_configs)
        logger.info("Setup {njobs} jobs.")
        submit = self.prompt(f"Submit {njobs} jobs? [y/n]: ")
        if submit:
            for config in job_configs:
                self.submit_job(config["job_id"])
        elif self.prompt("Save setup jobs for later?"):
            # TODO - actually save the jobs
            pass
        else:
            for config in job_configs:
                self.cleanup_job(job_config)
