from taccjm import taccjm as tjm
from pathlib import Path
import shutil
import json
import logging

logger = logging.getLogger(__name__)


class SimulationManager:
    """An wrapper around TACCJobManager meant for working with simulations - which can be collections of jobs.
    """

    def __init__(self, system, user=None, psw=None):
        jms = tjm.list_jms()
        self.jm_id = f"ch-sim-{system}"
        if self.jm_id not in [jm['jm_id'] for jm in jms]:
            tjm.init_jm(self.jm_id, system, user=user, psw=psw)

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
            "short_desc": "",
            "long_desc": "",
            "default_queue": config.get("queue", "development"),
            "default_node_count": config.get("nodes", 1),
            "default_processors_per_node": 48,
            "default_max_run_time": "0:30:00",
            "default_memory_per_node": 128,
            "entry_script": "run.sh",
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
        app_config = tjm.deploy_app(self.jm_id, local_app_dir=tmpdir, overwrite=True)
        print(app_config)
        shutil.rmtree(tmpdir)

        job_configs = []
        for job_config in simulator.generate_job_configs(self, **config):
            config = tjm.deploy_job(self.jm_id, job_config)
            job_configs.append(config)

        njobs = len(job_configs)
        logger.info("Setup {njobs} jobs.")
        submit = self.prompt(f"Submit {njobs} jobs? [y/n]: ")
        if submit:
            for config in job_configs:
                tjm.submit_job(self.jm_id, config["job_id"])
        elif self.prompt("Save setup jobs for later?"):
            # TODO - actually save the jobs
            pass
        else:
            for config in job_configs:
                tjm.cleanup_job(self.tjm_id, job_config)
