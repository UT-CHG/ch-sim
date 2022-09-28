try:
    from taccjm import taccjm as tjm
    have_taccjm = True
except ImportError:
    have_taccjm = False

from pathlib import Path
import shutil
import json
import logging
import os
from datetime import datetime
import time

logger = logging.getLogger(__name__)

submit_script_template = """#!/bin/bash
#----------------------------------------------------
# {job_name} 
#
#----------------------------------------------------

#SBATCH -J {job_id}     # Job name
#SBATCH -o {job_dir}/{job_id}.o%j # Name of stdout output file
#SBATCH -e {job_dir}/{job_id}.e%j # Name of stderr error file
#SBATCH -p {queue}      # Queue (partition) name
#SBATCH -N {node_count}          # Total num nodes
#SBATCH -n {cores}          # Total num mpi tasks
#SBATCH -t {run_time}         # Run time (hh:mm:ss)
#SBATCH --ntasks-per-node {processors_per_node} # tasks per node
#SBATCH -A {allocation} # Allocation name
{extra_directives}
#----------------------------------------------------

cd {job_dir}


export NP={cores}
{job_dir}/run.sh
"""


class SimulationManager:
    """A class for submitting and setting up multiple jobs at once.
    
    It's a lightweight and local version of TACCJobManager.
    """

    def __init__(self):
        basedir = os.getenv("SCRATCH")
        if basedir is None or not os.path.exists(basedir):
            basedir = os.getenv("HOME")
        basedir = basedir + "/" + "ch-sim"
        self.jobs_dir = basedir + "/jobs"
        self.apps_dir = basedir + "/apps"
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.apps_dir, exist_ok=True)


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
            if simulator.modules is not None:
                module_str = "module load " + " ".join(simulator.modules)
            else:
                module_str = ""

            fp.write(
                "#!/bin/bash\n\n"
                #'eval "$(conda shell.`basename -- $SHELL` hook)"'
                "source ~/.bashrc\n"
                "\nconda activate ch-sim\n"
                f"{module_str}\n"
                "python3 sim.py --action=run"
            )


    def prompt(self, question):
        return input(question + " [y/n]").strip().lower() == "y"

    def get_app_dir(self, app_name):
        return self.apps_dir + "/" + app_name

    def get_job_dir(self, job_id):
        return self.jobs_dir + "/" + job_id

    def deploy_app(self, local_app_dir, **kwargs):
        with open(local_app_dir+"/app.json") as fp:
            app_config = json.load(fp)
            app_name = app_config['name']
        
        app_dir = self.get_app_dir(app_name)
        os.makedirs(app_dir, exist_ok=True)
        os.system(f"cp {local_app_dir}/assets/* {app_dir}")
        return app_config

    def deploy_job(self, job_config):
        name = job_config["name"]
        job_id = name + "_" + datetime.fromtimestamp(
                        time.time()).strftime('%Y%m%d_%H%M%S')
        while True:
            job_dir = self.get_job_dir(job_id)
            if os.path.exists(job_dir):
                job_id += "0"
            else:
                os.makedirs(job_dir)
                break

        app_dir = self.get_app_dir(job_config["app"])
        os.system(f"cp {app_dir}/* {job_dir}")
        job_config["job_id"] = job_id
        job_config["job_dir"] = job_dir
        with open(job_dir + "/job.json", "w") as fp:
            json.dump(job_config, fp)

        self.add_submit_script(job_config)
        return job_config

    def add_submit_script(self, job_config):
        fname = job_config['job_dir'] + "/submit_script.sh"
        with open(fname, "w") as fp:
            extra_directives = ""
            dependency = job_config.get("dependency")
            if dependency is not None:
                extra_directives += "\n#SBATCH --dependency=afterok:"+str(dependency)
            fp.write(
                    submit_script_template.format(
                        job_name=job_config["name"],
                        job_id=job_config["job_id"],
                        job_dir=job_config['job_dir'],
                        allocation=job_config["allocation"],
                        queue=job_config["queue"],
                        run_time=job_config["max_run_time"],
                        cores=job_config["node_count"]*job_config["processors_per_node"],
                        node_count=job_config["node_count"],
                        processors_per_node=job_config["processors_per_node"],
                        extra_directives=extra_directives,
                        )
                    )

        os.chmod(fname, 0o700)

    def submit_job(self, job_id):
        job_dir = self.get_job_dir(job_id)
        os.chmod(f"{job_dir}/run.sh", 0o700)
        print("Submitting job in ", job_dir)
        os.system(f"sbatch {job_dir}/submit_script.sh")

    def remove_job(self, job_config):
        job_dir = job_config['job_dir']
        shutil.rmtree(job_dir)

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
            updated_job_config = self.deploy_job(job_config)
            job_configs.append(updated_job_config)

        njobs = len(job_configs)
        logger.info("Setup {njobs} jobs.")
        submit = self.prompt(f"Submit {njobs} jobs? [y/n]: ")
        if submit:
            for job_config in job_configs:
                self.submit_job(job_config["job_id"])
        elif self.prompt("Save setup jobs for later?"):
            # TODO - actually save the jobs
            pass
        else:
            for job_config in job_configs:
                self.remove_job(job_config["job_id"])

class RemoteSimulationManager():
    """For dealing with simulations on a remote system.
    """

    def __init__(self, system, user=None, psw=None):
        
        if not have_taccjm:
            raise ValueError("Must have taccjm installed to launch simulations on a remote system.")

        jms = tjm.list_jms()
        self.jm_id = f"ch-sim-{system}"
        if self.jm_id not in [jm['jm_id'] for jm in jms]:
            tjm.init_jm(self.jm_id, system, user=user, psw=psw)

    def submit_job(self, job_id):
        tjm.submit_job(self.jm_id, job_id)

    def remove_job(self, job_id):
        tjm.remove_job(self.jm_id, job_id)

    def deploy_job(self, job_config):
        return tjm.deploy_job(self.jm_id, job_config)

    def deploy_app(self, local_app_dir, **kwargs):
        return tjm.deploy_app(self, local_app_dir=local_app_dir, **kwargs)

