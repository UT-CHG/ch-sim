from .sim import BaseSimulator, EnsembleSimulator
from . import adcirc_utils as au
import netCDF4 as nc
import os
from glob import glob

class SegmentedSimulator(BaseSimulator):
    """Runs a single ADCIRC simulation - with stops for custom logic
    """

    def run_job(self):
        self.steps = 0
        while not self.done():
            self.run_segment()

    def add_commandline_args(self, parser):
        parser.add_argument("--num-steps", required=True, type=int)

    def done(self):
        return self.steps >= self.get_arg("num_steps")

    def run_segment(self):
        if not self.steps:
            self.init_fort15(self.job_config, self.job_config['job_dir'])
        super().run_job()
        self.steps += 1

    def make_preprocess_command(self, run, run_dir):
        fort15 = run_dir+"/fort.15"
        if self.steps:
            # fix the fort.15 files
            hotstart_file = self.get_last_hotstart(run_dir)
            with nc.Dataset(hotstart_file) as ds:
                hotstart_days = self._get_hotstart_days(ds)
            
            if "interval" not in run:
                run['interval'] = self.get_hotstart_params(fort15)["interval"]

            new_rndy = run['interval'] + hotstart_days
            new_params = {"RND": self.fix_rndy(new_rndy)}
            new_params["IHOT"] = "567" if hotstart_file.endswith("67.nc") else "568"
            au.fix_fort_params(fort15, new_params)
            return au.fix_all_fort15_params_cmd(run_dir, new_params)

        else:
            return super().make_preprocess_command(run, run_dir)

    def get_hotstart_params(self, fort15):
        params = au.snatch_fort_params(fort15, ["DT", "NHSINC", "IHOT"])
        ihot = params["IHOT"].strip()
        dt = float(params["DT"])
        nhsinc = int(params["NHSINC"].split()[-1])
        return {"interval": dt*nhsinc/(24*3600), "ihot": ihot}

    def init_fort15(self, run, run_dir):
        fort15 = run_dir + "/fort.15"
        hot_params = self.get_hotstart_params(fort15)
        ihot, run['interval'] = hot_params["ihot"], hot_params["interval"]
        # check to see if we have an existing hotstart file
        if ihot.endswith("67") or ihot.endswith("68"):
            with nc.Dataset(run_dir + "/fort."+ihot[-2:]+".nc") as ds:
                base_date = ds["time"].base_date.split("!")[0]
                new_rndy = run['interval'] + self._get_hotstart_days(ds)
                au.fix_fort_params(fort15, {"BASE_DATE": base_date, "RND": self.fix_rndy(new_rndy)})
        else:
            # take one step
            au.fix_fort_params(fort15, {"RND": self.fix_rndy(run['interval'])})


    def fix_rndy(self, rndy):
        """Fix the rndy before updating fort.15
        """

        # add a little bit to ensure the simulation will generate a hotstart file
        # round to match the format expected by ADCIRC (too many digits results in an error)
        return round(rndy + 5e-3, 2)

    def get_last_hotstart(self, run_dir=None):
        """Return the most recent hotstart file
        """
        if run_dir is None:
            run_dir = self.job_config['job_dir']
        # determine which hotstart file is more recent
        files = [run_dir+"/fort.67.nc", run_dir+"/fort.68.nc"]
        return max(files, key=os.path.getmtime)
            
    def _get_hotstart_days(self, ds):
        return ds["time"][0] / (24 * 3600)


class SegmentedEnsembleSimulator(SegmentedSimulator, EnsembleSimulator):

    """A class for performing ensemble simulations in which members of the ensemble need to be hotstarted repeatedly.
    """

    def run_segment(self):
        if not self.steps:
            for run, run_dir in zip(self.job_config['jobRuns'], self.run_dirs):
                self.init_fort15(run, run_dir)
        
        # remove pylauncher temporary directories
        job_dir = self.job_config['job_dir']
        for d in glob(f"{job_dir}/pylauncher_tmp*"):
            self._run_command(f"rm -r {d}")
        EnsembleSimulator.run_job(self)
        self.steps += 1
