from .sim import BaseSimulator, EnsembleSimulator
from . import adcirc_utils as au
import netCDF4 as nc
import os
from glob import glob

SECS_IN_DAY = 24*3600

class SegmentedSimulator(BaseSimulator):
    """Runs a single ADCIRC simulation - with stops for custom logic
    """

    def setup_job(self):
        super().setup_job()
        self.init_fort15(self.job_config["job_dir"])

    def run_job(self):
        self.sim_days = 0.0
        self.steps = 0
        while not self.done():
            self.sim_days += self.interval
            self.run_segment()
            self.steps += 1

    def add_commandline_args(self, parser):
        parser.add_argument("--run-days", required=True, type=float)
        parser.add_argument("--interval-hours", type=float)

    def done(self):
        # add fudge factor to account for roundoff error
        return self.sim_days >= self.get_arg("run_days") - 1e-6

    def run_segment(self):
        super().run_job()

    def make_preprocess_command(self, run, run_dir):
        fort15 = run_dir+"/fort.15"
        if self.steps > 0:
            # fix the fort.15 files
            hotstart_file = self.get_last_hotstart(run_dir)
            with nc.Dataset(hotstart_file) as ds:
                hotstart_days = self._get_hotstart_days(ds)
            

            new_rndy = self.interval + hotstart_days
            dt = au.snatch_fort_params(fort15, ["DT"])["DT"]
            new_params = self.get_new_rndy_params(new_rndy, dt=dt)
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
        return {"interval": dt*nhsinc/SECS_IN_DAY, "ihot": ihot, "dt": dt}

    def init_fort15(self, run_dir):
        fort15 = run_dir + "/fort.15"
        hot_params = self.get_hotstart_params(fort15)
        ihot, fort15_interval = hot_params["ihot"], hot_params["interval"]
        interval_hours = self.get_arg("interval_hours")
        if interval_hours is not None:
            self.interval = interval_hours / 24
        else:
            self.interval = fort15_interval
        # check to see if we have an existing hotstart file
        new_params = {}
        if ihot.endswith("67") or ihot.endswith("68"):
            with nc.Dataset(run_dir + "/fort."+ihot[-2:]+".nc") as ds:
                base_date = ds["time"].base_date.split("!")[0]
                new_rndy = self.interval + self._get_hotstart_days(ds)
                new_params["BASE_DATE"] = base_date
        else:
            new_rndy = self.interval
       
        new_params.update(self.get_new_rndy_params(new_rndy, dt=hot_params["dt"]))
        au.fix_fort_params(fort15, new_params)


    def get_new_rndy_params(self, rndy, dt):
        """Fix the rndy and hsinc before updating fort.15
        """

        # add a little bit to ensure the simulation will generate a hotstart file
        # round to match the format expected by ADCIRC (too many digits results in an error)
        return {
             "RND": round(rndy + 5e-3, 2),
             "NHSINC": "5 " + str(int(float(rndy) * SECS_IN_DAY / float(dt)))
        }

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

    def setup_job(self):
        EnsembleSimulator.setup_job(self)
        for run_dir in self.run_dirs:
            self.init_fort15(run_dir)

    def run_segment(self):
        # remove pylauncher temporary directories
        job_dir = self.job_config['job_dir']
        for d in glob(f"{job_dir}/pylauncher_tmp*"):
            self._run_command(f"rm -r {d}")
        EnsembleSimulator.run_job(self)
