from .sim import BaseSimulator, EnsembleSimulator
from .adcirc_utils import fix_fort_params, snatch_fort_params
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
        job_dir = self.job_config["job_dir"]
        fort15 = job_dir + "/fort.15"
        print("Running segment", self.steps) 
        if not self.steps:
            params = snatch_fort_params(fort15, ["DTDP", "NHSINC", "IHOT"])
            ihot = params["IHOT"].strip()
            dt = float(params["DTDP"])
            nhsinc = int(params["NHSINC"].split()[-1])
            self.interval = dt*nhsinc/(24*3600)
            
            # check to see if we have an existing hotstart file
            if ihot.endswith("67") or ihot.endswith("68"):
                with nc.Dataset(job_dir + "/fort."+ihot[-2:]+".nc") as ds:
                    base_date = ds["time"].base_date.split("!")[0]
                    new_rndy = self.interval + self._get_hotstart_days(ds)
                    fix_fort_params(fort15, {"BASE_DATE": base_date, "RNDY": new_rndy})
            super().run_job()
        else:
            # fix the fort.15 files
            hotstart_file = self.get_last_hotstart()
            with nc.Dataset(hotstart_file) as ds:
                hotstart_days = self._get_hotstart_days(ds)
            new_rndy = self.interval + hotstart_days
            new_params = {"RNDY": new_rndy}
            new_params["IHOT"] = "567" if hotstart_file.endswith("67.nc") else "568"
            fix_fort_params(fort15, new_params)
            # Faster than calling adcprep
            for f in glob(job_dir + "/PE*/fort.15"):
                fix_fort_params(f, new_params)
            self._run_command("ibrun " + self.make_main_command(self.config, job_dir))
        
        self.steps += 1
            

    def get_last_hotstart(self):
        """Return the most recent hotstart file
        """
            
        job_dir = self.job_config['job_dir']
        # determine which hotstart file is more recent
        files = [job_dir+"/fort.67.nc", job_dir+"/fort.68.nc"]
        return max(files, key=os.path.getmtime)
            
    def _get_hotstart_days(self, ds):
        return ds["time"][0] / (24 * 3600)
        
