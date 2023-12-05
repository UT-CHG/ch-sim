======
ch-sim
======


A framework for running simulations with ADCIRC on HPC resources.

Description
===========

This repository contains a collection of tools to easily benchmark, prototype and run ensemble simulations
with ADCIRC. One purpose of the library is to automate common workflows and tasks involved in simulations - such as
benchmark runs, parameter scans, editing parameter files, hostarts, etc. The goal is to provide a framework for
running simulations that supports complex workflows such as MCMC, Metropolis Sampling, Bayesian inversion, etc. 

Setup on TACC
===========

While the framework should work with any slurm-type system, it is currently targeted for use on TACC's resources. Below is a guide to get started on TACC.

1. Download Miniconda for TACC ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
2. Run the installer ``bash Miniconda3-latest-Linux-x86_64.sh``. When prompted for the installation directory, be sure to specify something not in the home partition, e.g. (``$WORK/minconda3``). This will prevent conda packages from taking up space in the limited home partition. When asked if you want to initialize conda, respond 'yes'.
3. Refresh the shell by running ``source .bashrc`` and create a new Conda environment by running ``conda create ch-sim``.
4. Activate the new environment ``conda activate ch-sim`` and clone the ch-sim repo ``git clone https://github.com/UT-CHG/ch-sim.git``.
5. Install pip ``conda install pip``.
6. Install ch-sim ``cd ch-chim && pip install .``

Example Usage
=============

The simplest workfow is running a single ADCIRC simulation. Suppose we create a file named test.py below::

  from ch_sim import BaseSimulator
  import os

  # make sure to replace with your allocation here
  allocation = "ADCIRC"
  # get path to WORK directory
  workdir = os.path.expandvars("$WORK")

  if __name__ == "__main__":
    sim = BaseSimulator(allocation=allocation)

    sim.run(
        inputs_dir=workdir+"/inputs", # directory with ADCIRC input files (fort.14, ....)
        outputs_dir=workdir+"/outputs", # directory to copy outputs
        execs_dir=workdir+"/execs", # directory with padcirc and adcprep executables
        node_count=10, # number of nodes to run the job on
        processors_per_node=56, # set to the number of CPUs per node on the target system (Frontera here)
        runtime=4, # number of hours
        queue="normal"
    )



Submitting the simulation is as simple as running the above Python file.
``python3 test.py``. When the simulation completes, outputs will be copied to the directory specified via the argument ``outputs_dir``.

The second most common workflow is a parameter sweep. In this workflow, a potentially large number of ADCIRC simulations need to be run with varying inputs. Frequently, these simulations share common input files (e.g. the fort.14 mesh file, etc.). We can easily submit a batch of ADCIRC simulations with varying inputs using the EnsembleSimulator class::

  from ch_sim import EnsembleSimulator
  import os

  # make sure to replace with your allocation here
  allocation = "ADCIRC"
  # get path to WORK directory
  workdir = os.path.expandvars("$WORK")

  if __name__ == "__main__":
    sim = EnsembleSimulator(allocation=allocation)

    # Define run-specific input and output directories
    # Note that we don't need to have a full set of ADCIRC parameters in each input
    runs = [
       {
         "inputs_dir": workdir+"/harvey/inputs",
         "outputs_dir": workdir+"/harvey/outputs"
       },
       {
         "inputs_dir": workdir+"/katrina/inputs",
         "outputs_dir": workdir+"/katrina/outputs"
       },
       {
         "inputs_dir": workdir+"/ike/inputs",
         "outputs_dir": workdir+"/ike/outputs"
       }       
    ]


    sim.run(
        # directory with ADCIRC input files that are shared between runs
        # If the same file is specified in both the base input directory and a run input directory,
        # the run file takes precedence
        inputs_dir=workdir+"/inputs",
        execs_dir=workdir+"/execs", # directory with padcirc and adcprep executables
        node_count=10, # number of nodes PER INDIVIDUAL RUN
        processors_per_node=56, # set to the number of CPUs per node on the target system (Frontera here)
        runtime=4, # expected runtime for a single ADCIRC simulations
        queue="normal",
        maxJobNodes=20, # max allowed nodes per job
        maxJobRuntime=4, # max allowed runtime per job
        # if all ADCIRC runs can't complete in a single job, multiple jobs will be submitted
        # tweaking maxJobNodes and maxJobRuntime can help optimize queue wait times and throughput
        runs=runs
    )

Running this file will submit two jobs - one that runs on 20 nodes for 4 hours and completes the first two ADCIRC runs, and one that handles just a single ADICRC run. If we want all simulations in the same job, we could increase ``maxJobNodes`` to 30, or alternatively increase ``maxJobRuntime``.

Much more complex workflows are possible with the framework, including data assimilation, on-the-fly customization of the ADCIRC input files, and arbitrary pre/post processing steps. These are accomplished by subclassing the ``EnsembleSimulator`` and ``BaseSimulator`` classes. Future documentation is forthcoming. . .

Contact
===========

Please reach out to

Benjamin Pachev <benjamin.pachev@gmail.com>
Carlos del-Castillo-Negrete <cdelcastillo21@gmail.com>

to report bugs or suggest features.
