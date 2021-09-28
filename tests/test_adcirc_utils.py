from ch_sim import adcirc_utils as au
from ch_sim.adcirc_param_parser import ParamParser, InstructionParser
from ch_sim.parsing_instructions import fort15_instructions
import os
import numpy as np

inputs_dir = os.path.dirname(__file__) + "/inputs"

def test_wtiminc():
  data = au.read_fort15(inputs_dir+"/fort.15")
  print(data["WTIMINC"])
  swan_data = au.read_fort15(inputs_dir+"/fort.15")
  print(swan_data["WTIMINC"])
  assert False

def test_instruction_parser():
    p = InstructionParser(inputs_dir+"/fort15_desc.txt")
    for i in p.instructions:
        if type(i) is list:
            # Ensure we picked up the conditional for this parameter
            assert i[0] != "Tau0FullDomainMin"

def test_param_parser():
    parser = ParamParser(fort15_instructions, starting_params={'NE': 6675517, 'NP': 3352598, 'NOPE': 1, 'NETA': 292})
    params = parser.parse(inputs_dir+"/fort.15")

    # Check correct data is read
    assert int(params["NSPOOLGW"]) == 7200

    # Check that when we write data to a file and read it back in, nothing changes
    parser.dump("tmp.out")
    parser2 = ParamParser(fort15_instructions, starting_params={'NE': 6675517, 'NP': 3352598, 'NOPE': 1, 'NETA': 292})
    params2 = parser2.parse("tmp.out")

    for p, val in params.items():
        val2 = params2[p]
        if type(val) is np.ndarray:
            assert np.all(val == val2)
        elif type(val) is list:
            assert all([v1 == v2 for v1, v2 in zip(val, val2)])
        else:
            assert val == val2

