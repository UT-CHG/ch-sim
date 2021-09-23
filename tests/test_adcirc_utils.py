from ch_sim import adcirc_utils as au
from ch_sim.adcirc_param_parser import ParamParser, InstructionParser
from ch_sim.parsing_instructions import fort15_instructions
import os

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

    assert int(params["NSPOOLGW"]) == 7200
    print(parser.data)
    parser.dump("tmp.out")
    assert False
