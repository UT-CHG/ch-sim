from ch_sim import adcirc_utils as au
from ch_sim.adcirc_param_parser import ParamParser, InstructionParser
from ch_sim.parsing_instructions import fort15_instructions
import os
import numpy as np

inputs_dir = os.path.dirname(__file__) + "/inputs"

def test_wtiminc():
  data = au.read_fort15(inputs_dir+"/fort.15", ds={'NETA': 292})
  assert data["WTIMINC"] == 900
  swan_data = au.read_fort15(inputs_dir+"/swan/fort.15", ds={'NETA': 292})
  assert swan_data["WTIMINC"].endswith("1200")

def test_instruction_parser():
    p = InstructionParser(inputs_dir+"/fort15_desc.txt")
    for i in p.instructions:
        if type(i) is list:
            # Ensure we picked up the conditional for this parameter
            assert i[0] != "Tau0FullDomainMin"

def assert_params_equal(params, params2):
    for p, val in params.items():
        # meta-data like comments
        if p.startswith("_"): continue
        val2 = params2[p]
        if type(val) is np.ndarray:
            assert np.all(val == val2)
        else:
            assert val == val2

def test_fort15_readwrite():
    ds = {'NETA': 292}
    data = au.read_fort15(inputs_dir+"/fort.15", ds=ds)
    data["RNDAY"] = -1
    # create a symlink
    os.system(f"ln -s {inputs_dir}/fort.15 tmp.link")
    au.write_fort15(data, "tmp.link")
    data2 = au.read_fort15("tmp.link", ds=ds)
    assert_params_equal(data, data2)
    # Make sure we didn't modify the original data by chance
    data = au.read_fort15(inputs_dir+"/fort.15", ds=ds)
    assert data["RNDAY"] > 0
    os.remove("tmp.link")

def test_param_parser():
    parser = ParamParser(fort15_instructions)
    params = parser.parse(inputs_dir+"/fort.15",
        starting_params={'NOPE': 1, 'NETA': 292})

    # Check correct data is read
    assert int(params["NSPOOLGW"]) == 7200

    # Check that when we write data to a file and read it back in, nothing changes
    parser.dump("tmp.out", params)
    parser2 = ParamParser(fort15_instructions)
    params2 = parser2.parse("tmp.out",
        starting_params={'NOPE': 1, 'NETA': 292})

    assert_params_equal(params, params2)
    os.remove("tmp.out")
