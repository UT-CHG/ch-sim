from ch_sim import adcirc_utils as au
import os

inputs_dir = os.path.dirname(__file__) + "/inputs"

def test_wtiminc():
  data = au.read_fort15(inputs_dir+"/fort.15")
  print(data["WTIMINC"])
  swan_data = au.read_fort15(inputs_dir+"/fort.15")
  print(swan_data["WTIMINC"])
  assert False
