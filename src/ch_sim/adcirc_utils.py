import re
import os
import pdb
import glob
import pprint
import logging
import subprocess
import bs4 as bs
import numpy as np
import pandas as pd
import xarray as xr
import linecache as lc
import urllib.request
from functools import reduce
from geopy.distance import distance
from time import perf_counter, sleep
from contextlib import contextmanager
from .parsing_instructions import fort15_instructions
from .adcirc_param_parser import ParamParser

P_CONFIGS = {}
pd.options.display.float_format = "{:,.10f}".format
logger = logging.getLogger()


@contextmanager
def timing(label: str):
    t0 = perf_counter()
    yield lambda: (label, t1 - t0)
    t1 = perf_counter()


def get_def(param: str):
    try:
        desc = P_CONFIGS[param]
        pprint.pp(f"{param} = {desc}")
    except:
        print(f"Did not find parameter {param}'")
        pass


def update_param_configs(url: str):
    global P_CONFIGS

    P_CONFIGS = pull_param_configs(url)


def pull_param_configs(
    url: str = "https://adcirc.org/home/documentation/users-manual-v53/parameter-definitions",
):
    params = {}
    try:
        source = urllib.request.urlopen(url).read()
        rows = bs.BeautifulSoup(source, "lxml").findAll("p", {"class": "MsoNormal"})
        for row in rows:
            p_name = row.text.split()[0]
            if "(" in p_name:
                p_name = p_name.split("(")[0]
            params[p_name] = " ".join(row.text.split()[2:])
    except:
        pass

    return params


def read_param_line(out, params, f, ln=None, dtypes=None):
    if ln:
        line = lc.getline(f, ln)
    else:
        line = f.readline().strip()
    logger.info(",".join(params) + " : " + line)
    vals = [x for x in re.split("\\s+", line) if x != ""]
    for i in range(len(params)):
        try:
            if dtypes:
                out.attrs[params[i]] = dtypes[i](vals[i])
            else:
                out.attrs[params[i]] = vals[i]
        except ValueError:
            out.attrs[params[i]] = np.nan

    if ln:
        ln += 1
        return out, ln
    else:
        return out


def read_text_line(out, param, f, ln=None):
    if ln:
        line = lc.getline(f, ln).strip()
    else:
        line = f.readline().strip()
    logger.info(param + " : " + line)
    out.attrs[param] = line
    if ln:
        ln += 1
        return out, ln
    else:
        return out


def write_numeric_line(vals, f):
    line = " ".join([str(v) for v in vals])
    f.write(line + "\n")


def write_param_line(ds, params, f):
    if type(ds) == xr.Dataset:
        line = " ".join([str(ds.attrs[p]) for p in params])
    else:
        line = " ".join([str(x) for x in ds])
    if len(line) < 80:
        line += ((80 - len(line)) * " ") + "! " + ",".join(params)
    else:
        logger.warning("WARNING - fort config files shouldn't be wider than 80 cols!")
    logger.info("Writing param line for " + ",".join(params) + " - " + line)
    f.write(line + "\n")


def write_text_line(ds, param, f):
    if type(ds) == xr.Dataset:
        line = ds.attrs[param]
    else:
        line = ds
    logger.info("Writing text line for " + param + " - " + line)
    if len(line) < 80:
        if param != "":
            line += ((80 - len(line)) * " ") + "! " + param
    else:
        logger.warning("WARNING - fort config files shouldn't be wider than 80 cols!")
    f.write(line + "\n")


def find_closest(x, y, x_t, y_t):
    closest = 1e20
    closest_idx = None
    logger.info(f"Searching {len(x)} nodes for closest node to {x_t},{y_t}")
    for i, v in enumerate(zip(x, y)):

        dis = distance(v, (x_t, y_t)).km
        if dis < closest:
            closest = dis
            closest_idx = i

    logger.info(
        f"Found closest at index {closest_idx} and coordiante {x.item(closest_idx)}, {y.item(closest_idx)} with distance :{closest}"
    )
    return closest_idx


def get_latest_ts(path: str):
    if "\n" in path:
        tail = path
    else:
        res = subprocess.run(["tail", path], stdout=subprocess.PIPE)
        if res.returncode != 0:
            raise Exception("Unable to access adcirc.log file.")
        tail = res.stdout.decode("utf-8")
    for line in reversed(tail.split("\n")):
        split = re.split("TIME STEP = ", line)
        if len(split) > 1:
            ts = split[1].split(" ")[1]
            return ts
    raise Exception("adcirc.log found but time stepping hasn't started yet.")

def snatch_fort14_params(fname):
    """Read a few key parameters from the fort.14 file - not the actual mesh
    """

    res = {}
    with open(fname, "r") as fp:
        fp.readline()
        res["NE"], res["NP"] = map(int, fp.readline().strip().split()[:2])
        for i in range(res["NE"] + res["NP"]): fp.readline()
        res["NOPE"] = int(fp.readline().split()[0])
        res["NETA"] = int(fp.readline().split()[0])

    return res

def read_fort14(f14_file, ds=None):
    """read_fort14.
  Reads in ADCIRC fort.14 f14_file

  :param f14_file: Path to Python file.
  """
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    # 1 : AGRID = alpha-numeric grid identification (<=24 characters).
    ds, ln = read_text_line(ds, "AGRID", f14_file, ln=1)

    # 2 : NE, NP = number of elements, nodes in horizontal grid
    ds, ln = read_param_line(ds, ["NE", "NP"], f14_file, ln=ln, dtypes=2 * [int])

    # 3-NP : NODES
    # for k=1 to NP
    #    JN, X(JN), Y(JN), DP(JN)
    # end k loop
    logger.info("Reading Node Map.")
    ds = xr.merge(
        [
            ds,
            pd.read_csv(
                f14_file,
                delim_whitespace=True,
                nrows=ds.attrs["NP"],
                skiprows=ln - 1,
                header=None,
                names=["JN", "X", "Y", "DP"],
            )
            .set_index("JN")
            .to_xarray(),
        ],
        combine_attrs="override",
    )
    ln += ds.attrs["NP"]

    # (2+NP)-(2+NP+NE) : ELEMENTS
    # for k=1 to NE
    #    JE, NHY, NM(JE,1),NM(JE,2), NM(JE,3)
    # end k loop
    logger.info("Reading Element Map.")
    ds = xr.merge(
        [
            ds,
            pd.read_csv(
                f14_file,
                delim_whitespace=True,
                nrows=ds.attrs["NE"],
                skiprows=ln - 1,
                header=None,
                names=["JE", "NHEY", "NM_1", "NM_2", "NM_3"],
            )
            .set_index("JE")
            .to_xarray(),
        ],
        combine_attrs="override",
    )
    ln += ds.attrs["NE"]

    # (3+NP+NE) : NOPE = number of elevation specified boundary forcing segments.
    ds, ln = read_param_line(ds, ["NOPE"], f14_file, ln=ln, dtypes=[int])

    # (4+NP+NE) : NETA = total number of elevation specified boundary nodes
    ds, ln = read_param_line(ds, ["NETA"], f14_file, ln=ln, dtypes=[int])

    # Rest of the file contains boundary information. Read all at once
    bounds = pd.read_csv(
        f14_file, delim_whitespace=True, header=None, skiprows=ln - 1, usecols=[0]
    )
    bounds["BOUNDARY"] = None
    bounds["IBTYPEE"] = None
    bounds["IBTYPE"] = None
    bounds = bounds.rename(columns={0: "BOUNDARY_NODES"})

    # Get elevation sepcified boundary forcing segments
    bnd_idx = 0
    for i in range(ds.attrs["NOPE"]):
        sub = xr.Dataset()
        logger.info("Reading NOPE #" + str(i))

        # NVDLL(k), IBTYPEE(k) = number of nodes, and boundary type
        sub, ln = read_param_line(
            sub, ["NVDLL", "IBTYPEE"], f14_file, ln=ln, dtypes=2 * [int]
        )
        bounds = bounds.drop(bnd_idx)
        bounds.loc[bnd_idx : bnd_idx + sub.attrs["NVDLL"], "BOUNDARY"] = i
        bounds.loc[bnd_idx : bnd_idx + sub.attrs["NVDLL"], "IBTYPEE"] = sub.attrs[
            "IBTYPEE"
        ]
        ln += sub.attrs["NVDLL"]
        bnd_idx += sub.attrs["NVDLL"] + 1

    bounds["BOUNDARY_NODES"] = bounds["BOUNDARY_NODES"].astype(int)
    elev_bounds = bounds[["BOUNDARY", "BOUNDARY_NODES", "IBTYPEE"]].dropna()
    elev_bounds["ELEV_BOUNDARY"] = elev_bounds["BOUNDARY"].astype(int)
    elev_bounds["ELEV_BOUNDARY_NODES"] = elev_bounds["BOUNDARY_NODES"].astype(int)
    elev_bounds["IBTYPEE"] = elev_bounds["IBTYPEE"].astype(int)
    elev_bounds = elev_bounds.drop(["BOUNDARY", "BOUNDARY_NODES"], axis=1)
    ds = xr.merge(
        [ds, elev_bounds.set_index("ELEV_BOUNDARY").to_xarray()],
        combine_attrs="override",
    )

    # NBOU = number of normal flow (discharge) specified boundary segments
    bounds = bounds.drop(bnd_idx)
    bnd_idx += 1
    ds, ln = read_param_line(ds, ["NBOU"], f14_file, ln=ln, dtypes=[int])

    # NVEL = total number of normal flow specified boundary nodes
    bounds = bounds.drop(bnd_idx)
    bnd_idx += 1
    ds, ln = read_param_line(ds, ["NVEL"], f14_file, ln=ln, dtypes=[int])

    # Get flow sepcified boundary segments
    for i in range(ds.attrs["NBOU"]):
        logger.info("Reading NBOU #" + str(i))

        # NVELL(k), IBTYPE(k)
        sub, ln = read_param_line(
            sub, ["NVELL", "IBTYPE"], f14_file, ln=ln, dtypes=2 * [int]
        )
        bounds = bounds.drop(bnd_idx)
        bounds.loc[bnd_idx : bnd_idx + sub.attrs["NVELL"], "BOUNDARY"] = (
            i + ds.attrs["NOPE"]
        )
        bounds.loc[bnd_idx : bnd_idx + sub.attrs["NVELL"], "IBTYPE"] = sub.attrs[
            "IBTYPE"
        ]
        ln += sub.attrs["NVELL"]
        bnd_idx += sub.attrs["NVELL"] + 1

    normal_bounds = bounds[["BOUNDARY", "BOUNDARY_NODES", "IBTYPE"]].dropna()
    normal_bounds["NORMAL_BOUNDARY"] = (
        normal_bounds["BOUNDARY"].astype(int) - ds.attrs["NOPE"]
    )
    normal_bounds["NORMAL_BOUNDARY_NODES"] = normal_bounds["BOUNDARY_NODES"].astype(int)
    normal_bounds["IBTYPE"] = normal_bounds["IBTYPE"].astype(int)
    normal_bounds = normal_bounds.drop(["BOUNDARY", "BOUNDARY_NODES"], axis=1)
    ds = xr.merge(
        [ds, normal_bounds.set_index("NORMAL_BOUNDARY").to_xarray()],
        combine_attrs="override",
    )

    return ds

def read_fort15(f15_file, ds=None):
    """read_fort15.
    Reads in ADCIRC fort.15 f15_file

    Args:
        f15_file (str) - Path to parameter file.
        ds (dict) - a dictionary with parameter data
    """

    if ds is None:
        ds = {}

    if "NETA" not in ds:
        # we need NETA in order to parse the fort15 file
        # Try to find the fort.14 file
        dirname = os.path.dirname(f15_file)
        f14_file = dirname + "/fort.14" if dirname else "fort.14"
        if os.path.exists(f14_file):
            ds.update(snatch_fort14_params(f14_file))
        else:
            raise ValueError("NETA must be provided in order to parse the fort.15 file!"
                f"Tried and failed to find the fort.14 file in the directory '{dirname}'!")

    parser = ParamParser(fort15_instructions)
    return parser.parse(f15_file, starting_params=ds)


def read_fort13(f13_file, ds=None):
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    ds, ln = read_param_line(ds, ["AGRID"], f13_file, ln=1)

    # Note this must match NP
    ds, ln = read_param_line(ds, ["NumOfNodes"], f13_file, ln=ln, dtypes=[int])

    # Note this must be >= NWP
    ds, ln = read_param_line(ds, ["NAttr"], f13_file, ln=ln, dtypes=[int])

    # Read Nodal Attribute info
    nodals = []
    for i in range(ds.attrs["NAttr"]):
        tmp, ln = read_param_line(xr.Dataset(), ["AttrName"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["Units"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["ValuesPerNode"], f13_file, ln=ln, dtypes=[int])
        tmp, ln = read_param_line(
            tmp,
            ["v" + str(i) for i in range(tmp.attrs["ValuesPerNode"])],
            f13_file,
            ln=ln,
            dtypes=tmp.attrs["ValuesPerNode"] * [float],
        )
        nodals.append(tmp.attrs)
    ds = xr.merge(
        [ds, pd.DataFrame(nodals).set_index("AttrName").to_xarray()],
        combine_attrs="override",
    )

    # Read Non Default Nodal Attribute Values
    non_default = []
    line = lc.getline(f13_file, ln)
    while line != "":
        tmp, ln = read_param_line(tmp, ["AttrName"], f13_file, ln=ln)
        tmp, ln = read_param_line(tmp, ["NumND"], f13_file, ln=ln, dtypes=[int])

        num_vals = ds["ValuesPerNode"][ds["AttrName"] == tmp.attrs["AttrName"]].values[
            0
        ]
        cols = ["JN"] + [
            "_".join([tmp.attrs["AttrName"], str(x)]) for x in range(num_vals)
        ]
        tmp_df = pd.read_csv(
            f13_file,
            skiprows=ln - 1,
            nrows=tmp.attrs["NumND"],
            delim_whitespace=True,
            names=cols,
        )
        non_default.append(tmp_df)
        ln += tmp.attrs["NumND"]
        line = lc.getline(f13_file, ln)
    ds = xr.merge(
        [
            ds,
            reduce(lambda x, y: x.merge(y, how="outer"), non_default)
            .set_index("JN")
            .to_xarray(),
        ],
        combine_attrs="override",
    )

    return ds


def read_fort22(f22_file, NWS=12, ds=None):
    if type(ds) == xr.Dataset:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ds, _ = read_param_line(ds, ["NWSET"], f22_file, ln=1, dtypes=[float])
        ds, _ = read_param_line(ds, ["NWBS"], f22_file, ln=2, dtypes=[float])
        ds, _ = read_param_line(ds, ["DWM"], f22_file, ln=3, dtypes=[float])
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    return ds


def read_fort24(f22_file, ds=None):
    if type(ds) != xr.Dataset:
        ds = xr.Dataset()

    data = pd.read_csv(
        f22_file,
        delim_whitespace=True,
        names=["JN", "SALTAMP", "SALTPHA"],
        low_memory=False,
        header=None,
    )
    tides = data[data["SALTPHA"].isna()]
    all_tmp = []
    for i in range(int(tides.shape[0] / 4)):
        stop = (tides.index[(i + 1) * 4] - 1) if i != 7 else data.index[-1]
        tmp = data.loc[(tides.index[i * 4 + 3] + 1) : stop][
            ["JN", "SALTAMP", "SALTPHA"]
        ].copy()
        tmp["JN"] = tmp["JN"].astype(int)
        tmp["SALTAMP"] = tmp["SALTAMP"].astype(float)
        tmp["SALTPHA"] = tmp["SALTPHA"].astype(float)
        tmp["SALTFREQ"] = float(tides["JN"].iloc[i * 4 + 1])
        tmp = tmp.set_index("JN").to_xarray()
        tmp = tmp.expand_dims(dim={"SALTNAMEFR": [tides["JN"].iloc[i * 4 + 3]]})
        all_tmp.append(tmp)

    ds = xr.merge([ds, xr.concat(all_tmp, "SALTNAMEFR")], combine_attrs="override")

    return ds


def read_fort25(f25_file, NWS=12, ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ds, _ = read_param_line(ds, ["NUM_ICE_FIELDS"], f25_file, ln=1, dtypes=[float])
        ds, _ = read_param_line(
            ds, ["NUM_BLANK_ICE_SNAPS"], f25_file, ln=2, dtypes=[float]
        )
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    return ds


def read_fort221(f221_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        pressure_data = read_owi_met(f221_file, vals=["press"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"press_" + str(key): val for key, val in pressure_data.attrs.items()}
    pressure_data.attrs = attrs
    return xr.merge([ds, pressure_data], combine_attrs="no_conflicts")


def read_fort222(f222_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        wind_data = read_owi_met(f222_file, vals=["u_wind", "v_wind"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"wind_" + str(key): val for key, val in wind_data.attrs.items()}
    wind_data.attrs = attrs
    return xr.merge([ds, wind_data], combine_attrs="no_conflicts")


def read_fort225(f225_file, NWS=12, times=[], ds=None):
    if ds != None:
        if "NWS" in ds.attrs.keys():
            NWS = ds.attrs["NWS"]
    else:
        ds = xr.Dataset()

    if NWS in [12, 12012]:
        ice_data = read_owi_met(f225_file, vals=["ice_cov"], times=times)
    else:
        msg = f"NWS {NWS} Not yet implemented!"
        logger.error(msg)
        raise Exception(msg)

    attrs = {"ice_" + str(key): val for key, val in ice_data.attrs.items()}
    ice_data.attrs = attrs
    return xr.merge([ds, ice_data], combine_attrs="no_conflicts")


def read_owi_met(path, vals=["v1"], times=[0]):
    # NWS 12 - Ocean Weather Inc (OWI) met data
    attrs = {}

    # Title line:
    # 10 format (t56,i10,t71,i10)
    # read (20,10) date1,date2
    line = lc.getline(path, 1)
    attrs["source"] = line[0:56]
    attrs["start_ts"] = pd.to_datetime(line[55:66].strip(), format="%Y%m%d%H")
    attrs["end_ts"] = pd.to_datetime(line[70:80].strip(), format="%Y%m%d%H")

    if len(lc.getline(path, 2)) > 79:
        tf = "%Y%m%d%H%M%S"
        ti_idx = 67
    else:
        tf = "%Y%m%d%H"
        ti_idx = 68

    cur_line = 2
    all_data = []
    line = lc.getline(path, cur_line)
    for t in times:
        if line == "":
            break
        # Grid Spec Line:
        # 11 format (t6,i4,t16,i4,t23,f6.0,t32,f6.0,t44,f8.0,t58,f8.0,t69,i10,i2)
        # read (20,11) iLat, iLong, dx, dy, swlat, swlong, lCYMDH, iMin
        grid_spec = re.sub("[^\-0-9=.]", "", line)[1:].split("=")
        ilat = int(grid_spec[0])
        ilon = int(grid_spec[1])
        dx = float(grid_spec[2])
        dy = float(grid_spec[3])
        swlat = float(grid_spec[4])
        swlon = float(grid_spec[5])
        ts = pd.to_datetime(grid_spec[6], format=tf)

        # swlon = float(line[57:65])
        # ts = pd.to_datetime(line[68:len(line)-1], format=tf)

        logger.info(f"Processing data at {ts}")

        data = {}
        line_count = int(ilat * ilon / 8.0)
        remainder = int(ilat * ilon - (line_count * 8))
        for v in vals:
            vs = np.zeros(ilat * ilon)

            with open(path, "r") as f:
                vs[0 : (8 * line_count)] = pd.read_fwf(
                    f, nrows=line_count, skiprows=cur_line, widths=8 * [10], header=None
                ).values.flatten()
            if remainder > 0:
                with open(path, "r") as f:
                    vs[(8 * line_count) :] = pd.read_fwf(
                        f,
                        nrows=1,
                        skiprows=cur_line + line_count,
                        widths=remainder * [10],
                        header=None,
                    ).values.flatten()
                cur_line = cur_line + line_count + 1
            else:
                cur_line = cur_line + line_count

            vs = vs.reshape(1, ilat, ilon)
            data[v] = (["time", "latitude", "longitude"], vs)

        # Convert swlon to positive longitude value
        if swlon < 0:
            swlon = 360 + swlon
        lon = np.arange(start=swlon, stop=(swlon + (ilon * dx)), step=dx)
        # lon = np.where(lon<180.0,lon,lon-360)

        coords = {
            "time": [ts],
            "longitude": lon,
            "latitude": np.arange(start=swlat, stop=(swlat + (ilat * dy)), step=dy),
        }
        all_data.append(xr.Dataset(data, coords=coords))

        # Get next line corresponding to next datapoint, or empty if done
        cur_line += 1
        line = lc.getline(path, cur_line)

    if len(all_data) > 0:
        ret_ds = xr.concat(all_data, "time")
    else:
        ret_ds = xr.Dataset()

    ret_ds.attrs = attrs

    return ret_ds


def write_fort14(ds, f14_file):
    """write_fort14.
  Reads in ADCIRC fort.14 f14_file

  :param f14_file: Path to Python file.
  """
    with open(f14_file, "w") as f14:
        # 1 : AGRID = alpha-numeric grid identification (<=24 characters).
        write_text_line(ds, "AGRID", f14)

        # 2 : NE, NP = number of elements, nodes in horizontal grid
        write_param_line(ds, ["NE", "NP"], f14)

    # 3-NP : NODES
    # for k=1 to NP
    #    JN, X(JN), Y(JN), DP(JN)
    # end k loop
    logger.info("Wriing Node Map.")
    ds[["JN", "X", "Y", "DP"]].to_dataframe().to_csv(
        f14_file, sep=" ", mode="a", header=False
    )

    # (2+NP)-(2+NP+NE) : ELEMENTS
    # for k=1 to NE
    #    JE, NHY, NM(JE,1),NM(JE,2), NM(JE,3)
    # end k loop
    logger.info("Writing Element Map.")
    ds[["JE", "NHEY", "NM_1", "NM_2", "NM_3"]].to_dataframe().to_csv(
        f14_file, sep=" ", mode="a", header=False
    )

    # Elevation specified boundaries
    with open(f14_file, "a") as f14:
        # (3+NP+NE) : NOPE = number of elevation specified boundary forcing segments.
        write_param_line(ds, ["NOPE"], f14)

        # (4+NP+NE) : NETA = total number of elevation specified boundary nodes
        write_param_line(ds, ["NETA"], f14)

    for (bnd_idx, bnd) in ds[["ELEV_BOUNDARY_NODES", "IBTYPEE"]].groupby(
        "ELEV_BOUNDARY"
    ):
        with open(f14_file, "a") as f14:
            # NVDLL(k), IBTYPEE(k) = number of nodes, and boundary type
            write_param_line(
                xr.Dataset(
                    attrs={
                        "NVDLL": bnd["ELEV_BOUNDARY_NODES"].shape[0],
                        "IBTYPEE": bnd["IBTYPEE"].item(0),
                    }
                ),
                ["NVDLL", "IBTYPEE"],
                f14,
            )
        bnd["ELEV_BOUNDARY_NODES"].to_dataframe().to_csv(
            f14_file, sep=" ", mode="a", header=False, index=False
        )

    # Normal flow specified boundaries
    with open(f14_file, "a") as f14:
        # NBOU = number of normal flow (discharge) specified boundary segments
        write_param_line(ds, ["NBOU"], f14)

        # NVEL = total number of normal flow specified boundary nodes
        write_param_line(ds, ["NVEL"], f14)

    for (bnd_idx, bnd) in ds[["NORMAL_BOUNDARY_NODES", "IBTYPE"]].groupby(
        "NORMAL_BOUNDARY"
    ):
        with open(f14_file, "a") as f14:
            # NVELL(k), IBTYPE(k) = number of nodes, and boundary type
            write_param_line(
                xr.Dataset(
                    attrs={
                        "NVELL": bnd["NORMAL_BOUNDARY_NODES"].shape[0],
                        "IBTYPE": bnd["IBTYPE"].item(0),
                    }
                ),
                ["NVELL", "IBTYPE"],
                f14,
            )
        bnd["NORMAL_BOUNDARY_NODES"].to_dataframe().to_csv(
            f14_file, sep=" ", mode="a", header=False, index=False
        )


def write_fort15(ds, f15_file):
    """write_fort15.
    Writes out an ADCIRC fort.15 f15_file

    Args:
        ds (dict) - Dictionary of parameters
        f15_file (str) - output filename
    """

    parser = ParamParser(fort15_instructions)
    # Needed before writing to the file
    remove_symlink(f15_file)
    parser.dump(f15_file, data=ds)

def fix_fort_params(fname, params):
    """Read a file line by line - and make some quick-and-dirty fixes
    """
    params = params.copy()
    with open(fname, "r") as fp:
        lines = fp.readlines()

    for i in range(len(lines)):
        l = lines[i]
        if "!" not in l: continue
        parts = l.split("!")
        value = parts[0]
        comment = "!".join(parts[1:])
        for param, new_value in list(params.items()):
            if param.lower() in comment.lower():
                new_value = str(new_value).ljust(len(value), " ")
                lines[i] = new_value+"!"+comment
                del params[param]
                break

    with open(fname, "w") as fp:
        fp.write("".join(lines))

def snatch_fort_params(fname, params):
    res = {}
    with open(fname, "r") as fp:
        for line in fp:
            if "!" not in line: continue
            parts = line.split("!")
            val, comment = parts[0], "!".join(parts[1:])
            for p in params:
                if p in res: continue
                if p in comment:
                    res[p] = val.strip()

    return res
    

def create_nodal_att(name, units, default_vals, nodal_vals):
    str_vals = [f"v{str(x)}" for x in range(len(default_vals))]
    base_df = (
        pd.DataFrame(
            [[name, units, len(default_vals)] + list(default_vals)],
            columns=["AttrName", "Units", "ValuesPerNode"] + str_vals,
        )
        .set_index("AttrName")
        .to_xarray()
    )
    default_vals = (
        pd.DataFrame(
            [],
            columns=["JN"]
            + ["_".join([name, str(x)]) for x in range(len(default_vals))],
        )
        .set_index("JN")
        .to_xarray()
    )

    return xr.merge([base_df, default_vals])


def add_nodal_attribute(f13, name, units, default_vals, nodal_vals):
    if type(f13) != xr.Dataset:
        f13 = read_fort13(f13)
    if name in f13["AttrName"]:
        raise Exception(f"Error - Nodal Attribute {name} already in f13 configs.")
    new_nodal = create_nodal_att(name, units, default_vals, nodal_vals)

    df = xr.merge([f13, new_nodal], combine_attrs="override")
    df.attrs["NAttr"] = len(df["AttrName"].values)

    return df


def write_fort13(ds, f13_file):

    with open(f13_file, "w") as f13:
        write_param_line(ds, ["AGRID"], f13)

        write_param_line(ds, ["NumOfNodes"], f13)

        write_param_line(ds, ["NAttr"], f13)

        # Write Nodal Attribute info
        for attr in ds["AttrName"].values:
            write_text_line(attr, "", f13)
            write_text_line(str(ds.sel(AttrName=attr)["Units"].item(0)), "", f13)
            write_text_line(
                str(ds.sel(AttrName=attr)["ValuesPerNode"].item(0)), "", f13
            )
            def_vs = [
                "v" + str(x)
                for x in range(int(ds.sel(AttrName=[attr])["ValuesPerNode"].item(0)))
            ]
            def_vals = [
                str(x.item(0)) for x in ds.sel(AttrName=[attr])[def_vs].values()
            ]
            write_text_line(" ".join(def_vals), "", f13)

    # Write non default values
    for attr in ds["AttrName"].values:
        with open(f13_file, "a") as f13:
            write_text_line(attr, "", f13)
            cols = [
                "_".join([attr, str(x)])
                for x in range(int(ds.sel(AttrName=attr)["ValuesPerNode"].item(0)))
            ]
            out_df = ds[cols].dropna("JN").to_dataframe()
            write_text_line(str(out_df.shape[0]), "", f13)
        out_df.to_csv(f13_file, sep=" ", mode="a", header=None)


def process_adcirc_configs(path, filt="fort.*", met_times=[]):
    ds = xr.Dataset()

    # Always read fort.14 and fort.15
    adcirc_files = glob.glob(os.path.join(path, filt))
    for ff in adcirc_files:
        ftype = int(ff.split(".")[-1])
        with timing(f"Reading {ff}") as read_time:
            if ftype == 14:
                logger.info(f"Reading fort.14 file {ff}...")
                ds = read_fort14(ff, ds=ds)
            elif ftype == 15:
                logger.info(f"Reading fort.15 file {ff}...")
                ds = read_fort15(ff, ds=ds)
            elif ftype == 13:
                logger.info(f"Reading fort.13 file {ff}...")
                ds = read_fort13(ff, ds=ds)
            elif ftype == 22:
                logger.info(f"Reading fort.22 file {ff}...")
                ds = read_fort22(ff, ds=ds)
            elif ftype == 24:
                logger.info(f"Reading fort.24 file {ff}...")
                ds = read_fort24(ff, ds=ds)
            elif ftype == 25:
                logger.info(f"Reading fort.25 file {ff}...")
                ds = read_fort25(ff, ds=ds)
            elif ftype == 221:
                logger.info(f"Reading fort.221 file {ff}...")
                ds = read_fort221(ff, ds=ds, times=met_times)
            elif ftype == 222:
                logger.info(f"Reading fort.222 file {ff}...")
                ds = read_fort222(ff, ds=ds, times=met_times)
            elif ftype == 225:
                logger.info(f"Reading fort.225 file {ff}...")
                ds = read_fort225(ff, ds=ds, times=met_times)
            else:
                msg = f"Uncreognized file type = {ff}"
                logger.error(msg)
                raise Exception(msg)
        logger.info(f"Read {ff} successfully! - {read_time()[1]}")

    return ds


#P_CONFIGS = pull_param_configs()

def getlines(fname):
    with open(fname, "r") as fp: return fp.readlines()

def writelines(fname, lines):
    with open_paramfile(fname) as fp: fp.writelines(lines)

def remove_symlink(fname):
    """If a filename is a symlink - remove the symlink.
    """

    if os.path.islink(fname):
        os.unlink(fname)

def open_paramfile(fname):
    """Safely open paramfile for writing - removing a symlink if one exists.

    Often parameter files will be symlinks initially. We often need to modify the local file
    without changing the shared source/template file.
    """

    remove_symlink(fname)
    return open(fname, "w")

def set_swan_params(fname, start_date, last_date, tstep=1200):
    """Set dates and timestep in SWAN control file.

    Args:
    fname (str) - input file name - will be overwritten
    start_date (datetime.datetime) - starting date
    last_date (dateimime.datetime - The last date in the forcing data.
    Even if the simulation doesn't go that far, the 
    COMPUTE line in fort.26 needs to use the last date from the fort.22 file.
    """

    lines = getlines(fname)
    # support only hours for now
    fmt = "%Y%m%d.%H0000"
    start_str = start_date.strftime(fmt)
    last_str = last_date.strftime(fmt)
    new_lines = []
    wtiminc = start_date.strftime("%Y %m %d %H 1 0.9 1 ") + str(tstep)
    for i in range(len(lines)):
        l = lines[i]
        if "INIT HOTSTART" in l+lines[i-1]: continue
        if " SEC " in l:
            parts = l.split(" ")
            ind = parts.index("SEC")
            parts[ind-1] = str(tstep)
            if "COMPUTE" in l:
                parts[ind-2], parts[ind+1] = start_str, last_str
            else:
                parts[ind-2], parts[ind+1] = start_str, last_str
            new_lines.append(" ".join(parts)+"\n")
        elif "WTIMINC" in l:
            new_lines.append("'WTIMINC Line in ADCIRC fort.15 " + wtiminc +"'\n")
        else:
            new_lines.append(l)
    writelines(fname, new_lines)
    return wtiminc

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
