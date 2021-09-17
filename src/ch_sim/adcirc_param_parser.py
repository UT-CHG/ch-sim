import os
import numpy as np
import re

class ParamParser():
    """A class for parsing generic ADCIRC parameter files
    """

    def __init__(self, instructions, starting_params={}):
        """Initialize the parser

        Args:
            instructions (list) - a list of parsing instructions
            starting_params (dict) - a dictionary with starting parameters
                Sometimes parameters from one file are needed to parse another file.
                For example, NETA is found in the fort.14 file, and is required to parse the fort.15 file.
        """

        self.instructions = instructions
        self.starting_params = starting_params
 
    def parse(self, fname, strict=False):
        """Parse the given file.

        Args:
            fname (str) - parameter filename
            strict (bool) - If True, then spurious parameter data on a line will result in an error.
                Use strict=True for debugging. If False (default), then extra parameter values will be ignored.
        Returns:
            data (dict) - the parsed parameters and arrays
        """

        self.data = self.starting_params.copy()
        self.comments = {}
        self.trailing_lines = []
        self.skipped_params = set()
        self.ln = 0
        self.strict = strict

        with open(fname, "r") as f:
            self.lines = [l.strip() for l in f.readlines()]

        for i in self.instructions:
            self._handle_instruction(i)

        self.trailing_lines = self.lines[self.ln:]

        return self.data

    def getline(self):
        """Return line and comment
        """

        l = self.lines[self.ln]
        self.ln += 1
        if "!" in l:
            parts = l.split("!")
            return parts[0].strip(), parts[1]
        # None for no comment
        return l.strip(), None
        

    def _handle_instruction(self, i):
        # single param

        if type(i) in [list, tuple]:
            if len(i) == 1:
                # For single parameters we don't care if there are spaces in the line
                self.data[i[0]], self.comments[i[0]] = self.getline()
                return

            params = i
            l, self.comments[tuple(i)] = self.getline()
            parts = l.split()
            if self.strict and len(parts) != len(params):
                raise ValueError(f"Expected to find the params {i} in the line '{l}'! "
                    f"Found {len(parts)} values instead of the expected {len(params)}.")
            elif not self.strict and len(parts) < len(params):
                raise ValueError(f"Expected to find the params {i} in the line '{l}'! "
                    f"Found only {len(parts)} values instead of the expected {len(params)}.")

            for param, val in zip(params, parts):
                self.data[param] = val

        # dict - general case
        elif type(i) is dict:
            if not self._check_condition(i): return
            if self._is_loop(i):
                self._handle_loop(i)
            else:
                # assume it is either a single parameter or a group of them
                self._handle_instruction(i["params"])

    def _check_condition(self, i):
        cond = i.get("condition")
        if cond is None: return True
        val = self.data[cond["param"]]
        allowed = cond["allowed"]
        try:
            val = float(val)
            allowed = [float(a) for a in allowed]
        except TypeError: 
            pass
        finally:
            if val in allowed: return True
            else:
                for p in i["params"]:
                    self.skipped_params.add(p)
                return False

    def _is_loop(self, i):
        return i.get("loop") == True

    def _prep_loop(self, loop, shape=()):
        """Preprocess a loop instruction and allocate arrays
        """

        bound_param = loop["bound"]
        bound = self.data.get(bound_param, None)
        if bound is None:
            # we don't have the bound because it was skipped by a condition
            if bound_param in self.skipped_params:
                bound = 0
            else:
                raise ValueError(f"Missing loop bound '{bound_param}'!"
                    " This is likely because it is in another file."
                    " Please provide this value to the parser by setting the starting_params keyword argument.")
        else: bound = int(bound)

        shape += (bound,)
        num_iteration_lines = 0
        active_instructions = []
        for i in loop["instructions"]:
            if type(i) is dict:
                if not self._check_condition(i): continue
                if self._is_loop(i):
                    active_instructions.append(self._prep_loop(i, shape))
                    continue
                else:
                    params = i['params']
            else:
                params = i
            if type(params) is str: params = [params]

            for p in params:
                if len(shape) == 1 and len(params) == 1:
                    # allow for general data (i.e, strings)
                    self.data[p] = np.array([None]*bound)
                else:
                    # Enforce floating point data
                    self.data[p] = np.zeros(shape)

            active_instructions.append(params)

        return {"bound": bound, "instructions": active_instructions}

    def _execute_loop(self, prepped_loop, inds=()):
        """Parse array data given a prepped loop instruction
        """

        for i in range(prepped_loop['bound']):
            iteration_inds = inds + (i,)
            for instruction in prepped_loop['instructions']:
                if type(instruction) is dict:
                    self._execute_loop(instruction, iteration_inds)
                    continue
                l, comment = self.getline()
                params = instruction
                # string data (potentially)
                if len(params) == 1 and not len(inds):
                    self.data[params[0]][iteration_inds] = l
                else:
                    for param, val in zip(params, map(float, l.split())):
                       self.data[param][iteration_inds] = val

    def _handle_loop(self, loop):
        """Parse an array of data (potentially nested)
        """
        prepped_loop = self._prep_loop(loop)

        self._execute_loop(prepped_loop)

class InstructionParser:

    def __init__(self, fname):
        """Convert a plain-text description of a parameter file into parsing instructions.

        The plain-text descriptions of parameter-files typically come from the ADCIRC documentation.
        """

        res = []
        loop_cnt = 0
        loop_stack = []
        with open(fname, "r") as fp:
            for line in fp:
                l = line.strip()
                if not len(l): continue
                if l.startswith("for"):
                    bound = l.split("to")[-1].strip()
                    loop_stack.append({"loop": True, "bound": bound, "instructions": []})
                    loop_cnt += 1
                    continue
                elif l.startswith("end"):
                    loop_cnt -= 1
                    # pop loop
                    instruction = loop_stack.pop()
                else:
                    instruction = self._parse_instruction_line(l)

                
                if loop_cnt:
                    loop_stack[-1]["instructions"].append(instruction)
                else:
                    res.append(instruction)

        self.instructions = res
                    
    def _parse_instruction_line(self, line):
        if "(" in line:
            # probably loop variables - remove the loop indices which add spurious commas
            line = re.sub(r'\([^)]*\)', '', line)
        parts = line.split(",")
        params = []
        for i, p in enumerate(parts):
            p = p.strip()
            space_ind = p.find(" ")
            if space_ind >= 0:
                condition = self._parse_condition(p[space_ind:])
                params.append(p[:space_ind])
                if condition is not None:
                    return {"params": params, "condition": condition}
            else:
                params.append(p)

        return params

    def _parse_condition(self, text):

        if "if" in text:
            # strip of punctuation
            text = text.split("if")[1].strip().strip(".,!;")
        else:
            # no conditional found
            return

        param = text.split()[0]
        for operator in ["=", "is"]:
            if operator not in text:
                continue

            try:
                vals = [float(v) for v in text.split(operator)[1].split(",")]
            # TODO - support for a wider variety of conditionals
            except (ValueError, TypeError):
                return

            return {"param": param, "allowed": vals}
