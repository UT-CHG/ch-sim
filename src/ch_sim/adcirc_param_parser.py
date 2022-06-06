import os
import numpy as np
import re

class ParsingInstruction:

    def __init__(self, params=None, condition=None, instructions=None, bound=None):
        """Parsing instruction for ADCIRC input parameter file

        Args:
            params - a list of parameter names. Provided for all but loop instructions.
            condition - a Condition that determines whether or not the instruction is executed.
            instructions (list of ParsingInstructions) - content of a loop if this is a loop.
            bound str - the parameter name indicating the loop bound.
        """

        if params is None and instructions is None:
            raise ValueError("Must either provide params or loop information!")

        self.loop = instructions is not None
        self.bound = bound
        self.params, self.condition = params, condition

        if self.loop:
            self.instructions = []
            for i in instructions:
                if isinstance(i, ParsingInstruction): self.instructions.append(i)
                else:
                    # assume a dictionary data format
                    self.instructions.append(ParsingInstruction(**i))

    def key(self):
        return tuple(self.params)

    def comment(self):
        return ", ".join(self.params)

    def __str__(self):
        if self.loop: return f"LOOP({self.bound}, "  +  " ".join([str(i) for i in self.instructions]) +")"
        else: return self.comment()


class ParamParser():
    """A class for parsing generic ADCIRC parameter files
    """

    def __init__(self, instructions):
        """Initialize the parser

        Args:
            instructions (list) - a list of parsing instructions
        """

        self.instructions = [i if isinstance(i, ParsingInstruction) else ParsingInstruction(**i) for i in instructions]
 
    def parse(self, fname, starting_params={}, strict=False):
        """Parse the given file.

        Args:
            fname (str) - parameter filename
            starting_params (dict) - a dictionary with starting parameters
                Sometimes parameters from one file are needed to parse another file.
                For example, NETA is found in the fort.14 file, and is required to parse the fort.15 file.
            strict (bool) - If True, then spurious parameter data on a line will result in an error.
                Use strict=True for debugging. If False (default), then extra parameter values will be ignored.
        Returns:
            data (dict) - the parsed parameters and arrays
        """

        self.data = starting_params.copy()
        self.comments = {}
        self.trailing_lines = []
        self.skipped_params = set()
        self.ln = 0
        self.strict = strict
        # oarsing as opposed to dumnping
        self.parsing=True

        with open(fname, "r") as f:
            self.lines = [l.strip() for l in f.readlines()]

        for i in self.instructions:
            self._parse_instruction(i)

        self.data['_trailing_lines'] = self.lines[self.ln:]
        self.data['_comments'] = self.comments.copy()

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
        

    def _parse_instruction(self, i):
        """Execute one parsing instruction (for reading)
        """

        #print("Parsing instruction", i)

        if i.loop:
            self._handle_loop(i)
        else:
            if not self._check_condition(i): return
            params = i.params
            l, self.comments[i.key()] = self.getline()
            if len(params) == 1:
                # Convert to a float if we can
                try:
                    l = float(l)
                except:
                    pass
                # For single parameters we don't care if there are spaces in the line
                self.data[params[0]] = l
                return

            parts = l.split()
            if self.strict and len(parts) != len(params):
                raise ValueError(f"Expected to find the params {i} in the line '{l}'! "
                    f"Found {len(parts)} values instead of the expected {len(params)}.")
            elif not self.strict and len(parts) < len(params):
                raise ValueError(f"Expected to find the params {i} in the line '{l}'! "
                    f"Found only {len(parts)} values instead of the expected {len(params)}.")

            for param, val in zip(params, parts):
                self.data[param] = val

    def _check_condition(self, instruction):
        cond = instruction.condition
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
                for p in instruction.params:
                    self.skipped_params.add(p)
                return False

    def _prep_loop(self, loop, shape=()):
        """Preprocess a loop instruction and allocate arrays if needed
        """

        bound_param = loop.bound
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
        for i in loop.instructions:
            if i.loop:
                active_instructions.append(self._prep_loop(i, shape))
                continue
            elif not self._check_condition(i): continue

            for p in i.params:
                if not self.parsing:
                    if p not in self.data:
                        raise ValueError(f"Missing required array '{p}' in data.")
                    elif self.data[p].shape != shape:
                        raise ValueError(f"Shape mismatch - expected shape of {shape} for variable "
                            f" '{p}', got {self.data[p].shape}")
                    continue
                if len(shape) == 1 and len(i.params) == 1:
                    # allow for general data (i.e, strings)
                    self.data[p] = np.array([None]*bound)
                else:
                    # Enforce floating point data
                    self.data[p] = np.zeros(shape)

            active_instructions.append(i.params)

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
                params = instruction

                if self.parsing:
                    l, comment = self.getline()
                    # string data (potentially)
                    if len(params) == 1 and not len(inds):
                        self.data[params[0]][iteration_inds] = l
                    else:
                        for param, val in zip(params, map(float, l.split())):
                           self.data[param][iteration_inds] = val
                else:
                    # dumping
                    self.writeline(" ".join([str(self.data[param][iteration_inds]) for param in params]))

    def _handle_loop(self, loop, parse=True):
        """Parse an array of data (potentially nested)
        """
        prepped_loop = self._prep_loop(loop)

        self._execute_loop(prepped_loop)

    def dump(self, fname, data):
        """Dump parameters to file

        Args:
            fname (str) - parameter output filename
            data (dict) - data to dump.
        """

        self.data = data
        self.comments = self.data.get("_comments", {})
        self.skipped_params = set()

        self.lines = []
        # We are in dumping mode, not parsing mode
        self.parsing = False

        for i in self.instructions:
            if i.loop:
                self._handle_loop(i)
            else:
                if not self._check_condition(i): continue
                line = " ".join([str(self.data[p]) for p in i.params])
                k = i.key()
                # It is possible for the comment to be None
                comment = self.comments.get(k, None)
                # add back in trailing whitespace just to be safe
                # This can make the difference between adcprep crashing or not
                line += " " * 10
                if comment is not None:
                    line += " ! " + comment
                else:
                    line += " ! " + i.comment()

                self.writeline(line)

        for l in self.data.get('_trailing_lines', []):
            self.writeline(l)

        with open(fname, "w") as fp:
            fp.writelines(self.lines)

    def writeline(self, l):
        # Ensure there is a newline at the end
        self.lines.append(l.strip() + "\n")

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
                    loop_stack.append({"bound": bound, "instructions": []})
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

        return {"params": params}

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
