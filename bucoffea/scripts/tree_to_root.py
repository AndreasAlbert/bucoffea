#!/usr/bin/env python

import sys
from coffea.util import load
import uproot
import numpy as np
infiles = sys.argv[1:]

for fname in infiles:
    acc = load(fname)

    with uproot.recreate("tree.root") as f:
        for region in acc['tree'].keys():
            
            rdict = acc['tree'][region]
            variables = rdict.keys()
            f[region] = uproot.newtree({x:np.float64 for x in variables})

            # for variable in variables:
            #     values = rdict[variable].value
            d = {x: rdict[x].value for x in variables}
            print(d)
            f[region].extend(d)
