#!/usr/bin/env python

import sys
from coffea.util import load
import uproot
import numpy as np

infiles = sys.argv[1:]

# Find region and branch names
variables = []
regions = []
for fname in infiles:
    acc = load(fname)
    
    for region in acc['tree'].keys():
        regions.append(region)
        variables.extend(acc['tree'][region].keys())


# Combine
with uproot.recreate("tree.root") as f:
    for region in set(regions):
        f[region] = uproot.newtree({x:np.float64 for x in variables})
        for fname in infiles:
            acc = load(fname)
            d = {x: acc['tree'][region][x].value for x in variables}
            f[region].extend(d)
