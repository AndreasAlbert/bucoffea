#!/usr/bin/env python
import copy
from pprint import pprint
import re
from bucoffea.plot.util import klepto_load
import numpy as np
import re
from bucoffea.plot.util import klepto_load
from bucoffea.plot.util import (acc_from_dir, lumi, merge_datasets,
                                merge_extensions, scale_xs_lumi)
from matplotlib import pyplot as plt
from bucoffea.plot.stack_plot import Style
from coffea.hist.plot import plot1d, plot2d
from coffea import hist

import os
pjoin = os.path.join


data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'color':'k',
    'elinewidth': 1,
}

def get_acc(inpath, distributions):
    acc = klepto_load(inpath)
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')

    for distribution in distributions:
        acc.load(distribution)
        acc[distribution] = merge_extensions(acc[distribution], acc, reweight_pu=not ('nopu' in distribution))
        scale_xs_lumi(acc[distribution]) 
        acc[distribution] = merge_datasets(acc[distribution])
        acc[distribution].axis('dataset').sorting = 'integral'

    return acc





def plot_kinematics(acc, distributions,outdir):
    regions_to_plot = {
                "tau_pt" : "sr_vbf_no_veto_tau",
                "electron_pt_eta"  : "sr_vbf_no_veto_ele",
                "muon_pt_abseta" :  "sr_vbf_no_veto_muon",
    }

    datasets = {
        "ewk" : "EWKW.*2017",
        "qcd" : "WJetsToLNu.*2017"
    }

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for distribution in distributions:
        for dataset, regex in datasets.items():
            plt.gcf().clf()
            h = copy.deepcopy(acc[distribution])
            # pprint(list(map(str,h.axis("dataset").identifiers())))
            h = h.integrate("region",regions_to_plot[distribution])
            h = h[re.compile(regex)].integrate("dataset")
            # h = h.integrate("dataset")
            print(dataset, distribution, np.sum(h.values(overflow='over')[()]))
            h.scale(1./np.sum(h.values()[()]))

            if h.dim() == 1:
                ax = plot1d(h)
                ax.set_xlim(0,100)
                ax.set_ylabel("Fraction of veto taus")
            else:
                axname = [x.name for x in h.axes() if 'eta' in x.name][0]
                ax = plot2d(h,xaxis=axname,patch_opts={'cmap':'Greys'},clear=True)
                plot2d(h, text_opts={'color':'dodgerblue','format':'%.2f','fontsize':6,'fontweight':'bold'},patch_opts={'cmap':'Greys'},xaxis=axname, ax=ax, clear=False)
            
            for extension in ['pdf','png']:
                ax.figure.savefig(pjoin(outdir,f"{dataset}_{distribution}.{extension}"))
            
            ax.figure.clf()
    return h        

def main():
    distributions = [
                    "tau_pt",
                    "electron_pt_eta",
                    "muon_pt_abseta"
                    ]

    inpath = './input/2020-02-10_vetoweights_v4'
    acc = get_acc(inpath, distributions)


    h = plot_kinematics(acc, distributions, outdir='./output/kinematics')


if __name__ == "__main__":
    main()