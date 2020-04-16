#!/usr/bin/env python
import numpy as np
from plot_kinematics import get_acc
from coffea.hist.plot import plot1d,plotratio
import os
import re
from bucoffea.plot.util import fig_ratio
from bucoffea.plot.stack_plot import Style
from matplotlib import pyplot as plt
from tabulate import tabulate

pjoin = os.path.join
datasets = {
    "ewk" : "EWKW.*{year}",
    "qcd" : "WJetsToLNu.*{year}"
}

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    # 'color':'k',
    'elinewidth': 0,
}

def get_ratio(num, denom):
    y = num.values()[()] / denom.values()[()]
    # print(dir(num))
    x = num.axis('mjj').centers()
    return x,y 
# data_err_opts=None
style=Style()
distributions = ['mjj_veto_weight','mjj']
def plot_weight_effect(acc, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for dataset, regex in datasets.items():
        for distribution in distributions[:1]:
            for year in [2017,2018]:

                def get_hist(name, region):
                    h = acc[name]
                    h = h[re.compile(regex.format(year=year))].integrate("dataset")
                    # h = h[re.compile("sr_vbf(_no_veto.*)?$")]
                    h = h.integrate("region",region)
                    h = h.rebin("mjj", style.get_binning("mjj"))
                    return h
                h = get_hist(distribution, "sr_vbf_no_veto_all")
                ref = get_hist('mjj',"sr_vbf")
                print(ref)
                
                
                # plot1d(h[re.compile('(nominal|ele_id).*')], overlay="variation", ax=ax, clear=False)
                # plotratio(h["sr_vbf"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = {},ax=rax)
                # plotratio(h["nominal"].integrate("variation"),ref,unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(h["ele_id_up"].integrate("variation"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(h["ele_id_dn"].integrate("variation"),ref,unc='num', error_opts = {'marker':'x'},ax=rax, clear=False)
                # plotratio(h["ele_reco_up"].integrate("variation"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(h["ele_reco_dn"].integrate("variation"),ref,unc='num', error_opts = {'marker':'x'},ax=rax, clear=False)
                # plotratio(h["tau_id_up"].integrate("variation"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(h["tau_id_dn"].integrate("variation"),ref,unc='num', error_opts = {'marker':'x'},ax=rax, clear=False)

                y = {}
                for variation in ['nominal','ele_id_up','ele_id_dn','ele_reco_up','ele_reco_dn','tau_id_up','tau_id_dn','muon_iso_dn','muon_iso_up','muon_id_dn','muon_id_up']:
                    x, y[variation] = get_ratio(h[variation].integrate("variation"), ref)

                ax = plt.gca()
                ax.plot([min(x), max(x)],[1,1],'--', color='gray',label='Hard veto', linewidth=3)
                ax.plot(x, y['nominal'],'-ok', label='Nominal veto weight')
                ax.plot(x, y['ele_id_up'],'-ob',color='dodgerblue',label="Ele ID variation")
                ax.plot(x, y['ele_id_dn'],'-ob',color='dodgerblue')
                ax.plot(x, y['ele_reco_up'],'-ob',color='orange',label="Ele reco variation")
                ax.plot(x, y['ele_reco_dn'],'-ob',color='orange')
                # ax.plot(x, y['muon_id_up'],'-ob',color='green',label="Muon ID variation")
                # ax.plot(x, y['muon_id_dn'],'-ob',color='green')
                # ax.plot(x, y['muon_iso_up'],'-ob',color='green',label="Muon iso variation")
                # ax.plot(x, y['muon_iso_dn'],'-ob',color='green')
                ax.plot(x, y['tau_id_up'],'-o',color='crimson',label="Tau ID variation")
                ax.plot(x, y['tau_id_dn'],'-o',color='crimson')

                with open(pjoin(outdir,f"uncertainty_{dataset}_{year}.txt"),"w") as f:
                    table = []
                    f.write(f'{dataset}, {year}\n')
                    for i in range(len(x)):
                        line = [
                            i,
                            x[i],
                            100*np.abs(y['ele_id_up'][i]/y['nominal'][i]-1),
                            100*np.abs(y['ele_reco_up'][i]/y['nominal'][i]-1),
                            100*np.abs(y['tau_id_up'][i]/y['nominal'][i]-1),
                        ]
                        table.append(line)
                    f.write(tabulate(table, headers=['','Mjj','Ele ID','Ele Reco','Tau'], floatfmt=".1f")+'\n')
                # plotratio(h["sr_vbf_no_veto_ele"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(h["sr_vbf_no_veto_muon"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(h["sr_vbf_no_veto_tau"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                ax.set_ylim(0.95,1.15)
                ax.grid(linestyle='--')
                # rax.set_ylabel("Weight / veto")
                ax.legend()
                ax.set_title(f"{dataset}, {year}")
                ax.set_xlabel("$m_{jj}$ (GeV)")
                ax.set_ylabel("Ratio to hard veto")
                for extension in ['pdf','png']:
                    ax.figure.savefig(pjoin(outdir, f'{dataset}_{distribution}_{year}.{extension}'))
                ax.figure.clf()
                # plt.close(fig)

def main():
    inpath = './input/2020-02-10_vetoweights_v15'

    acc = get_acc(inpath,distributions)
    outdir = os.path.join("./output/", inpath.split('/')[-1], "uncertainty")


    plot_weight_effect(acc, outdir=outdir)

if __name__ == "__main__":
    main()