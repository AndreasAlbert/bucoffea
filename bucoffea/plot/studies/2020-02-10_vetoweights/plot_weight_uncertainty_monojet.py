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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

pjoin = os.path.join
datasets = {
    # "ewk" : "EWKW.*{year}",
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
    x = num.axis('recoil').centers()
    return x,y 
# data_err_opts=None
import uproot
style=Style()
distributions = ['recoil_veto_weight','recoil']
def plot_weight_effect(acc, outdir,particle):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = uproot.recreate(pjoin(outdir, f"{particle}_veto_unc.root"))
    for dataset, regex in datasets.items():
        for distribution in distributions[:1]:
            for year in [2017,2018]:
                def get_hist(name, region):
                    h = acc[name]
                    h = h[re.compile(regex.format(year=year))].integrate("dataset")
                    # h = h[re.compile("sr_vbf(_no_veto.*)?$")]
                    h = h.integrate("region",region)
                    h = h.rebin("recoil", style.get_binning("recoil"))
                    return h
                h = get_hist(distribution, f"sr_j_no_veto_{particle}")
                ref = get_hist('recoil',"sr_j")
                edges  = ref.axis("recoil").edges()
                
                
                y,y2 = {},{}
                for variation in ['nominal','ele_id_up','ele_id_dn','ele_reco_up','ele_reco_dn','tau_id_up','tau_id_dn','muon_iso_dn','muon_iso_up','muon_id_dn','muon_id_up']:
                    x, y[variation] = get_ratio(h.integrate("variation",variation), ref)
                    y2[variation] = savgol_filter(y[variation],9,2)
                plt.gcf().clf()
                ax = plt.gca()
                
                ax.plot([min(x), max(x)],[1,1],'--', color='gray',label='Hard veto', linewidth=3)
                ax.plot(x, y['nominal'],'-ok', label='Nominal veto weight')

                ax.plot(x, y[f"{particle}_id_dn"],'-o',color='crimson',label=f"{particle} ID variation")
                ax.plot(x, y[f"{particle}_id_up"],'-o',color='crimson')

                try:
                    ax.plot(x, y[f"{particle}_iso_dn"],'-o',color='crimson',label=f"{particle} iso variation")
                    ax.plot(x, y[f"{particle}_iso_up"],'-o',color='crimson')
                except KeyError:
                    pass
                try:
                    ax.plot(x, y[f"{particle}_reco_dn"],'-o',color='crimson',label=f"{particle} reco variation")
                    ax.plot(x, y[f"{particle}_reco_up"],'-o',color='crimson')
                except KeyError:
                    pass

                with open(pjoin(outdir,f"uncertainty_{particle}_{dataset}_{year}.txt"),"w") as f:
                    table = []
                    f.write(f'{dataset}, {year}\n')
                    for i in range(len(x)):
                        line = [
                            i,
                            x[i],
                            100*(y['nominal'][i]-1),
                            100*np.abs(y[f'{particle}_id_up'][i]/y['nominal'][i]-1),
                        ]
                        table.append(line)
                    f.write(tabulate(table, headers=['','Recoil','Nominal',particle], floatfmt=".1f")+'\n')

                
                ax.set_ylim(0.95,1.05)
                ax.grid(linestyle='--')
                ax.legend()
                ax.set_title(f"{particle} veto, {year}")
                ax.set_xlabel("$Recoil$ (GeV)")
                ax.set_ylabel("Ratio to hard veto")
                for extension in ['pdf','png']:
                    ax.figure.savefig(pjoin(outdir, f'{dataset}_{particle}_{distribution}_{year}.{extension}'),bbox_inches='tight')
                ax.figure.clf()


                centers = edges[:-1] + 0.5*np.diff(edges)
                # print(centers)
                # print(y['tau_id_up'] / y['nominal'])


                ax=plt.gca()
                ax.clear()
                for sys in ['id','reco','iso']:
                    try:
                        unc_up = savgol_filter(y[f'{particle}_{sys}_up'] / y['nominal'],15,2)
                        unc_dn = savgol_filter(y[f'{particle}_{sys}_dn'] / y['nominal'],15,2)
                    except KeyError:
                        continue
                    edges  = ref.axis("recoil").edges()
                    outfile[f'{particle}_{sys}_veto_sys_monojet_up_{year}'] = unc_up, edges
                    outfile[f'{particle}_{sys}_veto_sys_monojet_down_{year}'] = unc_dn, edges

                    f_up = interp1d(x, unc_up, bounds_error=False, fill_value=1)
                    f_dn = interp1d(x, unc_dn, bounds_error=False, fill_value=1)

                    edges = np.array([250,300,350,400,500,600,750,1000])
                    centers = edges[:-1] + 0.5*np.diff(edges)


                    print(f_up(centers), edges)
                    outfile[f'{particle}_{sys}_veto_sys_monov_up_{year}'] = f_up(centers), edges
                    outfile[f'{particle}_{sys}_veto_sys_monov_down_{year}'] = f_dn(centers), edges

                    plt.plot(x, y[f'{particle}_{sys}_up'] / y['nominal'],'o',color='crimson',label=f"{particle} {sys} up")
                    plt.plot(x, y[f'{particle}_{sys}_dn'] / y['nominal'],'o',color='navy',label=f"{particle} {sys} down")
                    plt.plot(x, unc_up,'-',color='crimson',label="Up, smoothed")
                    plt.plot(x, unc_dn,'-',color='navy',label="Down, smoothed")
                ax.set_title(f"{particle} veto uncertainty, {year}")
                ax.set_xlabel("Recoil (GeV)")
                ax.set_ylim(0.97,1.03)
                ax.legend()
                ax.grid(linestyle="--")
                ax.figure.savefig(pjoin(outdir, f'variation_{particle}_{dataset}_{distribution}_{year}.{extension}'),bbox_inches='tight')
                # plt.close(fig)

def main():
    inpath = './input/2020-03-09_monojet_veto_weight_v1'

    acc = get_acc(inpath,distributions)
    outdir = os.path.join("./output/", inpath.split('/')[-1], "uncertainty")


    plot_weight_effect(acc, outdir=outdir, particle="tau")
    plot_weight_effect(acc, outdir=outdir, particle="muon")

if __name__ == "__main__":
    main()