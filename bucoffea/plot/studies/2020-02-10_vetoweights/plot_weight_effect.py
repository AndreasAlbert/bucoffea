#!/usr/bin/env python
from plot_kinematics import get_acc
from coffea.hist.plot import plot1d,plotratio
import os
import re
from bucoffea.plot.util import fig_ratio
from bucoffea.plot.stack_plot import Style
from matplotlib import pyplot as plt
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
# data_err_opts=None
style=Style()
distributions = ['mjj']


def get_ratio(num, denom):
    y = num.values()[()] / denom.values()[()]
    # print(dir(num))
    x = num.axis('mjj').centers()
    return x,y 
def plot_weight_effect(acc, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for dataset, regex in datasets.items():
        for distribution in distributions[:1]:
            for year in [2017,2018]:

                def get_hist(region):
                    h = acc[distribution]
                    h = h[re.compile(regex.format(year=year))].integrate("dataset")
                    h = h.integrate("region",region)
                    h = h.rebin("mjj", style.get_binning("mjj"))
                    return h
                h = get_hist(distribution)
                ref = get_hist("sr_vbf")
                
                
                # plot1d(h[re.compile('(nominal|ele_id).*')], overlay="variation", ax=ax, clear=False)
                # plotratio(h["sr_vbf"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = {},ax=rax)
                # plotratio(get_hist("sr_vbf"),ref,unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(get_hist("sr_vbf_no_veto_all"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(get_hist("sr_vbf_no_veto_ele"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(get_hist("sr_vbf_no_veto_tau"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(get_hist("sr_vbf_no_veto_muon"),ref,unc='num', error_opts = {'marker':'o'},ax=rax, clear=False)
                # plotratio(h["sr_vbf_no_veto_ele"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(h["sr_vbf_no_veto_muon"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)
                # plotratio(h["sr_vbf_no_veto_tau"].integrate("region"),h["sr_vbf"].integrate("region"),unc='num', error_opts = data_err_opts,ax=rax, clear=False)

                ax = plt.gca()
                ax.set_ylim(0.95,1.1)
                ax.grid(linestyle='--')
                ax.set_ylabel("Ratio to hard veto")
                ax.set_title(f"{dataset}, {year}")
                y = {}
                for region in ["sr_vbf",'sr_vbf_no_veto_all','sr_vbf_no_veto_ele','sr_vbf_no_veto_tau','sr_vbf_no_veto_muon']:
                    x, y[region] = get_ratio(get_hist(region), ref)


                ax.plot(x, y["sr_vbf"], '-o',label='Hard veto')
                ax.plot(x, y["sr_vbf_no_veto_ele"], '-o',label='Ele veto weights')
                ax.plot(x, y["sr_vbf_no_veto_tau"], '-o',label='Tau veto weights')
                ax.plot(x, y["sr_vbf_no_veto_all"], '-o',label='Full veto weights')
                ax.legend()

                for extension in ['pdf','png']:
                    ax.figure.savefig(pjoin(outdir, f'{dataset}_{distribution}_{year}.{extension}'))
                ax.figure.clf()
                # plt.close(fig)

def main():
    inpath = './input/2020-02-10_vetoweights_v8'

    acc = get_acc(inpath,distributions)

    outdir = os.path.join("./output/", inpath.split('/')[-1], "uncertainty")
    plot_weight_effect(acc, outdir=outdir)

if __name__ == "__main__":
    main()