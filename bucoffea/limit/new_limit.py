#!/bin/env python
import re
import rhalphalib as rl
import numpy as np

from bucoffea.plot.util import merge_extensions, scale_xs_lumi, merge_datasets
import copy
from coffea import hist
rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False
import os

import pickle

regions  = ['sr_j','cr_1m_j','cr_2m_j','cr_1e_j','cr_2e_j','cr_g_j']
sr = 'sr_j'

def region_name(region):
    return region.replace('_','-')

def recoil_bins_2016():
    return [ 250.,  280.,  310.,  340.,  370.,  400.,
             430.,  470.,  510., 550.,  590.,  640.,
             690.,  740.,  790.,  840.,  900.,  960.,
             1020., 1090., 1160., 1250., 1400.]

def datasets(year):
    data = {
                    'cr_1m_j' : f'MET_{year}',
                    'cr_2m_j' : f'MET_{year}',
                    'cr_1e_j' : f'EGamma_{year}',
                    'cr_2e_j' : f'EGamma_{year}',
                    'cr_g_j' : f'EGamma_{year}',
                    # 'sr_j' : f'MET_{year}',
                    'sr_j' : f'nomatch',
                }
    tmp = {}
    for k, v in data.items():
        tmp[k] = re.compile(v)
    data.update(tmp)



    mc = {
                'cr_1m_j' : re.compile(f'(TTJets.*FXFX.*|Diboson.*|ST.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJet.*HT.*).*{year}'),
                'cr_1e_j' : re.compile(f'(TTJets.*FXFX.*|Diboson.*|ST.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJet.*HT.*).*{year}'),
                'cr_2m_j' : re.compile(f'(TTJets.*FXFX.*|Diboson.*|ST.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
                'cr_2e_j' : re.compile(f'(TTJets.*FXFX.*|Diboson.*|ST.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
                'cr_g_j' : re.compile(f'(GJets.*HT.*|QCD_HT.*|W.*HT.*).*{year}'),
                'sr_j' : re.compile(f'(.*WJ.*HT.*|.*ZJetsToNuNu.*HT.*|W.*HT.*|TTJets.*FXFX.*|Diboson.*|QCD_HT.*).*{year}'),
            }
    return data, mc
def prepare_histogram(acc):

    histogram = copy.deepcopy(acc['recoil'])
    newax = hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016())
    histogram = histogram.rebin(histogram.axis(newax.name), newax)
    histogram = merge_extensions(histogram, acc)
    scale_xs_lumi(histogram)
    histogram = merge_datasets(histogram)
    return histogram


def populate_non_v(model, histogram, data, mc, channels):
    """Defines regions and defines the non-leading backgrounds.

    :param rl: The rhalphalib model to populate
    :type rl: rhalphalib.model
    :param histogram: Coffea histogram containing the relevant distributions
    :type histogram: coffea.hist.Hist
    :param data: Regular expressions to identify datasets for data per region
    :type data: dict
    :param mc: Regular expressions to identify datasets for backgrounds per region
    :type mc: dict
    :param channels: Dictionary to store the channels defined here, will be changed in place.
    :type channels: dict
    """

    # Create channels and populate non-V backgrounds
    for region in regions:
        # Create Channel, add to model
        channels[region_name(region)] = rl.Channel(region_name(region))
        model.addChannel(channels[region_name(region)])

        h = histogram.integrate(histogram.axis('region'),region)

        for dataset in map(str,h.axis('dataset').identifiers()):
            is_mc = mc[region].match(dataset)
            is_data = data[region].match(dataset)

            # Skip unwanted processes
            if not (is_mc or is_data):
                continue

            # Skip V backgrounds
            if re.match('(DY|W).*(HT|LHE)', dataset):
                continue

            # Integrate to given dataset and either set
            # observation or create template sample
            template = h.integrate(h.axis('dataset'),dataset)
            if is_data:
                channels[region_name(region)].setObservation(template)
            else:
                sample_name = f'{region.replace("_","-")}_{dataset}'
                sample = rl.TemplateSample(
                                        sample_name,
                                        rl.Sample.BACKGROUND,
                                        template,
                                        )
                channels[region_name(region)].addSample(sample)

            # TODO: Nuisances

def pick_region(regex):
    '''
    Helper to select a region based on a regular expression.

    The regular expression must match exactly one of the regions
    in the list of regions, which will then be returned.
    If no matches or more than one match are found, the function
    raises an AssertionError.
    '''
    matches = [x for x in regions if re.match(regex, x)]
    assert(len(matches) == 1)
    return matches[0]

def monojet(acc, year=2017, outdir='./output'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model = rl.Model("testMonojet")

    recoil = rl.Observable('recoil', recoil_bins_2016())

    channels = {}

    # Prepare histogram
    histogram = prepare_histogram(acc)
    data, mc = datasets(year)

    # Treat trailing backgrounds first
    # (Simple template samples)
    populate_non_v(model, histogram, data, mc, channels)
    print(channels.keys())
    # Set up templates
    templates = {}
    templates['zvv'] = histogram.integrate('region',pick_region('sr_.*')).integrate('dataset',re.compile('ZJetsToNuNu.*'))
    templates['zmm'] = histogram.integrate('region',pick_region('cr_2m.*')).integrate('dataset',re.compile('ZJetsToNuNu.*'))
    templates['zee'] = histogram.integrate('region',pick_region('cr_2e.*')).integrate('dataset',re.compile('ZJetsToNuNu.*'))
    templates['wlv'] = histogram.integrate('region',pick_region('sr_.*')).integrate('dataset',re.compile('WJetsTo.*'))
    templates['wev'] = histogram.integrate('region',pick_region('cr_1m.*')).integrate('dataset',re.compile('WJetsTo.*'))
    templates['wmv'] = histogram.integrate('region',pick_region('cr_1e.*')).integrate('dataset',re.compile('WJetsTo.*'))
    templates['gj'] = histogram.integrate('region',pick_region('cr_g.*')).integrate('dataset',re.compile('GJets.*'))

    samples = {}
    zvv_yield_params = np.array([rl.IndependentParameter('tmp', b, 0, templates['zvv'].values()[()].max()*2) for b in templates['zvv'].values()[()]])

    # SR Zvv
    samples['zvv'] = rl.ParametericSample(
                                    region_name(pick_region('sr_.*'))+'_zvv',
                                    rl.Sample.BACKGROUND,
                                    recoil,
                                    zvv_yield_params)
    channels[region_name(pick_region('sr_.*'))].addSample(samples['zvv'])

    def add_tf_sample(name, region_regex, denominator):
        """Helper to quickly set up TransferFactorSample

        :param name: Name of the new process, e.g. wlv
        :type name: str
        :param region_regex: Regular expression to determine the region for this process
        :type region_regex: str
        :param denominator: Name of the process used as a denominator
        :type denominator: str
        """
        rname = region_name(pick_region(region_regex))
        samples[name] = rl.TransferFactorSample(
                                             rname+'_'+name,
                                             rl.Sample.BACKGROUND,
                                             templates[name].values()[()] / templates[denominator].values()[()],
                                             samples[denominator]
                                            )
        channels[rname].addSample(samples[name])

    add_tf_sample('wlv', 'sr_.*',   'zvv')
    add_tf_sample('wev', 'cr_1e.*', 'wlv')
    add_tf_sample('wmv', 'cr_1m.*', 'wlv')
    add_tf_sample('zee', 'cr_2e.*', 'zvv')
    add_tf_sample('zmm', 'cr_2m.*', 'zvv')
    add_tf_sample('gj',  'cr_g.*',  'zvv')


    with open(os.path.join(outdir, 'monojetModel.pkl'), "wb") as fout:
        pickle.dump(model, fout)

    # model.renderCombine(os.path.join(outdir, 'monojetModel'))
