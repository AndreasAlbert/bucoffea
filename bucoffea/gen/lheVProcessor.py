import coffea.processor as processor
import numpy as np
from coffea import hist

from bucoffea.helpers import min_dphi_jet_met, dphi
from bucoffea.helpers.dataset import (is_lo_g, is_lo_g_ewk, is_nlo_g, is_nlo_g_ewk,
                                      is_lo_w, is_lo_w_ewk, is_nlo_w,
                                      is_lo_z, is_lo_z_ewk, is_nlo_z)
from bucoffea.helpers.gen import setup_gen_candidates, setup_gen_jets
from bucoffea.helpers.genboson import find_v

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def vbf_selection(vphi, dijet, genjets):
    selection = processor.PackedSelection()

    selection.add(
                  'two_jets',
                  dijet.counts>0
                  )
    selection.add(
                  'leadak4_pt_eta',
                  (dijet.i0.pt.max() > 80) & (np.abs(dijet.i0.eta.max()) < 5.0)
                  )
    selection.add(
                  'trailak4_pt_eta',
                  (dijet.i1.pt.max() > 40) & (np.abs(dijet.i1.eta.max()) < 5.0)
                  )
    selection.add(
                  'hemisphere',
                  (dijet.i0.eta.max()*dijet.i1.eta.max() < 0)
                  )
    selection.add(
                  'mindphijr',
                  min_dphi_jet_met(genjets, vphi.max(), njet=4, ptmin=30, etamax=5.0) > 0.5
                  )
    selection.add(
                  'detajj',
                  np.abs(dijet.i0.eta-dijet.i1.eta).max() > 1
                  )
    selection.add(
                  'dphijj',
                  dphi(dijet.i0.phi,dijet.i1.phi).min() < 1.5
                  )

    return selection

def monojet_selection(vphi, genjets):
    selection = processor.PackedSelection()

    selection.add(
                  'at_least_one_jet',
                  genjets.counts>0
                  )
    selection.add(
                  'leadak4_pt_eta',
                  (genjets.pt.max() > 100) & (np.abs(genjets[genjets.pt.argmax()].eta.max()) < 2.4)
                  )
    selection.add(
                  'mindphijr',
                  min_dphi_jet_met(genjets, vphi.max(), njet=4, ptmin=30) > 0.5
                  )

    return selection


class lheVProcessor(processor.ProcessorABC):
    def __init__(self):

        # Histogram setup
        dataset_ax = Cat("dataset", "Primary dataset")

        vpt_ax = Bin("vpt",r"$p_{T}^{V}$ (GeV)", 50, 0, 2000)
        jpt_ax = Bin("jpt",r"$p_{T}^{j}$ (GeV)", 50, 0, 2000)
        mjj_ax = Bin("mjj",r"$m(jj)$ (GeV)", 75, 0, 7500)
        res_ax = Bin("res",r"pt: dressed / stat1 - 1", 80,-0.2,0.2)

        items = {}
        for tag in ['stat1','dress','lhe']:
            items[f"gen_vpt_inclusive_{tag}"] = Hist("Counts",
                                    dataset_ax,
                                    vpt_ax)
            items[f"gen_vpt_monojet_{tag}"] = Hist("Counts",
                                    dataset_ax,
                                    jpt_ax,
                                    vpt_ax)
            items[f"gen_vpt_vbf_{tag}"] = Hist("Counts",
                                    dataset_ax,
                                    jpt_ax,
                                    mjj_ax,
                                    vpt_ax)
        items["resolution"] = Hist("Counts",
                                dataset_ax,
                                res_ax)
        items['sumw'] = processor.defaultdict_accumulator(float)
        items['sumw2'] = processor.defaultdict_accumulator(float)

        self._accumulator = processor.dict_accumulator(items)

    @property
    def accumulator(self):
        return self._accumulator


    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']


        if is_lo_w(dataset) or is_nlo_w(dataset) or is_lo_w_ewk(dataset):
            wanted_boson_pdg = 24
        elif is_lo_z(dataset) or is_nlo_z(dataset) or is_lo_z_ewk(dataset):
            wanted_boson_pdg = 23
        elif is_lo_g(dataset) or is_nlo_g(dataset) or is_lo_g_ewk(dataset) or is_nlo_g_ewk:
            wanted_boson_pdg = 22

        genjets = setup_gen_jets(df)
        if wanted_boson_pdg in [23, 24]:
            boson_pt, boson_eta, boson_phi, \
            daughter_pt1, daughter_eta1, daughter_phi1, \
            daughter_pt2, daughter_eta2, daughter_phi2 = find_v(ngen=df['nGenPart'],
                                                gen_pdg=df['GenPart_pdgId'],
                                                gen_pt=df['GenPart_pt'],
                                                gen_eta=df['GenPart_eta'],
                                                gen_phi=df['GenPart_phi'],
                                                gen_status=df['GenPart_status'],
                                                gen_mass=df['GenPart_mass'],
                                                gen_mother=df['GenPart_genPartIdxMother'],
                                                ndressed=df['nGenDressedLepton'],
                                                dressed_pdg=df['GenDressedLepton_pdgId'],
                                                dressed_pt=df['GenDressedLepton_pt'],
                                                dressed_eta=df['GenDressedLepton_eta'],
                                                dressed_phi=df['GenDressedLepton_phi'],
                                                wanted_boson_pdg=wanted_boson_pdg
                                                )
            # Clean the generator jets against daughter leptons
            matched = (np.hypot(genjets.phi-daughter_phi1, genjets.eta-daughter_eta1) < 0.4) \
                    (np.hypot(genjets.phi-daughter_phi2, genjets.eta-daughter_eta2) < 0.4)
            genjets = genjets[~matched]
        elif wanted_boson_pdg == 22:
            gen = setup_gen_candidates(df)
            photons = gen[(gen.pdg==22)&(gen.status==1)]
            lead_photon_index = photons.pt.argmax()
            boson_pt = photons.pt[lead_photon_index]
            boson_eta = photons.eta[lead_photon_index]
            boson_phi = photons.phi[lead_photon_index]

        for i in range(50):
            print(f'{i} {boson_phi[i]:.2f} {boson_pt[i]:.2f}')

        # Dijet for VBF
        dijet = genjets[:,:2].distincts()
        tags = ['revamp']
        for tag in tags:
            # Selection
            vbf_sel = vbf_selection(boson_phi, dijet, genjets)
            monojet_sel = monojet_selection(boson_phi, genjets)

            nominal = df['Generator_weight']

            output[f'gen_vpt_inclusive_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=boson_pt,
                                    weight=nominal
                                    )

            mask_vbf = vbf_sel.all(*vbf_sel.names)
            output[f'gen_vpt_vbf_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=boson_pt[mask_vbf],
                                    jpt=genjets.pt.max()[mask_vbf],
                                    mjj = dijet.mass.max()[mask_vbf],
                                    weight=nominal[mask_vbf]
                                    )

            mask_monojet = monojet_sel.all(*monojet_sel.names)

            output[f'gen_vpt_monojet_{tag}'].fill(
                                    dataset=dataset,
                                    vpt=boson_pt[mask_monojet],
                                    jpt=genjets.pt.max()[mask_monojet],
                                    weight=nominal[mask_monojet]
                                    )

        # Keep track of weight sum
        output['sumw'][dataset] +=  df['genEventSumw']
        output['sumw2'][dataset] +=  df['genEventSumw2']
        return output

    def postprocess(self, accumulator):
        return accumulator
