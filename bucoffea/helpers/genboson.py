import numba
import numpy as np

@numba.jit
def possible_daughter(gen_status, gen_pdg, iev, i):
    """
    Does the given status and PDG ID represent a possible boson daughter?
    """
    # Status 1 e, mu
    valid = (gen_status[iev][i]==1) and ((gen_pdg[iev][i]==11) or (gen_pdg[iev][i]==13))

    # Status 1 neutrinos
    valid |= (gen_status[iev][i]==1) and ((gen_pdg[iev][i]==12) or (gen_pdg[iev][i]==14) or (gen_pdg[iev][i]==16))

    # Status 2 taus
    valid |= (gen_status[iev][i]==2) and (gen_pdg[iev][i]==15)

    return valid

@numba.jit
def inv_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    """Invariant mass as a function of pt, eta, phi"""
    return np.sqrt(2 * pt1 * pt2 * (np.cosh(eta2-eta1) - np.cosh(phi2-phi1)))

@numba.jit
def vector_sum(pt1, eta1, phi1, pt2, eta2, phi2):
    x = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    y = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    z = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)

    pt = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    eta = np.arctanh( z / np.sqrt(x**2, y**2, z**2))

    return pt, eta, phi

@numba.jit
def find_v( ngen,     gen_pdg,     gen_pt,     gen_eta,     gen_phi, gen_status, gen_mass, gen_mother,
            ndressed, dressed_pdg, dressed_pt, dressed_eta, dressed_phi,
            wanted_boson_pdg):
    """Function to determine the generator boson and its leptonic daughters

    A multi-tier approach is taken:

    1. Try to find a generator boson with the right properties in the collection
    of generator particles. The search is performed by first finding all valid
    decay leptons, and tracing their history backwards to see if it contains the
    desired boson. If such a boson is found, search for its leptonic
    daughters and return them. If multiple bosons are found, the one with highest
    mass is picked.

    2. If no boson is found, revert to building dilepton candidates. This approach
    is split by whether we want to find a Z or W boson.
    
        2.1 For Z bosons, attempt to:
            
            2.1.1 Build a dilepton from dressed e / mu candidates.

            2.1.2 Build a dilepton from status 1 neutrinos, electrons, 
                  muons or status 2 taus.
    
        For each of these attempts, the consistency of the lepton flavor is required, 
        i.e. their PDG IDs must sum to zero.

        2.2 For W bosons, attempt to:

            2.2.1 Build a dilepton from dressed e / mu candidates and status 1 neutrinos.

            2.2.2 Build a dilepton from status 1 electrons, muons and neutrinos, 
                  as well as status 2 taus.

        For each of these attempts, the consistency of the lepton flavor is required, 
        i.e. their PDG IDs must have opposite signs and the two particle must form 
        a valid neutrino-charged lepton pair.

    If in any case multiple boson or dilepton candidates are identified, 
    the one with highest mass is chosen.
    """

    # Arrays to hold the kinematic values of the daughter leptons
    # These will be returned by this function
    all_pt1 = np.array(len(ngen))
    all_eta1 = np.array(len(ngen))
    all_phi1 = np.array(len(ngen))
    all_pt2 = np.array(len(ngen))
    all_eta2 = np.array(len(ngen))
    all_phi2 = np.array(len(ngen))

    # Event loop
    for iev in len(ngen):
        # Kinematic values of the leptons of this event
        pt1 = -1;
        eta1 = -1;
        phi1 = -1;
        pt2 = -1;
        eta2 = -1;
        phi2 = -1



        # Algorithm step 1: Attempt to find boson in generator history
        mothers = []
        daughters = []
        for i in range(ngen[iev]):

            # Start by finding all valid decay leptons
            if not possible_daughter(gen_status, gen_pdg, iev, i):
                continue

            # Trace the history of the particles
            # and see if it contains the boson we want
            parent = i
            while parent > 0:
                parent = gen_mother[parent]
                parent_pdg = gen_pdg[parent]
                if abs(parent_pdg) == wanted_boson_pdg:
                    mothers.append(parent)
                    break

        # If multiple bosons have been found,
        # choose the one with highest mass
        best_mother = None
        current_highest_mass = -1
        for mother in mothers:
            if gen_mass[mother] > current_highest_mass:
                best_mother = mother
                current_highest_mass = gen_mass[mother]

        # Finally, find the decay leptons
        if best_mother:
            for i in range(ngen[iev]):
                if not possible_daughter(gen_status, gen_pdg, iev, i):
                        continue
                parent = i

                is_daughter = False
                while parent > 0:
                    parent = gen_mother[parent]
                    if parent == best_mother:
                        is_daughter = True
                        break
                if is_daughter:
                    if pt1 < 0:
                        pt1 = gen_pt[iev][i]
                        eta1 = gen_eta[iev][i]
                        phi1 = gen_phi[iev][i]
                    elif pt2 < 0:
                        pt2 = gen_pt[iev][i]
                        eta2 = gen_eta[iev][i]
                        phi2 = gen_phi[iev][i]
                    else:
                        raise
        # Algorithm step 2: No boson has been found,
        # so we try to build dilepton candidates
        else:

            # The exact implementation differs by the type
            # of boson we are trying to find.
            # Step 2.1: Z bosons
            if wanted_boson_pdg == 23:
                best_mass = -1

                # Step 2.1.1: Dilepton from dressed electrons, muons
                for i in range(ndressed[iev]):
                    for j in range(i, ndressed[iev]):
                        if dressed_pdg[iev][i] + dressed_pdg[iev][j] > 0:
                            continue
                        mass = 2 * dressed_pt[iev][i] * dressed_pt[iev][j] \
                                * np.sinh(dressed_phi[iev][i] - dressed_phi[iev][j]) \
                                * np.cosh(dressed_eta[iev][i] - dressed_eta[iev][j])
                        if mass > best_mass:
                            best_mass = mass
                            pt1 = dressed_pt[iev][i]
                            pt2 = dressed_pt[iev][j]
                            phi1 = dressed_phi[iev][i]
                            phi2 = dressed_phi[iev][j]
                            eta1 = dressed_eta[iev][i]
                            eta2 = dressed_eta[iev][j]

                # Step 2.1.2: Dilepton from naked electrons, muons, taus, neutrinos
                for i in range(ngen[iev]):
                    if not possible_daughter(gen_status, gen_pdg, iev, i):
                        continue
                    for j in range(i, ngen[iev]):
                        if not possible_daughter(gen_status, gen_pdg, iev, j):
                            continue
                        if gen_pdg[i] + gen_pdg[j] !=0:
                            continue
                        mass = inv_mass(
                                        gen_pt[iev][i], gen_eta[iev][i], gen_phi[iev][i],
                                        gen_pt[iev][j], gen_eta[iev][j], gen_phi[iev][j]
                                        )
                        if mass > best_mass:
                            best_mass = mass
                            pt1 = gen_pt[iev][i]
                            pt2 = gen_pt[iev][j]
                            phi1 = gen_phi[iev][i]
                            phi2 = gen_phi[iev][j]
                            eta1 = gen_eta[iev][i]
                            eta2 = gen_eta[iev][j]
            # Step 2.2: W bosons
            elif abs(wanted_boson_pdg) == 24:
                best_mass = -1
                
                # Step 2.2.1: Dilepton from dressed electrons, muons + naked neutrinos
                for i in range(ndressed[iev]):
                    for j in range(ngen[iev]):
                        # Only consider status 1 neutrinos
                        if gen_status[iev][j] != 1:
                            continue

                        # Check that the pair has compatible IDs
                        # We want charged e / mu + matching neutrino
                        valid =   (abs(dressed_pdg[iev][i])==11 and abs(gen_pdg[iev][j])==12 ) \
                                & (abs(dressed_pdg[iev][i])==13 and abs(gen_pdg[iev][j])==14 ) \
                                & (dressed_pdg[iev][i] * gen_pdg[iev][j] < 0)
                        if not valid:
                            continue

                        mass = inv_mass(
                                        dressed_pt[iev][i], dressed_eta[iev][i], dressed_phi[iev][i],
                                        gen_pt[iev][j], gen_eta[iev][j], gen_phi[iev][j]
                                        )
                        if mass > best_mass:
                            best_mass = mass
                            pt1 = dressed_pt[iev][i]
                            pt2 = gen_pt[iev][j]
                            phi1 = dressed_phi[iev][i]
                            phi2 = gen_phi[iev][j]
                            eta1 = dressed_eta[iev][i]
                            eta2 = gen_eta[iev][j]

                # Step 2.2.1: Dilepton from naked electrons, muons, taus, neutrinos
                for i in range(ngen[iev]):
                    if not possible_daughter(gen_status, gen_pdg, iev, i):
                        continue
                    for j in range(i, ngen[iev]):
                        if not possible_daughter(gen_status, gen_pdg, iev, j):
                            continue

                    # Check that the pair has compatible IDs
                    # We want charged e / mu + matching neutrino
                    valid =   (abs(gen_pdg[iev][i])==11 and abs(gen_pdg[iev][j])==12 ) \
                            & (abs(gen_pdg[iev][i])==13 and abs(gen_pdg[iev][j])==14 ) \
                            & (abs(gen_pdg[iev][i])==15 and abs(gen_pdg[iev][j])==16 ) \
                            & (dressed_pdg[iev][i] * gen_pdg[iev][j] < 0)
                    if not valid:
                        continue

                    mass = inv_mass(
                                        gen_pt[iev][i], gen_eta[iev][i], gen_phi[iev][i],
                                        gen_pt[iev][j], gen_eta[iev][j], gen_phi[iev][j]
                                        )
                    if mass > best_mass:
                        best_mass = mass
                        pt1 = gen_pt[iev][i]
                        pt2 = gen_pt[iev][j]
                        phi1 = gen_phi[iev][i]
                        phi2 = gen_phi[iev][j]
                        eta1 = gen_eta[iev][i]
                        eta2 = gen_eta[iev][j]
        
        # Add the values of this event to the arrays
        all_pt1[iev] = pt1
        all_eta1[iev] = eta1
        all_phi1[iev] = phi1
        all_pt2[iev] = pt2
        all_eta2[iev] = eta2
        all_phi2[iev] = phi2

    boson_pt, boson_eta, boson_phi = vector_sum(
                                                all_pt1, all_eta1, all_phi1, 
                                                all_pt2, all_eta2, all_phi2
                                                )
    return boson_pt, boson_eta, boson_phi, all_pt1, all_eta1, all_phi1, all_pt2, all_eta2, all_phi2