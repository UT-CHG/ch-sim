fort15_instructions = [{'params': ['RUNDES']}, {'params': ['RUNID']}, {'params': ['NFOVER']}, {'params': ['NABOUT']}, {'params': ['NSCREEN']}, {'params': ['IHOT']}, {'params': ['ICS']}, {'params': ['IM']}, {'params': ['IDEN'], 'condition': {'param': 'IM', 'allowed': [21.0]}}, {'params': ['NOLIBF']}, {'params': ['NOLIFA']}, {'params': ['NOLICA']}, {'params': ['NOLICAT']}, {'params': ['NWP']}, {'bound': 'NWP', 'instructions': [{'params': ['AttrName']}]}, {'params': ['NCOR']}, {'params': ['NTIP']}, {'params': ['NWS']}, {'params': ['NRAMP']}, {'params': ['G']}, {'params': ['TAU0']}, {'params': ['Tau0FullDomainMin', 'Tau0FullDomainMax'], 'condition': {'param': 'TAU0', 'allowed': [-5.0]}}, {'params': ['DTDP']}, {'params': ['STATIM']}, {'params': ['REFTIM']}, {'params': ['WTIMINC']}, {'params': ['RNDAY']}, {'params': ['DRAMP']}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime'], 'condition': {'param': 'NRAMP', 'allowed': [2.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux'], 'condition': {'param': 'NRAMP', 'allowed': [3.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux', 'DRAMPElev'], 'condition': {'param': 'NRAMP', 'allowed': [4.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux', 'DRAMPElev', 'DRAMPTip'], 'condition': {'param': 'NRAMP', 'allowed': [5.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux', 'DRAMPElev', 'DRAMPTip', 'DRAMPMete'], 'condition': {'param': 'NRAMP', 'allowed': [6.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux', 'DRAMPElev', 'DRAMPTip', 'DRAMPMete', 'DRAMPWRad'], 'condition': {'param': 'NRAMP', 'allowed': [7.0]}}, {'params': ['DRAMP', 'DRAMPExtFlux', 'FluxSettlingTime', 'DRAMPIntFlux', 'DRAMPElev', 'DRAMPTip', 'DRAMPMete', 'DRAMPWRad', 'DUnRampMete'], 'condition': {'param': 'NRAMP', 'allowed': [8.0]}}, {'params': ['A00', 'B00', 'C00']}, {'params': ['H0'], 'condition': {'param': 'NOLIFA', 'allowed': [0.0]}}, {'params': ['H0', 'INTEGER', 'INTEGER', 'VELMIN'], 'condition': {'param': 'NOLIFA', 'allowed': [2.0]}}, {'params': ['SLAM0', 'SFEA0']}, {'params': ['TAU'], 'condition': {'param': 'NOLIBF', 'allowed': [0.0]}}, {'params': ['CF'], 'condition': {'param': 'NOLIBF', 'allowed': [1.0]}}, {'params': ['CF', 'HBREAK', 'FTHETA', 'FGAMMA'], 'condition': {'param': 'NOLIBF', 'allowed': [2.0]}}, {'params': ['ESLM'], 'condition': {'param': 'IM', 'allowed': [0.0, 1.0, 2.0, 511113.0]}}, {'params': ['ESLM', 'ESLC'], 'condition': {'param': 'IM', 'allowed': [10.0]}}, {'params': ['CORI']}, {'params': ['NTIF']}, {'bound': 'NTIF', 'instructions': [{'params': ['TIPOTAG']}, {'params': ['TPK', 'AMIGT', 'ETRF', 'FFT', 'FACET']}]}, {'params': ['NBFR']}, {'bound': 'NBFR', 'instructions': [{'params': ['BOUNTAG']}, {'params': ['AMIG', 'FF', 'FACE']}]}, {'bound': 'NBFR', 'instructions': [{'params': ['ALPHA']}, {'bound': 'NETA', 'instructions': [{'params': ['EMO', 'EFA']}]}]}, {'params': ['ANGINN']}, {'params': ['NOUTE', 'TOUTSE', 'TOUTFE', 'NSPOOLE']}, {'params': ['NSTAE']}, {'bound': 'NSTAE', 'instructions': [{'params': ['XEL', 'YEL']}]}, {'params': ['NOUTV', 'TOUTSV', 'TOUTFV', 'NSPOOLV']}, {'params': ['NSTAV']}, {'bound': 'NSTAV', 'instructions': [{'params': ['XEV', 'YEV']}]}, {'params': ['NOUTC', 'TOUTSC', 'TOUTFC', 'NSPOOLC'], 'condition': {'param': 'IM', 'allowed': [10.0]}}, {'params': ['NSTAC'], 'condition': {'param': 'IM', 'allowed': [10.0]}}, {'bound': 'NSTAC', 'instructions': [{'params': ['XEC', 'YEC']}]}, {'params': ['NOUTM', 'TOUTSM', 'TOUTFM', 'NSPOOLM']}, {'params': ['NSTAM']}, {'bound': 'NSTAM', 'instructions': [{'params': ['XEM', 'YEM']}]}, {'params': ['NOUTGE', 'TOUTSGE', 'TOUTFGE', 'NSPOOLGE']}, {'params': ['NOUTGV', 'TOUTSGV', 'TOUTFGV', 'NSPOOLGV']}, {'params': ['NOUTGC', 'TOUTSGC', 'TOUTFGC', 'NSPOOLGC'], 'condition': {'param': 'IM', 'allowed': [10.0]}}, {'params': ['NOUTGW', 'TOUTSGW', 'TOUTFGW', 'NSPOOLGW']}, {'params': ['NFREQ']}, {'bound': 'NFREQ', 'instructions': [{'params': ['NAMEFR']}, {'params': ['HAFREQ', 'HAFF', 'HAFACE']}]}, {'params': ['THAS', 'THAF', 'NHAINC', 'FMV']}, {'params': ['NHASE', 'NHASV', 'NHAGE', 'NHAGV']}, {'params': ['NHSTAR', 'NHSINC']}, {'params': ['ITITER', 'ISLDIA', 'CONVCR', 'ITMAX']}]
