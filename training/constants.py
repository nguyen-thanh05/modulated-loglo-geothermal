WELL_COORDS = [
    [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
    [4, 15], [18, 4], [18, 15], [45, 15],
]
WELL_NAMES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'Inj1', 'Inj2']
N_PRODUCER_WELLS = 7
RATE_SLICE_Z = 5

BHP_MIN = 0.0
BHP_MAX = 75000.0
ENERGY_MAX = 3.0e12

TRAIN_INDEX = list(range(300))
TEST_INDEX = list(range(350, 400))

CHANNEL_NAMES = ['T_form', 'T_frac', 'P_form', 'P_frac']
BHP_NAMES = [f'BHP_{name}' for name in WELL_NAMES]
ENERGY_NAMES = [f'Energy_{name}' for name in WELL_NAMES[:N_PRODUCER_WELLS]]
