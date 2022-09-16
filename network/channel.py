from utils import *
from . import config

@timeit
def compute_channel_gain(distances, c=10**-3.53, frequencies=None, tx_gain=config.antennaGain):
    return c / distances ** 3.76
    # path_loss = compute_path_loss(distances, frequencies)
    # return dB2lin(tx_gain - path_loss)


def compute_path_loss(distances, frequencies=None):
    return 37.6 * np.log10(distances) + 35.3
    # return 20 * np.log10(d/1e3) + 20 * np.log10(f/1e6) + 32.44


def compute_channel_gain_costhata(ue_pos, bs_pos, frequencies, env_type='urban'):
    """
    Compute Cost Hata Path Loss between UE and BS.
    Ref[1]: https://en.wikipedia.org/wiki/Hata_model
    Ref[2]: https://en.wikipedia.org/wiki/COST_Hata_model
    """

    f = frequencies / 1e6  # in MHz
    h_B, h_M = bs_pos[:, 2], ue_pos[:, 2]

    # check feasibility
    if np.any((f > 2000) | (f < 150)):
        logging.warning(
            "Cost Hata model is designed for carrier frequency in [150, 2000] MHz, check the results obtained before using them")
    if np.any((h_M > 10) | (h_M < 1)):
        logging.warning(
            "Cost Hata model is designed for UE height in [1, 10] m, check the results obtained before using them")
    if np.any((h_B > 200) | (h_B < 30)):
        logging.warning(
            "Cost Hata model is designed for BS height in [30, 200] m, check the results obtained before using them")

    case1 = np.all((f >= 150) & (f <= 1500))
    case2 = np.all((f >= 1500) & (f <= 2000))
    assert case1 or case2, "Carrier frequencies must be all in [150, 1500] or [1500, 2000] MHz"

    # compute distance first
    dist = np.sqrt(np.sum(
        (np.expand_dims(ue_pos, 1) - bs_pos) ** 2,
        axis=-1)) / 1000

    # Mobile station antenna height correction factor
    a = 0.8 + (1.1*np.log10(f) - 0.7) * h_M - 1.56*np.log10(f)

    b = (44.9 - 6.55*np.log10(h_B)) * np.log10(dist)

    if case1:
        C_0 = 69.55
        C_f = 26.16
        C_hb = 13.82
        if env_type == 'urban':
            C_m = 0
            # if bs_frequency >= 150 and bs_frequency <= 200:
            #     a = 8.29*(np.log10(1.54*ue_pos[2])**2) - 1.1
            # else:
            #     a = 3.2*(np.log10(11.75*ue_pos[2])**2) - 4.97
        elif env_type == 'suburban':
            C_m = -2*((np.log10(f/28))**2) - 5.4
        else:
            C_m = -4.78*((np.log10(f))**2) + 18.33*np.log10(f) - 40.94
    else:
        C_0 = 46.3
        C_f = 33.9
        C_hb = 13.82
        if env_type == 'urban':
            C_m = 3
        elif env_type == 'suburban':
            C_m = 0
        else:
            raise Exception(
                "COST-HATA model is not defined for frequencies in 1500-2000MHz with RURAL environments")

    path_loss = C_0 + C_f * np.log10(f) - C_hb * np.log10(h_B) - a + b + C_m
    return dB2lin(-path_loss)
