from pycbc.types.timeseries import TimeSeries, FrequencySeries
from pycbc.filter.matchedfilter import match

def pycbc_match(h1, h2):

    matches = []
    for i in range(h1.shape[0]):

        if 1==1:
            ts1 = TimeSeries(h1[i, 0].double().numpy(), delta_t=1/2048)
            ts2 = TimeSeries(h2[i, 0].double().numpy(), delta_t=1/2048)
            #fs = FrequencySeries(psds[i, 0].double().numpy(), delta_f=1/5)
            matches.append(match(ts1, ts2, low_frequency_cutoff=20, return_phase=False)[0])

    return matches