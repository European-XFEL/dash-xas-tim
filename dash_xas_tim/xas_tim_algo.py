"""
Offline data analysis and visualization tool for X-ray Absorption Spectroscopy
(XAS) at European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import re
import datetime

import numpy as np
import pandas as pd


def get_run_info(run):
    DETECTOR_SOURCE_RE = re.compile(r'(.+)/DET/(\d+)CH')
    info = ""
    first_train = run.train_ids[0]
    last_train = run.train_ids[-1]
    train_count = len(run.train_ids)
    span_sec = (last_train - first_train) / 10
    span_txt = str(datetime.timedelta(seconds=span_sec))

    detector_modules = {}
    for source in run.detector_sources:
        name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
        detector_modules[(name, modno)] = source

    # A run should only have one detector, but if that changes, don't hide it
    detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

    info += f'\n# of trains:   {train_count}'
    info += f'\nDuration:      {span_txt}'
    info += f'\nFirst train ID: {first_train}'
    info += f'\nLast train ID: {last_train}\n'

    info += f'\n{len(run.detector_sources)} detector modules ({detector_name})'
    if len(detector_modules) > 0:
        # Show detail on the first module (the others should be similar)
        mod_key = sorted(detector_modules)[0]
        mod_source = detector_modules[mod_key]
        dinfo = run.detector_info(mod_source)
        module = ' '.join(mod_key)
        dims = ' x '.join(str(d) for d in dinfo['dims'])
        info += f"\n  e.g. module {module} : {dims} pixels"
        info += (f"\n  {dinfo['frames_per_train']} frames per train, "
                 f" {dinfo['total_frames']} total frames")

    info += "\n"
    non_detector_inst_srcs = run.instrument_sources - run.detector_sources
    info += (f"\n{len(non_detector_inst_srcs)} instrument sources "
             f"(excluding detectors):")

    for d in sorted(non_detector_inst_srcs):
        info += f"\n  -{d}"
    info += "\n"
    info += f"\n{len(run.control_sources)} control sources:"
    for d in sorted(run.control_sources):
        info += f"\n  -{d}"
    return info


def compute_absorption_sigma(mu1, sigma1, mu2, sigma2, corr):
    """Compute the standard deviation for absorption.

    :param float/Series mu1: dataset 1 mean.
    :param float/Series sigma1: dataset 1 standard deviation.
    :param float/Series mu2: dataset 2 mean.
    :param float/Series sigma2: dataset 2 standard deviation.
    :param float/Series corr: correlation between dataset 1 and 2.

    :return float/Series: standard deviation of -log(mu2/mu1).
    """
    if mu1 == 0 or mu2 == 0:
        raise ValueError("mu1 and mu2 cannot be zero!")

    return np.sqrt((sigma1 / mu1) ** 2 + (sigma2 / mu2) ** 2
                   - 2 * corr * sigma1 * sigma2 / (mu1 * mu2))


def compute_absorption(I0, I1):
    """Compute absorption.

    A = -log(I1/I0)

    :param float/numpy.ndarray I0: incident intensity.
    :param float/numpy.ndarray I1: transmitted intensity.

    :return float/numpy/ndarray: absorption.
    """
    return -np.log(I1 / I0)


def compute_total_absorption(data):
    """Compute absorption for all data.

    :return: total absorption data in pandas.DataFrame with index being
        the MCP channel name and columns being:
        - muA: absorption mean;
        - sigmaA: absorption standard deviation;
        - muI0: I0 mean;
        - sigmaI0: I0 standard deviation;
        - weight: sum of I0 values;
        - muI1: I1 mean;
        - sigmaI1: I1 standard deviation;
        - corr: correlation coefficient between I0 and I1;
        - count: number of data.
    """
    absorption = pd.DataFrame(
        columns=['muA', 'sigmaA', 'muI0', 'sigmaI0', 'weight',
                 'muI1', 'sigmaI1', 'corr', 'count']
    )

    I0 = data['xgm']
    muI0 = I0.mean()
    sigmaI0 = I0.std()
    weight = I0.sum()
    count = I0.size

    for ch in ('mcp1', 'mcp2', 'mcp3'):
        I1 = data[ch]

        muI1 = I1.mean()
        sigmaI1 = I1.std()

        corr = np.corrcoef(I1, I0)[0, 1]
        absorption.loc[ch] = (
            compute_absorption(muI0, muI1),
            compute_absorption_sigma(muI0, sigmaI0, muI1, sigmaI1, corr),
            muI0, sigmaI0, weight, muI1, sigmaI1, corr, count
        )

    return absorption


def compute_spectrum(data, *, n_bins=60, point_wise=False):
    """Compute spectrum.

    :param int n_bins: number of energy bins.
    :param bool point_wise: if True, calculate the absorption point wise
        and then average. Otherwise, average over I0 and I1 first and
        then calculate the absorption. Default = False

    :return: spectrum data in pandas.DataFrame with index being the
        energy bin range and columnsbeing:
        - energy: central energy of each bin;
        - count: number of data points for each energy bin;
        - muXGM, muMCP1, muMCP2, muMCP3: intensity mean;
        - sigmaXGM, sigmaMCP1, sigmaMCP2, sigmaMCP3: intensity standard
            deviations;
        - muA1, muA2, muA3: absorption mean;
        - sigmaA1, sigmaA2, sigmaA3: absorption standard deviation;
        - corrMCP1, corrMCP2, corrMCP3: correlation between MCP and XGM.
    """
    # binning
    binned = data.groupby(pd.cut(data['energy'], bins=n_bins))

    if not point_wise:
        # mean

        binned_mean = binned.mean()
        # rename columns, e.g. 'A' -> 'muA'
        binned_mean.columns = ['mu' + col.upper() if col != 'energy' else col
                               for col in binned_mean.columns]
        # standard deviation

        binned_std = binned.std()
        # we use the "energy" column in binned_mean
        binned_std.drop("energy", axis=1, inplace=True)
        binned_std.columns = ['sigma' + col.upper()
                              for col in binned_std.columns]

        # correlation

        # calculate the correlation between 'xgm' and all the 'mcp'
        # columns for each group.
        binned_corr = binned.corr().loc[pd.IndexSlice[:, 'xgm'], :].drop(
            columns=['xgm', 'energy'], axis=1).reset_index(
            level=1, drop=True)
        binned_corr.columns = ['corr' + col.upper()
                               for col in binned_corr.columns]

        # combine all the above results
        spectrum = pd.concat(
            [binned_mean, binned_std, binned_corr], axis=1)
        spectrum['count'] = binned['energy'].count()

        # calculate absorption and its sigma for each bin
        for i, ch in enumerate(('MCP1', 'MCP2', 'MCP3'), 1):
            spectrum[f'muA{i}'] = spectrum.apply(
                lambda x: compute_absorption(x['muXGM'], x['mu' + ch]),
                axis=1)
            spectrum[f'sigmaA{i}'] = spectrum.apply(
                lambda x: compute_absorption_sigma(x['muXGM'],
                                                   x['sigmaXGM'],
                                                   x['mu' + ch],
                                                   x['sigma' + ch],
                                                   x['corr' + ch]), axis=1)
    else:
        raise NotImplementedError("point_wise=True is not implemented.")

    return spectrum
