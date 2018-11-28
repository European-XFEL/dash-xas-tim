"""
Offline data analysis and visualization tool for Xray Absorption Spectroscopy
(XAS) experiments at SCS, European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from collections import OrderedDict
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from karabo_data import RunDirectory


def find_peaks(trace, n_peaks, peak_start, peak_width, 
               background_end, background_width, peak_interval):
    """Return a list of peaks.
    
    :param xarray trace: trace for all the trains. shape = (trains, samples)
    :param int n_peaks: number of expected peaks.
    :param int peak_start: start position of the first peak.
    :param int peak_width: width of a peak.
    :param int background_end: end position of the background for the 
        first peak.
    :param int background_width: width of a background.
    :param int peak_interval: gap between peaks.

    :return list peaks: a list of peak data in 1D numpy.ndarray.
    :return list backgrounds: a list of background data in 1D numpy.ndarray.
    """
    peaks = []
    backgrounds = []
    peak0 = peak_start
    bkg0 = background_end - background_width
    for i in range(n_peaks):
        peaks.append(trace[:, peak0:peak0 + peak_width])
        backgrounds.append(trace[:, bkg0:bkg0 + background_width])
        peak0 += peak_interval
        bkg0 += peak_interval

    return peaks, backgrounds 


def compute_sigma(mu1, sigma1, mu2, sigma2, corr):
    """Compute error propagation for correlated data.
    
    :param float/Series mu1: dataset 1 mean.
    :param float/Series sigma1: dataset 1 standard deviation. 
    :param float/Series mu2: dataset 2 mean.
    :param float/Series sigma2: dataset 2 standard deviation.
    :param float/Series corr: correlation between dataset 1 and 2.

    :return float/Series: standard deviation of mu2/mu1.
    """
    if mu1 == 0 or mu2 == 0:
        raise ValueError("mu1 and mu2 cannot be zero!")

    return np.sqrt((sigma1 / mu1) ** 2 + (sigma2 / mu2) ** 2 
                   - 2 * corr * sigma1 * sigma2 / (mu1 * mu2))


def compute_absorption(I0, I1):
    """Compute absorption.

    A = -log(I1/I0)

    :param numpy.ndarray I0: incident beam intensity, 1D.
    :param numpy.ndarray I1: transmitted beam intensity, 1D.

    :return float muA: absorption mean.
    :return float sigmaA: absorption standard deviation.
    :return float muI0: I0 mean.
    :return float sigmaI0: I0 standard deviation.
    :return float weight: weight calculated from I0.
    :return float muI1: I1 mean.
    :return float sigmaI1: I1 standard deviation.
    :return float corr: correlation coefficient between I0 and I1.
    :return int count: number of data points.
    """
    count = len(I0)

    muI0 = I0.mean()
    sigmaI0 = I0.std()
    weight = np.sum(I0)

    muI1 = I1.mean()
    sigmaI1 = I1.std()

    corr = np.corrcoef(I1, I0)[0, 1]

    # we need the 'abs' for the background channel which has both positive
    # and negative data
    muA = -np.log(abs(muI1) / muI0)
    sigmaA = compute_sigma(muI0, sigmaI0, muI1, sigmaI1, corr)

    return muA, sigmaA, muI0, sigmaI0, weight, muI1, sigmaI1, corr, count


class XasAnalyzer(abc.ABC):
    """Abstract class for Xray Absoprtion Spectroscopy analysis."""
    sources = {
        'MONO': 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY',
        'XGM':'SCS_BLU_XGM/XGM/DOOCS',
        'XGM_OUTPUT': 'SCS_BLU_XGM/XGM/DOOCS:output',
        'SA3_XGM': 'SA3_XTD10_XGM/XGM/DOOCS',
        'SA3_XGM_OUTPUT': 'SA3_XTD10_XGM/XGM/DOOCS:output'
    }

    def __init__(self, run_folder):
        """Initialization.
        
        :param str run_folder: full path of the run folder.
        """
        self._run = RunDirectory(run_folder)

        # get the DataFrame for XGM control data
        self._xgm_df = self._run.get_dataframe(
            fields=[(self.sources['XGM'], '*value')])
        self._xgm_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
        self._sa3_xgm_df = self._run.get_dataframe(
            fields=[(self.sources['SA3_XGM'], '*value')])
        self._sa3_xgm_df.rename(columns=lambda x: x.split('/')[-1],
                                inplace=True)
        
        # get the DataFrame for SoftMono control data
        self._mono_df = self._run.get_dataframe(
            fields=[(self.sources['MONO'], '*value')])
        self._mono_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)

        self._photon_energies = None  # photon energies for each pulse
        self._I0 = None 
        self._I1 = OrderedDict()

        self._data = None  # pulse-resolved data in DataFrame

    def info(self):
        """Print out information of the run(s)."""
        first_train = self._run.train_ids[0]
        last_train = self._run.train_ids[-1]
        train_count = len(self._run.train_ids)
        span_sec = (last_train - first_train) / 10
        span_txt = str(datetime.timedelta(seconds=span_sec))
        photon_energies = self._mono_df['actualEnergy']

        print('# of trains:          ', train_count)
        print('Duration:             ', span_txt)
        print('First train ID:       ', first_train)
        print('Last train ID:        ', last_train)
        print('Min photon energy:    ', round(photon_energies.min(), 4), 'eV')
        print('Max photon energy:    ', round(photon_energies.max(), 4), 'eV')

        print('MCP channels:')
        for ch, value in self._channels.items():
            print('    - {}: {}'.format(ch, value['raw']))

    def _check_sources(self):
        """Check all the required sources are in the data."""
        sources = self._run.all_sources
        for src in self.sources.values():
            if src not in sources:
                raise ValueError("Source not found: {}!".format(src))

    def plot_xgm_run(self, *, figsize=(8, 5.6)):
        """Plot the train resolved data from XGM.

        :param tuple figsize: figure size.
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 12

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax1_tw = ax1.twinx()

        ln1 = ax1.plot(self._xgm_df['pulseEnergy.photonFlux'],
                       label=r"Pulse energy ($\mu$J)")
        number_of_bunches = self._xgm_df['pulseEnergy.nummberOfBrunches']
        ln2 = ax1_tw.plot(number_of_bunches, label="Number of pulses", c='g')

        lns = ln1 + ln2
        lables = [l.get_label() for l in lns]
        ax1.legend(lns, lables)
        ax1.set_ylabel(r"Pulse energy ($\mu$J)")
        ax1_tw.set_ylabel("Number of pulses")
        if number_of_bunches.max() - number_of_bunches.min() < 5:
            mean_n_bunches = int(number_of_bunches.mean())
            ax1_tw.set_ylim((mean_n_bunches - 4.5, mean_n_bunches + 4.5))

        ax2.plot(1000 * self._xgm_df['beamPosition.ixPos'], label="x")
        ax2.plot(1000 * self._xgm_df['beamPosition.iyPos'], label="y")

        ax2.set_xlabel("Train ID")
        ax2.set_ylabel(r"Beam position ($\mu$m)")
        ax2.legend()
        fig.tight_layout()

        return fig, (ax1, ax1_tw, ax2)

    def plot_xgm_train(self, *, index=0, train_id=None, figsize=(8, 5.6)):
        """Plot xgm measurement in a given train.
        
        :param int index: train index. Ignored if train_id is given.
        :param int train_id: train ID.
        :param tuple figsize: figure size.
        """
        import matplotlib.pyplot as plt

        if train_id is None:
            tid, data = self._run.train_from_index(index)
        else:
            tid, data = self._run.train_from_id(train_id)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        ax1.plot(data[self.sources['SA3_XGM_OUTPUT']]['data.intensityTD'],
                 marker='.')
        ax2.plot(data[self.sources['XGM_OUTPUT']]['data.intensityTD'],
                 marker='.')
        for ax in (ax1, ax2):
            ax.set_ylabel(r"Pulse energy ($\mu$J)")
            ax.set_xlim((-0.5, 100.5))

        ax1.set_title("SA3 XGM")
        ax2.set_title("SCS XGM")
        ax2.set_xlabel("Pulse ID")
        fig.suptitle("Train ID: {}".format(tid))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, (ax1, ax2)

    @abc.abstractmethod
    def process(self):
        """Process the run data."""
        pass

    def select(self, keys, lower=-np.inf, upper=np.inf):
        """Select data within the given boundaries.
        
        :param str/list/tuple/numpy.ndarray: key(s) for applying the filter.
        :param float lower: lower boundary (included).
        :param float upper: higher boundary (included).
        """
        n0 = len(self._data)
        if isinstance(keys, (list, tuple, np.ndarray)):
            # TODO: remove this for loop
            for key in keys:
                self._data.query("{} <= {} <= {}".format(lower, key, upper),
                                 inplace=True)
        else:
            self._data.query("{} <= {} <= {}".format(lower, keys, upper),
                             inplace=True)

        print("{} out of {} data are selected!".format(len(self._data), n0))
        return self

    @property
    def data(self):
        """Get the pulse-resolved data in pandas.DataFrame.

        The data is not filtered! The signs of signals in MCP channels 
        are flipped.
        """
        return self._data            

    @abc.abstractmethod
    def compute_total_absorption(self):
        """Compute absorption for all data.

        :return: total absorption data in pandas.DataFrame with index being the 
            MCP channel name and columns being:
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
        pass

    @abc.abstractmethod
    def compute_spectrum(self, n_bins=20, point_wise=False):
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
        pass


class XasTim(XasAnalyzer):
    def __init__(self, *args, channels=('D', 'C', 'B', 'A'), 
                 pulse_separation=880e-9, interleaved_mode=False, **kwargs):
        """Initialization.
        
        :param tuple channels: names of AdqDigitizer channels which 
            connects to MCP1 to MCP4.
        :param float pulse_separation: pulse separation in a train, in s.
        :param bool interleaved_mode: the resolution is improved by a factor
            of two in the interleaved mode. Default = False.
        """
        super().__init__(*args, **kwargs)

        self.sources.update({
            'DIGITIZER': 'SCS_UTC1_ADQ/ADC/1',
            'DIGITIZER_OUTPUT': 'SCS_UTC1_ADQ/ADC/1:network'
        })

        self._front_channels = ('MCP1', 'MCP2', 'MCP3')
        self._back_channels = ('MCP4',)

        self._channels = {
            'MCP{}'.format(i):
            {'raw': "digitizers.channel_1_{}.raw.samples".format(ch),
             'apd': "digitizers.channel_1_{}.apd.pulseIntegral".format(ch)}
              for i, ch in enumerate(channels, 1)}

        self._check_sources()

        self._resolution = 0.5e-9  # digitizer resolution, in s
        if interleaved_mode:
            self._resolution = 0.25e-9
        self._peak_interval = pulse_separation / self._resolution
        
        for ch in self._channels:
            self._I1[ch] = None 

    def plot_digitizer_train(self, *, index=0, train_id=None,
                             figsize=(8, 11.2), x_min=None, x_max=None):
        """Plot digitizer signals in a given train.

        :param int index: train index. Ignored if train_id is given.
        :param int train_id: train ID.
        :param tuple figsize: figure size.
        :param int x_min: minimum sample ID.
        :param int x_max: maximum sample ID.
        """
        if train_id is None:
            tid, data = self._run.train_from_index(index)
        else:
            tid, data = self._run.train_from_id(train_id)

        digitizer_raw_data = {
            ch: data[self.sources['DIGITIZER_OUTPUT']][value['raw']]
            for ch, value in self._channels.items()
        }

        n_channels = len(self._channels)

        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 12

        fig, axes = plt.subplots(n_channels, 1, figsize=figsize)

        for ax, (key, value) in zip(axes, digitizer_raw_data.items()):
            ax.plot(value)
            ax.set_title(key.upper())
            ax.set_ylabel("Intensity (arb.)")
            ax.set_xlim((x_min, x_max))
        ax.set_xlabel("Samples")
        
        fig.suptitle("Train ID: {}".format(tid))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, axes

    def _integrate_channel(self, channel_id, n_pulses, *args): 
        """Integration of a FastAdc channel for all trains in a run.
 
        :param str channel_id: full name of the output channel.
        :param int n_pulses: number of pulses in a train.

        :return numpy.ndarray: 1D array holding integration result for
            each train.
        """
        trace  = self._run.get_array(self.sources['DIGITIZER_OUTPUT'],
                                     channel_id)

        peaks, backgrounds = find_peaks(trace, n_pulses, *args)

        ret = []
        for peak, background in zip(peaks, backgrounds):
            ret.append(np.trapz(peak, axis=-1) - background.median(axis=-1))

        return np.ravel(ret, order="F") 

    def process(self, n_pulses, pulse_id0=0, *, use_apd=True,
                peak_start=None, peak_width=None, 
                background_end=None, background_width=None):
        """Override.
        
        :param int n_pulses: number of pulses in a train.
        :param int pulse_id0: first pulse ID. Default = 0.
        :param bool use_apd: use the integration calculated from the 
            hardware.
        :param int peak_start: start position of the first peak. Ignored if
            use_apd == True.
        :param int peak_width: width of a peak. Ignored if use_apd == True.
        :param int background_end: end position of the background for the 
            first peak. Ignored if use_apd == True.
        :param int background_width: width of a background. Ignored if 
            use_apd == True.
        """
        # self._I0 is a numpy.ndarray
        self._I0 = self._run.get_array(
            self.sources['XGM_OUTPUT'], 'data.intensityTD').values[...,
                pulse_id0:pulse_id0 + n_pulses].flatten()

        for ch, value in self._channels.items():
            if use_apd:
                integrals = self._run.get_array(
                    self.sources['DIGITIZER_OUTPUT'], value['apd']).values[
                        ..., :n_pulses]
                self._I1[ch] = -np.ravel(integrals)
            else:
                # self._I1[ch] is a list of numpy.ndarray
                # Note: the sign of I1 is reversed here!!!
                self._I1[ch] = -self._integrate_channel(
                    value['raw'], n_pulses, peak_start, peak_width, 
                    background_end, background_width, int(self._peak_interval))

        self._photon_energies = np.repeat(
            self._mono_df['actualEnergy'], n_pulses)
        
        data = {'energy': self._photon_energies, "XGM": self._I0}
        data.update({ch: self._I1[ch] for ch in self._channels})
            
        self._data = pd.DataFrame(data)

        return self

    def compute_total_absorption(self):
        """Override."""
        absorption = pd.DataFrame(
            columns=['muA', 'sigmaA', 'muI0', 'sigmaI0', 'weight', 
                     'muI1', 'sigmaI1', 'corr', 'count']
        )

        for ch in self._front_channels:
            absorption.loc[ch] = compute_absorption(
                self._data['XGM'], self._data[ch])

        return absorption

    def compute_spectrum(self, n_bins=20, point_wise=False):
        """Override."""
        # binning
        binned = self._data.groupby(pd.cut(self._data['energy'], bins=n_bins))

        if not point_wise:
            # mean
            binned_mean = binned.mean()
            binned_mean.columns = ['mu' + col if col != 'energy' else col
                                   for col in binned_mean.columns]
            # standard deviation
            binned_std = binned.std()
            binned_std.drop("energy", axis=1, inplace=True)
            binned_std.columns = ['sigma' + col for col in binned_std.columns]

            # correlation
            binned_corr = binned.corr().loc[pd.IndexSlice[:, 'XGM'], :].drop(
                columns=['XGM', 'energy'], axis=1).reset_index(
                level=1, drop=True)
            binned_corr.columns = ['corr' + col for col in binned_corr.columns]

            spectrum = pd.concat(
                [binned_mean, binned_std, binned_corr], axis=1)
            spectrum['count'] = binned['energy'].count()

            # calculate absorption and its sigma for each bin
            for i, ch in enumerate(self._front_channels, 1):
                spectrum['muA{}'.format(i)] = spectrum.apply(
                    lambda x: -np.log(abs(x['mu' + ch])/x['muXGM']), 
                    axis=1)
                spectrum['sigmaA{}'.format(i)] = spectrum.apply(
                    lambda x: compute_sigma(x['muXGM'],
                                            x['sigmaXGM'],
                                            x['mu' + ch],
                                            x['sigma' + ch],
                                            x['corr' + ch]), axis=1)
        else:
            # TODO: implement
            pass

        return spectrum

    def plot_correlation(self, channel=None, *, figsize=(8, 6), ms=6, 
                         alpha=0.05, n_bins=20):
        """Generate correlation plots.
        
        :param str channel: MCP channel name, e.g. MCP1, for visualizing
            a single channel with four plots, or None (default) for 
            visualizing all the channels with one plot each. 
            Case insensitive.
        :param tuple figsize: figure size.
        :param int ms: marker size for the scatter plots.
        :param float alpha: transparency for the scatter plots.
        :param int n_bins: number of bins for the histogram plots.
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 12
        
        absorption = self.compute_total_absorption()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        I0 = self._data['XGM']
        if channel is None:
            for ax, ch in zip(axes.flatten(), self._channels):
                I1 = self._data[ch]
                ax.scatter(I0, I1, s=ms, alpha=alpha, label=None)
                reg = LinearRegression().fit(I0.values.reshape(-1, 1), I1)

                if ch in self._back_channels:
                    label = None 
                else:
                    muA = absorption.loc[ch, "muA"]
                    sigmaA = absorption.loc[ch, "sigmaA"]
                    label = "Abs: {:.3g} +/- {:.3g}".format(muA, sigmaA)  

                ax.plot(I0, reg.predict(I0.values.reshape(-1, 1)), 
                        c='#FF8000', lw=2, label=label)
                    
                ax.set_xlabel("$I_0$")
                ax.set_ylabel("$I_1$")
                ax.set_title(ch)
                if ch not in self._back_channels:
                    ax.legend()

            fig.tight_layout()
        elif channel.upper() in self._channels:
            ch = channel.upper()
            I1 = self._data[ch]
            axes[1][0].scatter(I0, I1, s=ms, alpha=alpha, label=None)
            reg = LinearRegression().fit(I0.values.reshape(-1, 1), I1)
            
            if ch in self._back_channels:
                label = None
            else:
                muA = absorption.loc[ch, "muA"]
                sigmaA = absorption.loc[ch, "sigmaA"]
                label = "Abs: {:.3g} +/- {:.3g}".format(muA, sigmaA)  

            axes[1][0].plot(I0, reg.predict(I0.values.reshape(-1, 1)), 
                c='#FF8000', lw=2, label=label)

            axes[1][0].set_xlabel("$I_0$")
            axes[1][0].set_ylabel("$I_1$")
            if ch not in self._back_channels:
                axes[1][0].legend()

            axes[0][0].hist(I0, bins=n_bins)
            axes[0][0].axvline(I0.mean(), c='#6A0888', ls='--')

            axes[1][1].hist(self._I1[ch],
                            bins=n_bins, orientation='horizontal')
            axes[1][1].axhline(I1.mean(), c='#6A0888', ls='--')

            with np.errstate(divide='ignore', invalid='ignore'):
                absp = -np.log(abs(I1) / I0) 
            axes[0][1].scatter(I0, absp, s=ms, alpha=alpha)
            axes[0][1].set_xlabel("$I_0$")
            axes[0][1].set_ylabel("$-log(I_1/I_0)$")
            if ch not in self._back_channels:
                axes[0][1].axhline(
                    absorption.loc[ch, 'muA'], c='#FF8000', ls='--',
                    label="SNR@labs: {:.3g}".format(1. / sigmaA),
                )
                axes[0][1].legend()
            
            fig.suptitle(ch)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            raise ValueError("Not understandable input!")
        
        return fig, axes

    def plot_spectrum(self, channel=None, *, figsize=(6, 4.5), capsize=4, 
                      n_bins=20, use_transmission=False):
        """Generate spectrum plots.

        :param str channel: MCP channel name, e.g. MCP1, for visualizing
            a single channel, or None (default) for visualizing MCP1-3 
            altogether. Case insensitive.
        :param tuple figsize: figure size.
        :param int capsize: cap size for the error bar.
        :param int n_bins: number of energy bins.
        :param bool use_transmission: False for plotting energy vs. 
            absorption, while True for plotting energy vs. I1.
            Default = False.
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 12

        spectrum = self.compute_spectrum(n_bins=n_bins)

        fig, ax = plt.subplots(figsize=figsize)

        if channel is None:
            for i, ch in enumerate(self._front_channels, 1):
                if use_transmission:
                    y = spectrum['mu' + ch]
                    y_err = spectrum['sigma' + ch] / \
                            spectrum['count'].apply(np.sqrt)
                else:
                    y = spectrum['muA' + str(i)]
                    y_err = spectrum['sigmaA' + str(i)] / \
                            spectrum['count'].apply(np.sqrt)

                ax.errorbar(spectrum.energy, y, y_err,
                            capsize=capsize, fmt='.', label=ch)

        elif channel.upper() in self._front_channels:
            ch = channel.upper()
            idx = list(self._channels.keys()).index(ch) + 1

            if use_transmission:
                y = spectrum['mu' + ch]
                y_err = spectrum['sigma' + ch] / \
                        spectrum['count'].apply(np.sqrt)
            else:
                y = spectrum['muA{}'.format(idx)]
                y_err = spectrum['sigmaA{}'.format(idx)] / \
                        spectrum['count'].apply(np.sqrt)

            ax.errorbar(spectrum.energy, y, y_err,
                        fmt='.', capsize=capsize, label=ch)
        else:
            raise ValueError("Not understandable input!")

        ax.set_xlabel("Energy (eV)")
        if use_transmission:
            ax.set_ylabel("I1")
        else:
            ax.set_ylabel("Absorption")

        ax.legend()

        fig.tight_layout()

        return fig, ax
