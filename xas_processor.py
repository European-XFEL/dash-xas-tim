"""
Offline data analysis and visualization tool for Xray Absorption Spectroscopy
(XAS) experiments at SCS, European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
import re
from collections import OrderedDict
import datetime

import numpy as np
import pandas as pd

from karabo_data import RunDirectory


def find_peaks(trace, n_peaks, peak_start, peak_width, 
               background_end, background_width, peak_interval):
    """Return a list of peaks.
    
    :param trace:
    :param n_peaks:
    :param peak_start:
    :param peak_width:
    :param background_end:
    :param background_width:
    :param peak_interval:

    :returns:
    """
    peaks = []
    backgrounds = []
    peak0 = peak_start
    bkg0 = background_end - background_width
    for i in range(n_peaks):
        peaks.append(trace[:, peak0:peak0 + peak_width])
        backgrounds.append(
            trace[:, bkg0:bkg0 + background_width].median(axis=-1)
        )
        peak0 += peak_interval
        bkg0 += background_width

    return peaks, backgrounds 


def compute_absorption(I0, I1):
    """Compute absorption.
    
    :param numpy.ndarray I0: incident beam intensity, 1D.
    :param numpy.ndarray I1: transmitted beam intensity, 1D.

    :returns: average absorption, standard deviation of absorption,
        weight, average I1, standard deviation of I1, average I0,
        standard deviation of I0, correlation coefficient betwen
        I0 and I1, number of data points.
    """    
    I0_mean = I0.mean()
    I0_std = I0.std()
    weight = np.sum(I0)

    I1_mean = I1.mean()
    I1_std = I1.std()

    p = np.corrcoef(I1, I0)[0, 1]

    absorption_mean = -np.log(abs(I1_mean)/I0_mean)

    c1 = (I1_std / I1_mean) ** 2 + (I0_std / I0_mean) ** 2
    c2 = 2 * I0_std * I1_std / (I0_mean * I1_mean)
    absorption_sigma = np.sqrt(c1 - c2*p) 

    return absorption_mean, absorption_sigma, weight, I1_mean, I1_std, \
        I0_mean, I0_std, p, len(I0)


class XasProcessor(abc.ABC):
    """Abstract class for Xray Absoprtion Spectroscopy analysis."""
    sources = {
        'mono': 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY',
        'xgm':'SCS_BLU_XGM/XGM/DOOCS',
        'xgm_output': 'SCS_BLU_XGM/XGM/DOOCS:output',
        'sa3_xgm': 'SA3_XTD10_XGM/XGM/DOOCS',
        'sa3_xgm_output': 'SA3_XTD10_XGM/XGM/DOOCS:output'
    }

    def __init__(self, run_folder, *, pulse_id_min=0, n_pulses=1):
        """Initialization.
        
        :param str run_folder: full path of the run folder.
        :param int pulse_id_min: start of the pulse ID.
        :param int n_pulses: number of pulses in a train.
        """
        self._run = RunDirectory(run_folder)

        self._pulse_id_min = pulse_id_min
        self._n_pulses = n_pulses

        # get the DataFrame for XGM control data
        self._xgm_df = self._run.get_dataframe(
            fields=[(self.sources['xgm'], '*value')])
        self._xgm_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)
        self._sa3_xgm_df = self._run.get_dataframe(
            fields=[(self.sources['sa3_xgm'], '*value')])
        self._sa3_xgm_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)

        # get the DataFrame for Softmono control data
        self._mono_df = self._run.get_dataframe(
            fields=[(self.sources['mono'], '*value')])
        self._mono_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)

        self._I0 = None 
        self._I1 = OrderedDict()

        # the naming convention of the columns is in line with the code
        # provided by Loic Le Guyader:
        #
        # - muA: absorption mean
        # - sigmaA: absorption standard deviation
        # - weights: sum of Io values
        # - muT: transmission mean
        # - sigmaT: transmission standard deviation
        # - muIo: Io mean
        # - sigmaIo: Io standard deviation
        # - p: correlation coefficient between T and Io
        # - counts: length of T
        self._absorption = pd.DataFrame(
            columns=['muA', 'sigmaA', 'weights', 'muT', 'sigmaT', 
                     'muIo', 'sigmaIo', 'p', 'counts']
        )

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
        print('# of pulses per train:', self._n_pulses)
        print('First pulse ID:       ', self._pulse_id_min)
        print('Min photon energy:    ', round(min(photon_energies), 4), 'eV')
        print('Max photon energy:    ', round(max(photon_energies), 4), 'eV')

        print('MCP channels:')
        for channel, channel_id in self._channels.items():
            print('    - {}: {}'.format(channel.upper(), channel_id))

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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax1_tw = ax1.twinx()

        ln1 = ax1.plot(self._xgm_df['pulseEnergy.photonFlux'], label=r"Pulse energy ($\mu$J)")
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

        ax1.plot(data[self.sources['sa3_xgm_output']]['data.intensityTD'], marker='.')
        ax2.plot(data[self.sources['xgm_output']]['data.intensityTD'], marker='.')
        for ax in (ax1, ax2):
            ax.set_ylabel(r"Pulse energy ($\mu$J)")
            ax.set_xlim((self._pulse_id_min - 0.5, 
                         self._pulse_id_min + self._n_pulses + 0.5))
        ax1.set_title("SA3 XGM")
        ax2.set_title("SCS XGM")
        ax2.set_xlabel("Pulse ID")
        fig.suptitle("Train ID: {}".format(tid))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, (ax1, ax2)

    @abc.abstractmethod
    def process(self, config):
        """Process the run data.
        
        :param dict config: configuration for integrating of digitizer signal.
        """
        pass

    @property
    def correlation(self):
        """Get the correlation data in pandas.DataFrame.

        :return pandas.DataFrame: DataFrame with columns I0 and all I1(s).  
        """
        data = {"XGM": self._I0}
        data.update({ch.upper(): self._I1[ch] for ch in self._channels})
            
        return pd.DataFrame(data)

    @property
    def absorption(self):
        return self._absorption


class XasFastCCD(XasProcessor):
    pass


class XasFastADC(XasProcessor):
    pass


class XasDigitizer(XasProcessor):
    def __init__(self, *args, channels=('D', 'B', 'C', 'A'), 
                 pulse_separation=880e-9, interleaved_mode=False, **kwargs):
        """
        
        :param tuple channels: names of AdqDigitizer channels which 
            connects to MCP1 to MCP4.
        :param float pulse_separation: pulse separation in a train, in s.
        :param bool interleaved_mode: the resolution is improved by a factor
            of two in the interleaved mode. Default = False.
        """
        super().__init__(*args, **kwargs)

        self.sources.update({
            'digitizer': 'SCS_UTC1_ADQ/ADC/1',
            'digitizer_output': 'SCS_UTC1_ADQ/ADC/1:network'
        })
        self._channels = {'mcp{}'.format(i+1): 
                          "digitizers.channel_1_{}.raw.samples".format(ch)
                          for i, ch in enumerate(channels)}

        self._check_sources()

        self._resolution = 0.5e-9  # digitizer resolution, in s
        if interleaved_mode:
            self._resolution = 0.25e-9
        self._peak_interval = pulse_separation / self._resolution
        
        for ch in self._channels:
            self._I1[ch] = None 

    def plot_digitizer_train(self, *, index=0, train_id=None, figsize=(8, 11.2),
                             x_min=None, x_max=None):
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

        digitizer_raw_data = {key: data[self.sources['digitizer_output']][value] 
                              for key, value in self._channels.items()}

        n_channels = len(self._channels)
        total_samples = len(list(digitizer_raw_data.values())[0])

#         traces = []
#         for i, (key, value) in enumerate(digitizer_raw_data.items()):
#              traces.append(go.Scatter(x=np.arange(end-start), y=value[start:end], name=key, mode="lines"))
#                         
#              layout = dict(xaxis=dict(title='Samples'), yaxis=dict(title='Intensity (arb.)'), 
#                            legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5))
# 
#              fig = dict(data=traces, layout=layout)    
#              iplot(fig)  
        import matplotlib.pyplot as plt

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

    def _integrate_channel(self, channel_id, config):
        """Integration of a FastAdc channel for all trains in a run.
 
        :param str channel_id: full name of the output channel.
        :param dict config: configuration for integrating of digitizer signal.
            If None, use automatic peak finding. If not, the following keys
            are mandantory:
            - peak_start: int
                Sample index at the start of the first peak.
            - peak_width: int
                Peak width.
            - background_end: int
                Sample index at the end of the background for the first peak.
            - background_width: int
                Background width.

        :return numpy.ndarray: 1D array holding integration result for each train. 
        """
        trace  = self._run.get_array(self.sources['digitizer_output'], channel_id)

        if config is None:
            cfg = {"auto": True}
        else:
            cfg = {"auto": False}
            cfg.update(config)

        peaks, backgrounds = find_peaks(trace, self._n_pulses, 
                                        cfg['sample_start'], 
                                        cfg['peak_width'], 
                                        cfg['background_end'], 
                                        cfg['background_width'], 
                                        int(self._peak_interval))

        ret = []
        for peak, background in zip(peaks, backgrounds):
            ret.append(np.trapz(peak, axis=-1) - background)

        return np.ravel(ret, order="F") 

    def process(self, config=None):
        """Override."""
        # self._I0 is a numpy.ndarray
        self._I0 = self._run.get_array(
            self.sources['xgm_output'], 'data.intensityTD').values[...,
                self._pulse_id_min: self._pulse_id_min + self._n_pulses].flatten()

        for channel, channel_id in self._channels.items():
            # self._I1 is a list of numpy.ndarray
            self._I1[channel] = self._integrate_channel(channel_id, config)

        # set condition for valid data: I0 > 0 and I1 < 0 
        channels = list(self._channels.keys())
        condition = self._I0 > 0
        # only apply to MCP1, MCP2 and MCP3
        for i in range(3):
            condition &= self._I1[channels[i]] < 0

        print("Removed {}/{} data with I0 <= 0 or I1 >= 0 (MCP1-3)".format(
            len(self._I0) - sum(condition), len(self._I0)))

        self._I0 = self._I0[condition]
        for channel in self._channels:
            self._I1[channel] = self._I1[channel][condition]
            self._absorption.loc[channel] = compute_absorption(
                self._I0, self._I1[channel])
            print("{} processed".format(channel.upper()))

    def plot_correlation(self, channel="all", *, figsize=(8, 6),
                         marker_size=6, alpha=0.05, n_bins=20):
        """Generate correlation plots.
        
        :param str channel: MCP channel name, e.g. mcp1, for visualizing
            a single channel with four plots, or 'all' for visualizing all 
            the channels with one plot each. Case insensitive.
        :param tuple figsize: figure size.
        :param int marker_size: marker size for the scatter plots.
        :param float alpha: transparency for the scatter plots.
        :param int n_bins: number of bins for the histogram plots.
        """
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        channel = channel.lower()
        if channel == "all":
            for ax, channel in zip(axes.flatten(), self._channels):
                ax.scatter(self._I0, self._I1[channel], s=marker_size, alpha=alpha)
                reg = LinearRegression().fit(self._I0.reshape(-1, 1), self._I1[channel])
                absorption = self._absorption.loc[channel, :]
                ax.plot(self._I0, reg.predict(self._I0.reshape(-1, 1)), 
                        c='#FF8000', lw=2, label="Abs: {:.3g} +/- {:.3g}".format(
                        absorption["muA"], absorption["sigmaA"]))

                ax.set_xlabel("$I_0$")
                ax.set_ylabel("$I_1$")
                ax.set_title(channel.upper())
                ax.legend()

            fig.tight_layout()
        elif channel in self._channels:
            absorption = self._absorption.loc[channel, :]
            axes[1][0].scatter(self._I0, self._I1[channel], s=marker_size, alpha=alpha)
            reg = LinearRegression().fit(self._I0.reshape(-1, 1), self._I1[channel])
            axes[1][0].plot(self._I0, reg.predict(self._I0.reshape(-1, 1)), 
                c='#FF8000', lw=2, label="Abs: {:.3g} +/- {:.3g}".format(
                    absorption["muA"], absorption["sigmaA"]))
            axes[1][0].set_xlabel("$I_0$")
            axes[1][0].set_ylabel("$I_1$")
            axes[1][0].legend()

            axes[0][0].hist(self._I0, bins=n_bins)
            axes[0][0].axvline(self._I0.mean(), c='#6A0888', ls='--')

            axes[1][1].hist(self._I1[channel], bins=n_bins, orientation='horizontal')
            axes[1][1].axhline(self._I1[channel].mean(), c='#6A0888', ls='--')

            with np.errstate(divide='ignore', invalid='ignore'):
                absp = -np.log(np.abs(self._I1[channel] / self._I0)) 
            axes[0][1].scatter(self._I0, absp, s=marker_size, alpha=alpha)
            axes[0][1].set_xlabel("$I_0$")
            axes[0][1].set_ylabel("$-log(I_1/I_0)$")
            axes[0][1].axhline(
                absorption['muA'], 
                label="SNR@labs: {:.3g}".format(1./absorption['sigmaA']),
                c='#FF8000', ls='--'
            )
            axes[0][1].legend()
            
            fig.suptitle(channel.upper())
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            raise ValueError("Not understandable input!")
        
        return fig, axes

    def _check_adc_channels(self, run):
        """Check the selected FastAdc channels all contain data."""
        tid, data = run.train_from_index(0)

        activate_adc_channels = []
        for key, value in sorted(data.items()):
            if re.search(re.compile(r"{}.*output".format(self._adc_id)), key) and 'data.rawData' in value:
                activate_adc_channels.append(key)
                            
        for ch in self._adc_channels.values():
            if ch not in activate_adc_channels:
                raise ValueError("{} is not an active FastAdc channel in train {}!".format(ch, tid))

