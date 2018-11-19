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
    """Return a list of peaks."""
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


class AbsorptionSpectrum:
    """Absorption spectrum."""
    def __init__(self):
        """Initialization."""
        self._photon_energies = [] 
        self._absorptions = [] 
        self._absorption_sigmas = [] 
        self._weights = [] 

    def add_point(self, photon_energy, absorption, sigma=0.0, weight=np.inf):
        """Add a data point."""
        self._photon_energies.append(photon_energy)
        self._absorptions.append(absorption)
        self._absorption_sigmas.append(sigma)
        self._weights.append(weight)

    def to_dataframe(self):
        """Return a pandas.DataFrame representation of the spectrum."""
        df = pd.DataFrame({
            "photon_energy": self._photon_energies, 
            "absorption": self._absorptions, 
            "absorption_sigma": self._absorption_sigmas, 
            "weight": self._weights 
            })
        return df


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
        :param int pulse_id_min: start of the pulse id.
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

        self._spectrums = OrderedDict() 

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
        print('Min photon energy:    ', min(photon_energies), 'eV')
        print('Max photon energy:    ', max(photon_energies), 'eV')

    def _check_sources(self):
        """Check all the required sources are in the data."""
        sources = self._run.all_sources
        for src in self.sources.values():
            if src not in sources:
                raise ValueError("Source not found: {}!".format(src))

    def plot_xgm_run(self, *, figsize=(8, 7.2)):
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
        """Plot xgm measurement in a given train."""
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
    def process(self):
        """Process data."""
        pass


    @abc.abstractmethod
    def correlation(self, name):
        """Get a pandas.DataFram for correlation analysis.

        :param str name: name of I1. For example, the name of a MCP channel.

        :return pandas.DataFrame: DataFrame with columns I0 and I1.  
        """
        pass

    @abc.abstractmethod
    def spectrum(self, name):
        """Get a pandas.DataFrame for spectrum analysis..
        
        :param str name: name of I1. For example, the name of a MCP channel. 

        :return pandas.DataFrame: DataFrame with spectrum data in columns. 
        """
        pass


class XasFastCCD(XasProcessor):
    pass


class XasFastADC(XasProcessor):
    pass


class XasDigitizer(XasProcessor):
    """Xray absorption spectroscopy analyser."""
    def __init__(self, *args,   
                 channels=('D', 'B', 'C', 'A'), 
                 pulse_separation=880e-9,
                 interleaved_mode=False, 
                 **kwargs):
        """Initialization.
        
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
        
        self._I0 = None 
        self._I1 = OrderedDict()
        for ch in self._channels:
            self._I1[ch] = None 
            self._spectrums[ch] = AbsorptionSpectrum()

    def plot_digitizer_train(self, *, index=0, train_id=None, figsize=(8, 11.2),
                             x_min=None, x_max=None):
        """Plot digitizer signals in a given train."""
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

    def _integrate_channel(self, ch, config):
        """Integration of a FastAdc channel for all trains in a run.
          
        :param str ch: full name of the output channel.
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
        # def integrate_mcp_channel(trace,  

        trace  = self._run.get_array(self.sources['digitizer_output'], ch)

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
        """Process the run data.
        
        :param dict config: configuration for integrating of digitizer signal.
        """
        self._I0 = self._run.get_array(
            self.sources['xgm_output'], 'data.intensityTD').values[...,
                self._pulse_id_min: self._pulse_id_min + self._n_pulses].flatten()
                                                                            
        for name, ch in self._channels.items():
            self._I1[name] = self._integrate_channel(ch, config)
            print("{} processed".format(name.upper()))

    def correlation(self, channel):
        return pd.DataFrame({
            "XGM": self._I0,
            "MCP": self._I1[channel.lower()]
        })

    def plot_correlation(self, channel="all", *, 
                         marker_size=6, alpha=0.05, n_bins=20):
        """Generate correlation plots.
        
        :param str channel: MCP channel name, e.g. mcp1, for visualizing
            a single channel with four plots, or 'all' for visualizing all 
            the channels with one plot each. Case insensitive.
        :param int marker_size: marker size for the scatter plots.
        :param float alpha: transparency for the scatter plots.
        :param int n_bins: number of bins for the histogram plots.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        channel = channel.lower()
        if channel == "all":
            for ax, ch in zip(axes.flatten(), self._channels):
                ax.scatter(self._I0, self._I1[ch], s=marker_size, alpha=alpha)
                ax.set_title(ch.upper())
        elif channel in self._channels:
            axes[1][0].scatter(self._I0, self._I1[channel], s=marker_size, alpha=alpha)
            axes[0][0].hist(self._I0, bins=n_bins)
            # TODO: use the absorption in Spectrum
            axes[0][1].scatter(self._I0, -np.log(self._I1[channel]/self._I0), 
                               s=marker_size, alpha=alpha)
            axes[1][1].hist(self._I1[channel], bins=n_bins, orientation='horizontal')
        else:
            raise ValueError("Not understandable input!")

        fig.tight_layout()

        return fig, axes

    def spectrum(self, name):
        return self._spectrums[name].to_dataframe()

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

