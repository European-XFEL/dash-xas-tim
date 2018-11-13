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

import numpy as np
import pandas as pd

from karabo_data import RunDirectory


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
    def __init__(self):
        self._spectrums = OrderedDict() 

    @abc.abstractmethod
    def process_run(self, run_folder, photon_energy=None):
        """Process data in a run folder.
        
        :param str run_folder: folder that contains run data. 
        :param float photon_energy: photon energy for this run. If None, 
            it checks if the data contains train-resolved photon energy 
            information, e.g. from monochromator.
        """
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
    """Xray absorption spectroscopy analyser."""
    def __init__(self, *, xgm_id=None, adc_id=None, adc_channels=None):
        """Initialization.
        
        :param str xgm_id: XgmDoocs device ID.
        :param str adc_id: FastAdc device ID.
        :param dict adc_channels: names of FastAdc channels which received data.
        """
        super().__init__()

        self._xgm_id = xgm_id
        self._adc_id = adc_id
        self._adc_channels = adc_channels
        
        self._xgm = None 
        self._mcps = OrderedDict()
        for ch in adc_channels:
            self._spectrums[ch] = AbsorptionSpectrum()
            self._mcps[ch] = None 

    @staticmethod
    def _integrate_adc_channel(run, ch, threshold=-200):
        """Integration of a FastAdc channel for all trains in a run.
          
        :param DataCollection run: run data. 
        :param str ch: full name of the output channel.
        :param int threshold: data above this threshold will be ignored.
                      
        :return numpy.ndarray: 1D array holding integration result for each train. 
        """
        background = run.get_array(ch, 'data.baseline')
        raw_data = run.get_array(ch, 'data.rawData') - background
  
        return np.trapz(raw_data.where(raw_data < threshold, 0), axis=1)

    def process_run(self, run_folder, photon_energy=None):
        run = RunDirectory(run_folder)

        self._check_adc_channels(run)

        self._xgm = run.get_array(self._xgm_id + ":output", 'data.intensityTD').max(axis=1)
                                                                            
        for name, ch in self._adc_channels.items():
            self._mcps[name] = self._integrate_adc_channel(run, ch)

        if photon_energy is not None:
            # TODO: I don't like the different interface between np.array and xarray
            i_0 = self._xgm.mean().item()
            weight = self._xgm.sum().item()
            for name in self._spectrums.keys():
                i_1 = np.abs(self._mcps[name].mean())
                # TODO: can i_0 be 0?
                self._spectrums[name].add_point(photon_energy, -np.log(i_1 / i_0), 0.0, weight)
        else:
            pass
            # TODO: search mono information

    def correlation(self, name):
        return pd.DataFrame({
            "XGM": self._xgm,
            "MCP": self._mcps[name]
        })

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

