import re
import abc

import numpy as np

from karabo_data import RunDirectory


class XasProcessor(abc.ABC):
    """Abstract class for Xray Absoprtion Spectroscopy analysis."""
    def __init__(self):
        self._spectrum = None

    @abc.abstractmethod
    def process_run(self, run_folder, energy=None):
        """Process data in a run folder.
        
        :param str run_folder: folder that contains run data. 
        :param float energy: photon energy for this run. If None, it 
            checks if the data contains train-resolved photon energy 
            information, e.g. from monochromator.
        """
        pass

    @abc.abstractmethod
    def plot_correlation(self):
        """Plot the correlation between I0 and I1."""
        pass

    @abc.abstractmethod
    def plot_spectrum(self):
        """Plot the absorption spectrum."""
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
        
        self._run_folder = None
        self._xgm = None
        self._mcps = None

    @staticmethod
    def _integrate_adc_channel(run, ch, threshold=-200):
        """Integration along a FastAdc channel.
          
        :param DataCollection run: run data. 
        :param str ch: output channel name.
        :param int threshold: data above this threshold will be ignored.
                      
        :return
        """
        background = run.get_array(ch, 'data.baseline')
        raw_data = run.get_array(ch, 'data.rawData') - background
  
        return np.trapz(raw_data.where(raw_data < threshold, 0), axis=1)

    def process_run(self, run_folder, energy=None):
        run = RunDirectory(run_folder)
        self._run_folder = run_folder

        self._check_adc_channels(run)

        xgm = run.get_array(self._xgm_id + ":output", 'data.intensityTD').max(axis=1)
                                                                            
        mcps = []
        for ch in self._adc_channels.values():
            mcps.append(self._integrate_adc_channel(run, ch))
        
        self._xgm = xgm
        self._mcps = mcps

        if energy is not None:
            # TODO: add a spectrum point
            pass

    def plot_correlation(self):
        if self._xgm is None or self._mcps is None or self._run_folder is None:
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i, ax in enumerate(np.ravel(axes)):
            ax.plot(self._xgm, self._mcps[i], '.', alpha=0.2)
            ax.set_xlabel("XGM pulse energy ($\mu$J)")
            ax.set_ylabel("{} integration (arb.)".format(list(self._adc_channels.keys())[i]))

        fig.suptitle(self._run_folder, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot_spectrum(self):
        import matplotlib.pyplot as plt

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

