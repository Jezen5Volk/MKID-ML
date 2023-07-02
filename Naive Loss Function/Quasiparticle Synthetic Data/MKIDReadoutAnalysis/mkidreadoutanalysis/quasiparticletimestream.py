import numpy as np
import random
import matplotlib.pyplot as plt
import skimage
from scipy.signal import welch
from logging import getLogger


class QuasiparticleTimeStream:
    """ A time series with values proportional to the change in quasiparticle density relative to the
    quasiparticle density when there are no photons hitting the depvice.
    ...
    Attributes:
     @type wl: list
        list containing discrete photon wavelengths greater than 808 nm [nm]
     @type fs: float
         sample rate [Hz]
     @type ts: float
         sample time [Sec]
     @type tvec: np.array
         time vector [Sec]
     @type points: int
         number of samples
     @type data_nonoise: np.array
         timestream with no added noise
     @type photon_arrivals: np.array of booleans
          whether or not a photon arrived in that time step
    1 per quasiparticle = 2 * gap energy
    # qp = photon energy / (2*gap energy) * efficiency_factor
    # normalized quasiparticle timestream
    # some class knows constants that turn it into frequency shifts (material dependent)
    # energy of photon
    # usually want 90 degree phase response
     """

    def __init__(self, wl, fs, ts, seed=3):
        self.wl = wl #nm
        self.fs = fs #Hz
        self.ts = ts #sec
        self.photon_arrival_rng = np.random.default_rng(seed=seed)
        self.points = int(self.ts * self.fs)
        self.tvec = np.arange(0, self.points) / self.fs
        self.data = np.zeros(self.points)
        self._holdoff = None
        self.photon_arrivals = None
        self.photon_pulse = None
        self.pulse_time = None


    @property
    def dt(self):
        return self.tvec[1]-self.tvec[0]

    def plot_timeseries(self, data, ax=None, fig=None):
        plt.figure()
        plt.plot(self.tvec, data)
        plt.xlabel('time (sec)')
        plt.ylabel(r"$\propto \Delta$ Quasiparticle Density")

    def gen_quasiparticle_pulse(self, tf = 30):
        """generates an instantaneous change in quasiparticle density
         which relaxes in tf fall time in usec. The height of the pulse is
        determined by wl, the wavelength of the incident photon in nanometers"""
        wl = random.choice(self.wl)
        amp = 808/wl #scaled so 808 nm gives qp density of 1, and other wavelengths give 1 * (808/wl)
        tp = np.linspace(0,10 * tf, int(self.fs*(10 * tf * 1e-6)))  # pulse duration
        self.photon_pulse = amp * np.exp(-tp / tf)
        self.pulse_time = tp

    def plot_pulse(self, ax=None, fig=None):
        plt.figure()
        plt.plot(self.pulse_time, self.photon_pulse)
        plt.xlabel('Time (usec)')
        plt.ylabel(r"$\propto \Delta$ Quasiparticle Density")

    def gen_photon_arrivals(self, cps=500):
        """ generate boolean list corresponding to poisson-distributed photon arrival events.
        Inputs:
        - cps: int, photon co
        unts per second.
        """
        photon_events = self.photon_arrival_rng.poisson(cps / self.fs, self.tvec.shape[0])
        self.photon_arrivals = np.array(photon_events, dtype=bool)
        if sum(photon_events) > sum(self.photon_arrivals):
            getLogger(__name__).warning(f'More than 1 photon arriving per time step. Lower the count rate?')
        if sum(photon_events) == 0:
            getLogger(__name__).warning(f"Warning: No photons arrived. :'(")
        return self.photon_arrivals

    def populate_photons(self, tf = None, wl = None):
        """ populate the quasiparticle time stream with photon arrival events
        Inputs:
        -tf: int, fall time in microseconds [usec]
        -wl: list with float/int elements, allowed (discrete) photon wavelengths in nanometers [nm]
        """
        for i in range(self.data.size - self.pulse_time.size):
            if self.photon_arrivals[i]:
                self.gen_quasiparticle_pulse(tf=tf)
                self.data[i:i + self.pulse_time.size] = self.photon_pulse
        return self.data
