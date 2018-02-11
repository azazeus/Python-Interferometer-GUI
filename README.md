# Python-Interferometer-GUI
A GUI console for a nonlinear Fourier transform interferometer I built.

Pre-requisites:
+ future
+ PyQt4
+ pyqtgraph
+ numpy
+ PyDAQmx
+ antonSerial (written by my colleague, included in the folder)
+ threading
+ time
+ logging
+ datetime
+ math
+ collections
+ h5py
+ scipy

This code runs a linear stage with a retro-reflector to generate a time-delay between two interfering ultrashort pulses from a Ti:Sapphire laser. The pulses passing through a solution of a fluorescent protein undergo two-photon absorption. The fluorescence is recorded using a PMT and read out with an ADC card. 

An independent He-Ne laser interferometer using the same retro-reflector records both in-phase and out-of-phase intensity as well. The He-Ne intensity is fit to an ellipse to calculate the phase difference in the He-Ne interferometer and ultimately the distance traveled by the retro-reflector.

The interfering pulses generate a fluorescence interferogram whose x-axis is calibrated using the He-Ne interferometer distance measurement. The interferogram is then fourier transformed to obtain the teo-photon absorption cross-section of the fluorescent protein.

The data is recorded in an open-source HDF5 format.
