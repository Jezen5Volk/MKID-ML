This directory contains the code that I wrote for the statistical implementation of my machine learning model. After my first attempt at implementation (see "First Implementation" directory, inside the "Naive Loss Function" directory), I sought to improve performance. A classic tenet of machine learning (and data science in general) is garbage in, garbage out. If I wanted to improve performance, I would need to start by improving the way I generated synthetic data. 

I had a dataset containing fifty thousand photon incidence events across five different wavelengths of light. The way I had been previously stitching them together to get a synthetic dataset involved generating a dataset of noise, then inserting photons on top of it. This resulted in discontinuities in the data set that aren't present in an actual dataset. To mitigate this, I generated a mapping between each of the fifty thousand photon incidence events. The ith element of the mapping contained an index for the photon incidence event which had the least discontinuous data. This mapping was somewhat computationally expensive to generate, so it was precomputed and then called repeatedly during the synthetic data generation step.

The next step was to improve the actual machine learning model itself. In order to be able to identify the energy of incident photons

ultimately produced hot garbage, le big sad
