# Searchinng for new Physics at the LHC with Machine Learning

<p align="left">
<a href="#"><img alt="License" src="https://img.shields.io/github/license/mackostya/ml-for-lhc-jets?label=license&color=green&style=flat"></a>
<a href="#"><img alt="Last commit" src="https://img.shields.io/github/last-commit/mackostya/ml-for-lhc-jets/main?color=orange&style=flat"></a>
</p>

In this work 3 different types of models are presented. This models are specified for a binary classification between 2 diffetent types of jets (Quantum chromodynamics and top jets). The task is to create a classifier, to be able differ between the background (QCD-jets) and signal (top-jets), which could result into finiding new Physics Beyond Standard Model (BSM). The paper with a detailed explination can be found [here](https://github.com/mackostya/ml-for-lhc-jets/blob/main/Searching_for_new_physics_at_the_LHC_with_Machine_Learning.pdf).

The 3 models implemented in this work are:
- **CNN** (convolutional neural network), which trains on representation of jets as 40x40 images.
- **AE** (autoencoder), which trains on the background (QCD) 40x40 images. Afterwards both signal and background are passed to the model. The reconstruction of the signal is expected to be less efficient, because the model was trained on the background.
- **edgeConvNet**, the network, which trains on the jets, that are represented as reclustered jets. This model is modified from [here](https://github.com/hqucms/ParticleNet).
