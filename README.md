# Parametric Bayesian multi-task learning for modeling biomarker trajectories


The code from our "Modeling longitudinal biomarkers with parameteric Bayesian multi-task learning" paper.

Here's a simple example that you can run and modify:

```bash
simple_example
```

You can also run the simulations described in the paper via:

```bash
sim_both_full
```
This will generate a couple of intermediate files for you in the out_blr_sim directory along with two figures (from the paper) that show mean absolute error (MAE) and parameter inference related metrics across the 50 simulation runs and 2 simulation scenarios (intercept-coupled and slope-coupled trajectories). 

This above command will take at least a few hours to run as it's building eight coupled models fifty times for four different noise levels, across two simulation scenarios (8 x 50 x 4 x 2 = 3,200 models for 200 subjects). 

You can run a faster version that will give you a good idea of how it all works by running:

```bash
sim_both_quick
```

This will build eight models ten times for the four noise levels and two scenarios for fewer subjects (8 x 10 x 4 x 2 = 640 for 100 subjects). It should run in minutes rather than hours. 


If you want to replicate our ADNI results, you need to have ([access to the ADNI dataset] http://adni.loni.usc.edu/data-samples/access-data/). Once you do, I can send you the necessary data and code. Contact: l.aksman@ucl.ac.uk.
