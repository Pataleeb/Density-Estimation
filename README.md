# Density-Estimation
Histogram and KDE

This is a reproduction of Kanai, R., Feilden, T., Firth, C., and Rees, G. (2011). Political orientations are correlated with brain structure in young adults. Current Biology, 21(8), pp. 677-680. The data were collected to study whether the two brain regions are independent of each other while considering different types of political views.

The dataset contains information on 90 university students to examine the relationship between the size of different brain regions and their political views.

The variables amygdala and acc indicate the volume of two particular brain regions.

First, we create 1-dimensional histograms and Kernel Density Estimators (KDE) to estimate the distribution of the two brain regions.

Next, we construct 2-dimensional histograms for the variable pairs (i.e., amygdala and acc).

We use KDE to estimate the 2-dimensional density function for these regions.

Finally, we estimate the conditional joint distribution of the volume of the amygdala and acc, conditioning on a function of political orientation.
