# co-op-code
Spring 2021 Co-Op Rotation 1 Code

The data included in each file came from the article "Machine Learning assisted design of high entropy alloys with desired property" (https://www.sciencedirect.com/science/article/abs/pii/S1359645419301430#cebib0010). It was my goal to replicate and expand on some of the machine learning models they developed and create a predictive model that produced results similar to those found in the study. I used skeleton code from GitHub (all could be found searching KNN regression for Python). I used StackOverflow to help me figure out how to fix any error messages from the code. These include the "rough drafts" of my code (everything that I've altered or written, whether successful or not). The code that doesn't work completely is code that I attempted to tweak to fit my data, but after hours of altering with little to no success, I moved onto a new outline. Most of the code includes messages explaining what different sections do, but a summary of each file is included below:

All KNN Valid Attempts: Each attempt shows a different method/approach for making a successful KNN model for my data on Spyder. Most of the code skeleton can be found on GitHub from other users, but I couldn't get it to work with my data (I kept getting error messages).

Article 1 SLR, PR, H: I analyzed the data using techniques I learned from the Analyzing Data, Data Science, and Machine Learning IBM courses on EdX. I imported the data and necessary libraries, formatted the data, and checked for missing data. Then, I made visualizations; I made scatter plots to show the relationship between the atomic percent of each element and hardness and histograms that showed the number of samples with each at% of each element. I then made a linear regression model for Al vs. Hardness and another for Cu vs. Hardness. I then practiced standardizing and normalizing the data and seeing how that changed the visulaization and results of the regressions. I attempted to make a multiple linear regression model, and ended with a polynomial regression of at% Al vs. Hardness Value (HV).

Attempt at classification: I imported and formatted my data, then added another column that would label the alloy depending on the HV. I then tried to make a KNN model using that classification, but it didn't work.

KNN Al Co - KNN Fe Ni: Each of these include a test KNN model, an elbow curve to determine the best value of K, then the implementation of a KNN model using two components to predict the hardness value, with a visualization. Each graph is formatted and color coded, and the class is predicted. Each also includes a predictive model at the bottom where you can change the values for the given elements and it will return the color/classification of the combination.

KNN very close: Using a new skeleton code and adjusting it, only a few things were wrong with this code. It almost successfully made the KNN model I was looking for. It also includes lines of code that didn't work for me, but theoretically could work if the data was different or if the syntax was changed slightly.

MLHEA: Attempt at KNN classifier and a decision tree. Neither included a visualization, and was very similar to the skeleton code.

Multidimensional KNN: aspects from the skeleton code or alterations that didn't work are in quotations and the code is outlined in fairly good detail. It used each element to predict the hardness value, instead of just two. I couldn't figure out how to visualize that, so it doesn't include a visualization graph. However, in the quotations, there are a few more visualizations for this dataset. The correlation matrix heatmaps, pair-wise scatter plots, Visualizing 3-D mix data using scatter plots, and visualizing 3-D numeric data with Scatter Plots all worked well. This included notes on what some of these graphs meant in the quotations, as well as what alterations were needed to make nonworking components work again.

PCA: I followed the following video tutorial to make the principal component analysis (PCA). https://www.youtube.com/watch?v=oiusrJ0btwA&t=411s

PyCaret Clustering: I made each of the PyCaret clustering algorithms. PyCaret can quickly make nine types of clustering algorithms. These include K-Means clustering, affinity propagation, mean shift clustering, spectral clustering, agglomerative clustering, density based spatial clustering, OPTICS clustering, birch clustering, and K-Modes clustering. Each include visualizations.

PyCaret KNN: a KNN model that uses each feature to predict the hardness value. The predictive table is not included in this, but multiple visualizations are included.

PyCaret Regressions: This only includes regressions using my classification system (hard, average, or soft). It uses PyCaret to evaluate a KNN regression model, a Bayesian Ridge model, a Random Forest regressor, and an extra trees regressor. 

PyCaret Regressions (ALL Hardness): This uses PyCaret to make predictive models (using each type of regression algorithm PyCaret can quickly make) on the actual hardness values of the alloys. The KNN regressor is split up into two parts: there is one KNN regression (the first one) that is simple and follows the same format of every other model. The second KNN model is broken up into multiple iterations, and each of those iterations is specified with a text line. Most algorithms, if not all, should include the spearman correlation and a predictive model. 

Violin PLots: Includes an outline of how I made violin plots for each of the alloys.

Tinkering KNN: the first successful KNN code; it's very similar, if not identical, to the KNN Al Co code file. 
