# eht-image-clustering-analysis

A set of Python Jupyter Notebooks for the comparison between different clustering algorithms

I worked on these notebooks as an undergraduate research project during the Summer of 2022. 

## Acknowledgements
This project was initially started by previous undergraduate students Yuan Jea Hew and Anthony Hsu. Hew was responsible for the initial version of the notebooks which performed clustering using the K-means algorithm, and Hsu made contributions to improving the image alignment algorithms making it much easier to compare individual images.

Special thanks to Dr. Chi-Kwan Chan for providing me a research opportunity and advising me through the coding process.

## Summary
These notebooks, separated into multiple steps, will read in EHT images from a folder, apply Principle Component Analysis (PCA) to the EHT image data to reduce dimensionality, cluster the images using one or multiple clustering algorithms, and provide ways of comparing the effectiveness and similarities between clustering using different algorithms. 

They are separated into steps using Joblib in Python to dump variables as files at the end of one notebook. These files are then loaded into another notebook to be used again. Having the notebooks work in multiple steps allows them to accomodate larger data sets and provides useful checkpoints for debugging.

Clustering:
The clustering portion of the notebook is meant to be relatively easily expandable to include other clustering algorithms if needed in the future. By default, the notebooks include

0. K-Means (the original method)
1. Agglomerative Hierarchical
2. DBSCAN
3. Spectral
4. OPTICS

These are all implemented through Scikit-learn. It is worth noting that the later comparisons are done under the assumption that all of the clustering algorithms have results containing the same amount of clusters. For this reason, some care should be taken in choosing parameters for algorithms that do not have a number of clusters as an input parameter if they should be compared to other algorithms.

Comparisons:
Comparisons are done between two chosen algorithms from those used in the clustering. The methods of comparison include

1. calculating mean images and deviations of individual images from their mean images
2. visualizing sample images from each cluster
3. matching and pairing similar clusters between algorithms
4. counting how many images match between the paired clusters from 3.
