
## K-means
- non-parametric method because there are no parameters to be optimized. K is a hyperparameter of the algorithm, so it's not considered a parameter that can be optimized within the algorithm itself. 
- We want to group the data into K clusters via some kind of similarity/distance metric between data points
- Most of the time will converge but will also get stuck in local minima bc is non-convex optimization

### Steps: 
1. Initialize cluster centroids randomly 
2. Repeat until convergence: 
	1. Assignment: For every data point, assign it to the closest centroid
	2. Centroid Update: take the average of all data points with the respective centroid label
