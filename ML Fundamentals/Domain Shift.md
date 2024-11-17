
Different types of domain shifts: 
- Covariate shift: The probability density of the data itself changes (but within the same label distribution)
	- For example, there's more examples closer to the decision boundary and thus the classification accuracy goes down due to harder samples
	- For example in CV object classification, we'd like to learn the object features themselves but often associate the background with the label and the classifier is not robust if the background changes (also known as spurious correlations due to shortcut learning of gradient descent)
- Concept shift: change in the conditional probability on labels