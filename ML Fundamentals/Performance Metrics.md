
Notation of Ground Truth/Predicted: 
- True Positive (+/+)
- False Positive (-/+)
- True Negative (-/-)
- False Negative (+/-)

Recall (also True Positive Rate TPR): TP/**GT+** same as TP/(TP + FN)
Precision (think positives/predicted): TP/**Predicted+** same as TP/(TP + FP)
False Positive Rate (FPR): FP/GT- same as FP/(FP + TN)
Accuracy: TP + TN / everything
- But Accuracy is not dependable for high class imbalance. For example, if model always predicts y=0 and there's very few y=1 samples, then accuracy would be high but recall is near 0 because there are no TP (because would only be predicting TN and that's not part of recall calculation)

Thresholding in Binary Classification: 
- Result was determined in logistic regression by threshold at sign(0.5)
- Varying this 0.5 threshold will have: 
	- higher threshold = less false positives
	- lower threshold = less false negatives

Receiver-Operating Characteristic (ROC) curve: 
- Varying threshold will have varying precision and recall
- Area Under Curve (AUC) of ROC provides summary statistic of overall performance

F1 score: 
- Another summary statistic at a given threshold (instead of across all thresholds in AUC-ROC)
- Let P = Precision and R = Recall 
$$F_{1} = \frac{2PR}{P+R}$$

