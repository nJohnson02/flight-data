# flight-data
## Random Forest
1. Random Forest with series gps altitude:
[gps alt]
Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.93      0.91       158
           1       0.54      0.42      0.47        31

    accuracy                           0.85       189
   macro avg       0.72      0.67      0.69       189
weighted avg       0.83      0.85      0.84       189

Confusion Matrix:
[[147  11]
 [ 18  13]]

2. With gps alt and vs:
[gps alt, vs]

Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.94      0.92       158
           1       0.60      0.48      0.54        31

    accuracy                           0.86       189
   macro avg       0.75      0.71      0.73       189
weighted avg       0.85      0.86      0.86       189

Confusion Matrix:
[[148  10]
 [ 16  15]]

3. added airspped
[GPS ALT, VS, IAS]
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.94      0.92       158
           1       0.61      0.45      0.52        31

    accuracy                           0.86       189
   macro avg       0.75      0.70      0.72       189
weighted avg       0.85      0.86      0.85       189

Confusion Matrix:
[[149   9]
 [ 17  14]]

3. Replaced vs with altitude rade (derive altitude)
[gps alt, alt rate, IAS]

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       158
           1       0.88      0.74      0.81        31

    accuracy                           0.94       189
   macro avg       0.92      0.86      0.89       189
weighted avg       0.94      0.94      0.94       189


Confusion Matrix:
[[155   3]
 [  8  23]]

4. [gps alt, alt rate]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       158
           1       0.90      0.84      0.87        31

    accuracy                           0.96       189
   macro avg       0.93      0.91      0.92       189
weighted avg       0.96      0.96      0.96       189


Confusion Matrix:
[[155   3]
 [  5  26]]

5. [gps alt, alt rate, vertical acceleration]
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.99      0.97       158
           1       0.93      0.81      0.86        31

    accuracy                           0.96       189
   macro avg       0.94      0.90      0.92       189
weighted avg       0.96      0.96      0.96       189

Confusion Matrix:
[[156   2]
 [  6  25]]

 6. Random forest K folds
 [gps alt, alt rate]
Cross-validation scores: [0.93650794 0.96825397 0.94148936 0.97340426 0.96276596]
Mean accuracy: 0.9564842958459978
Standard deviation: 0.01475244028636258

 ## Gradient Boosting
 1. Gradient boosting
 [gps alt, altitude rate, veritcal acceleration]
 Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       158
           1       0.90      0.90      0.90        31

    accuracy                           0.97       189
   macro avg       0.94      0.94      0.94       189
weighted avg       0.97      0.97      0.97       189

2. Removed vertical acceleration
Confusion Matrix:
[[155   3]
 [  3  28]]
 [gps alt, altitude rate]
 Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       158
           1       0.97      0.94      0.95        31

    accuracy                           0.98       189
   macro avg       0.98      0.96      0.97       189
weighted avg       0.98      0.98      0.98       189

Confusion Matrix:
[[157   1]
 [  2  29]]

 3. Same features, re-randomized test/train set
 Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       158
           1       0.93      0.84      0.88        31

    accuracy                           0.96       189
   macro avg       0.95      0.91      0.93       189
weighted avg       0.96      0.96      0.96       189


Confusion Matrix:
[[156   2]
 [  5  26]]

 4. same features, k-fold cross validation for testing
 Cross-validation scores: [0.99470899 0.97354497 0.96276596 0.98404255 0.96276596]
Mean accuracy: 0.9755656872678149
Standard deviation: 0.012410261291420765

5. same features but with confusion matrix and k-fold and plotted tree (see discord)
Cross-validation scores: [0.99470899 0.97354497 0.96276596 0.98404255 0.96276596]
Mean accuracy: 0.9755656872678149
Standard deviation: 0.012410261291420765

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       158
           1       0.97      0.94      0.95        31

    accuracy                           0.98       189
   macro avg       0.98      0.96      0.97       189
weighted avg       0.98      0.98      0.98       189


Confusion Matrix:
[[157   1]
 [  2  29]]

## Gradient Boost Prediction
1. Cut off the last 20 seconds of each approach (only the first 40 are considered)
[gps alt, alt rate]

Cross-validation scores: [0.83068783 0.84656085 0.85106383 0.86170213 0.83510638]
Mean accuracy: 0.8450242035348416
Standard deviation: 0.011143484743653351

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.96      0.91       158
           1       0.50      0.19      0.28        31

    accuracy                           0.84       189
   macro avg       0.68      0.58      0.59       189
weighted avg       0.80      0.84      0.80       189


Confusion Matrix:
[[152   6]
 [ 25   6]]

2. Added airspeed
[gps alt, alt rate, ias]
Cross-validation scores: [0.82010582 0.81481481 0.86702128 0.84574468 0.82978723]
Mean accuracy: 0.8354947652819993
Standard deviation: 0.01895277068187432

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.95      0.90       158
           1       0.47      0.23      0.30        31

    accuracy                           0.83       189
   macro avg       0.66      0.59      0.60       189
weighted avg       0.80      0.83      0.81       189


Confusion Matrix:
[[150   8]
 [ 24   7]]