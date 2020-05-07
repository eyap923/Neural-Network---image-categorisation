# Results

## CNN 4 Layer
---

### 1st Run

Iteration: 470, Loss: 1.8275398015975952, Accuracy: 55%
[[8345   67  402    0  164    0    0   14    0    8]
 [  89 8730   89    0   84    0    0    0    0    8]
 [ 192   16 7336    0 1452    0    0    2    0    2]
 [3775 2427  676    0 1731    0    0    2    0  389]
 [  49   24 1074    0 7850    0    0    3    0    0]
 [  71   30  109    0   77    0    0 5299    0 3414]
 [3156   61 2487    0 3290    0    0    2    0    4]
 [   0    0    0    0    0    0    0 8593    0  407]
 [1222  125 1606    0 2276    0    0  646    0 3125]
 [   0    0    9    0    0    0    0  430    0 8561]]

Classification report for CNN :
              precision    recall  f1-score   support

           0       0.49      0.93      0.64      9000
           1       0.76      0.97      0.85      9000
           2       0.53      0.82      0.64      9000
           3       0.00      0.00      0.00      9000
           4       0.46      0.87      0.61      9000
           5       0.00      0.00      0.00      9000
           6       0.00      0.00      0.00      9000
           7       0.57      0.95      0.72      9000
           8       0.00      0.00      0.00      9000
           9       0.54      0.95      0.69      9000

    accuracy                           0.55     90000
   macro avg       0.34      0.55      0.41     90000
weighted avg       0.34      0.55      0.41     90000

### 2nd run

Iteration: 470, Loss: 1.8622573614120483, Accuracy: 53%
[[   0    0    0 1752  162   63 6993   18    0   12]
 [   0    0    0 4522  797  112  322   30    0 3217]
 [   0    0    0  245 3344   25 5382    0    0    4]
 [   0    0    0 8093  298    5  604    0    0    0]
 [   0    0    0  507 7288   17 1180    8    0    0]
 [   0    0    0   18    0 8420    9  400    0  153]
 [   0    0    0  738 1203   34 7023    0    0    2]
 [   0    0    0    0    0  259    0 8305    0  436]
 [   0    0    0 1162 1429  910 2192  751    0 2556]
 [   0    0    0    3    0   84   24  420    0 8469]]
C:\Users\angel\Anaconda3\envs\CNN 2 layer\lib\site-packages\sklearn\metrics\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for CNN :
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      9000
           1       0.00      0.00      0.00      9000
           2       0.00      0.00      0.00      9000
           3       0.47      0.90      0.62      9000
           4       0.50      0.81      0.62      9000
           5       0.85      0.94      0.89      9000
           6       0.30      0.78      0.43      9000
           7       0.84      0.92      0.88      9000
           8       0.00      0.00      0.00      9000
           9       0.57      0.94      0.71      9000

    accuracy                           0.53     90000
   macro avg       0.35      0.53      0.41     90000
weighted avg       0.35      0.53      0.41     90000


### 3rd run

Iteration: 470, Loss: 1.7292522192001343, Accuracy: 66%
[[7464    3  232  624   97    0  564   14    0    2]
 [  19 8563   62  260   73    0   23    0    0    0]
 [ 126    4 6672  145 1384    0  669    0    0    0]
 [ 218   55  134 7929  342    0  322    0    0    0]
 [  12    5  872  498 6834    0  779    0    0    0]
 [ 108    0  148  201   21    0  317 5102    0 3103]
 [2159    0 1170  409 1127    0 4135    0    0    0]
 [   0    0    0    0    0    0    0 8535    0  465]
 [ 308   48 1007  546 2192    0 2254  657    0 1988]
 [   2    0    2    7    0    0    7  436    0 8546]]
C:\Users\angel\Anaconda3\envs\CNN 2 layer\lib\site-packages\sklearn\metrics\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for CNN :
              precision    recall  f1-score   support

           0       0.72      0.83      0.77      9000
           1       0.99      0.95      0.97      9000
           2       0.65      0.74      0.69      9000
           3       0.75      0.88      0.81      9000
           4       0.57      0.76      0.65      9000
           5       0.00      0.00      0.00      9000
           6       0.46      0.46      0.46      9000
           7       0.58      0.95      0.72      9000
           8       0.00      0.00      0.00      9000
           9       0.61      0.95      0.74      9000

    accuracy                           0.65     90000
   macro avg       0.53      0.65      0.58     90000
weighted avg       0.53      0.65      0.58     90000

## VGG
---
### Run #1
Iteration: 236, Loss: 1.5684336423873901, Accuracy: 84%
[[3251    5  101  247   37   22  240    1   95    1]
 [   6 3807    7  130   25    1    8    0   15    1]
 [  80    1 3311   47  319    8  206    0   28    0]
 [  90   28   61 3532  115    0  140    0   31    3]
 [   8    8  598  206 2809    4  325    0   42    0]
 [   3    0    0    4    0 3774    0  144    8   67]
 [ 741    5  705  224  549   13 1650    0  113    0]
 [   0    0    0    0    0   87    0 3552    4  357]
 [   8    8   28   47   18   27   11   11 3837    5]
 [   0    0    3    0    0   42    0   90    2 3863]]
Classification report for CNN :
              precision    recall  f1-score   support

           0       0.78      0.81      0.79      4000
           1       0.99      0.95      0.97      4000
           2       0.69      0.83      0.75      4000
           3       0.80      0.88      0.84      4000
           4       0.73      0.70      0.71      4000
           5       0.95      0.94      0.95      4000
           6       0.64      0.41      0.50      4000
           7       0.94      0.89      0.91      4000
           8       0.92      0.96      0.94      4000
           9       0.90      0.97      0.93      4000

    accuracy                           0.83     40000
   macro avg       0.83      0.83      0.83     40000
weighted avg       0.83      0.83      0.83     40000

### Run #2
Iteration: 236, Loss: 1.7647881507873535, Accuracy: 62%
[[3537    0  183    0  110    5    0    0   80   85]
 [  59    0   67    0   87    0    0    0   11 3776]
 [  78    0 2888    0  998    1    0    0   24   11]
 [ 491    0  146    0  628    1    1    0   85 2648]
 [  10    0  613    0 3331    0    0    0   27   19]
 [   3    0    1    0    0 3729    0  142   28   97]
 [1245    0 1092    0 1464    4    0    0  127   68]
 [   0    0    0    0    0  103    0 3679   11  207]
 [   7    0   77    0   71   10    0   13 3804   18]
 [   0    0    0    0    0   51    0  167    2 3780]]
C:\Users\angel\Anaconda3\envs\CNN 2 layer\lib\site-packages\sklearn\metrics\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for CNN :
              precision    recall  f1-score   support

           0       0.65      0.88      0.75      4000
           1       0.00      0.00      0.00      4000
           2       0.57      0.72      0.64      4000
           3       0.00      0.00      0.00      4000
           4       0.50      0.83      0.62      4000
           5       0.96      0.93      0.94      4000
           6       0.00      0.00      0.00      4000
           7       0.92      0.92      0.92      4000
           8       0.91      0.95      0.93      4000
           9       0.35      0.94      0.51      4000

    accuracy                           0.62     40000
   macro avg       0.49      0.62      0.53     40000
weighted avg       0.49      0.62      0.53     40000

### Run #3
Iteration: 236, Loss: 1.8928422927856445, Accuracy: 61%
[[   0    1    0  390   60   23 3420    1  103    2]
 [   0    1    0 3627  225    5  125    0   12    5]
 [   0    0    0  113 1384    5 2400    0   77   21]
 [   0    0    0 3490  172    0  312    0   25    1]
 [   0    0    0  211 2927    5  826    0   29    2]
 [   0    0    0    6    0 3781    8  142    7   56]
 [   0    0    0  179  712    5 2964    0  130   10]
 [   0    0    0    0    0  170    0 3464    1  365]
 [   0    0    0   39   26   18   75   16 3804   22]
 [   0    0    0    1    0   74    2  143    2 3778]]
C:\Users\angel\Anaconda3\envs\CNN 2 layer\lib\site-packages\sklearn\metrics\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Classification report for CNN :
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      4000
           1       0.50      0.00      0.00      4000
           2       0.00      0.00      0.00      4000
           3       0.43      0.87      0.58      4000
           4       0.53      0.73      0.62      4000
           5       0.93      0.95      0.94      4000
           6       0.29      0.74      0.42      4000
           7       0.92      0.87      0.89      4000
           8       0.91      0.95      0.93      4000
           9       0.89      0.94      0.91      4000

    accuracy                           0.61     40000
   macro avg       0.54      0.61      0.53     40000
weighted avg       0.54      0.61      0.53     40000