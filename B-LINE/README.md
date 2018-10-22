# Attempts on B-LINE

## Self loops

 I tried to eliminate the self-loops in the homogeneous graphs in **B-LINE**, since for a node to compare common neighbors with itself is meaningless, but only get minor improvement in the results(0.03% on AUC-ROC and 0.01% on AUC-PR). 

Notwithstanding， in the further exploration, the self-loops are all eliminated.



## Different types of sampling

Further experiments are implemented to find out the importance of each type of negative sampling method. The 4 types are *user-item, item-user, user-user, item-item*, in the following discussion we call them *A, B, C, D* respectively. The results of *A+B*, i.e., LINE(only heter) are already above.

| metric      | C                  | D                  | C+D                |
| ----------- | ------------------ | ------------------ | ------------------ |
| **AUC-ROC** | 0.5007851568810705 | 0.9367189461453098 | 0.9437038702181084 |
| **AUC-PR**  | 0.5099387885089641 | 0.9511483193665536 | 0.956057531218766  |

 I guess it is because the parameters for user vectors should be very small, and the following experiments confirmed my hypothesis:

| norm   | user vector weight   | item vector weight |
| ------ | -------------------- | ------------------ |
| **L1** | 0.31261186471094143  | 6.712333144444139  |
| **L2** | 0.033402900170662574 | 0.8330851154319653 |

The original LINE:

| norm   | user vector weight | item vector weight |
| ------ | ------------------ | ------------------ |
| **L1** | 6.582210594122601  | 23.78970681416144  |
| **L2** | 0.8885764159650376 | 2.7138167566984617 |

