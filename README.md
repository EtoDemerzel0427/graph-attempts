# graph-attempts
## About this Repo

This repo will include some experimental models on heterogeneous network embedding (At this point, to be precise, on bipartite network embedding). ___"Experimental"___ means these models are somehow immature, and maybe cannot provide with good results yet.

Note that these experiments are not repeated to avoid the contingency, so there is probability that the numbers listed below were not reliable.

## Models

Till now, I'v finished two models. They are both designed for bipartite networks. A common idea is that they both build two homogeneous graph from the original bipartite graph, which follows the way introduced in **BiNE**(Sigir18). They are:

### GCN + LINE

A **GCN+LINE** model. The original random-created node embeddings are fed into a shallow GCN(in this implementation, only one layer),  and then back prop with a loss function similar to LINE(MXNET Gluon written). 

It is not a good implementation, and I am considering re-implementing it in TensorFlow.

### B-LINE

A revised version of LINE. I call it *B-LINE* since it is a LINE model built specifically for a bipartite network. I modified the way of negative sampling, that is, I sampled different kinds of edges: the ___original bipartite edges, the homogeneous edges of users, the homogeneous edges of items___, so there are in total 4 types of negative sampling methods. On link prediction task, it provide a good result in a Wikipedia dataset, outperforming the original LINE on AUC-ROC/-PR by approximately 1%.  The results are listed below.

| metric       |LINE(original)|LINE(only heter)|B-LINE|
| ---------- | ------------------ |---|---|
| **AUC-ROC** | 0.9329263444610261 |0.9323199284520142|**0.9482962797963796**|
| **AUC-PR** | 0.939179432498551  |0.9249040457791543|**0.9461344054322227**|

  I also tried to eliminate the self-loops in the homogeneous graphs in **B-LINE**, since for a node to compare common neighbors with itself is meaningless, but only get minor improvement in the results(0.03% on AUC-ROC and 0.01% on AUC-PR). 

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

