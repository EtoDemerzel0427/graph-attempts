# graph-attempts
## About this Repo

This repo will include some experimental models on heterogeneous network embedding (At this point, to be precise, on bipartite network embedding). ___"Experimental"___ means these models are somehow immature, and maybe cannot provide with good results yet.

Note that these experiments are not repeated to avoid the contingency, so there is probability that the numbers listed below were not reliable.

## Models

So far, I'v finished two models. They are both designed for bipartite networks. A common idea is that they both build two homogeneous graph from the original bipartite graph, which follows the way introduced in **BiNE**(Sigir18). They are:

### GCN + LINE

A **GCN+LINE** model. The original random-created node embeddings are fed into a shallow GCN(in this implementation, only one layer),  and then back prop with a loss function similar to LINE(MXNET Gluon written). 

It is not a good implementation, and I am considering re-implementing it in TensorFlow.

### B-LINE

A revised version of LINE. I call it *B-LINE* since it is a LINE model built specifically for a bipartite network. I modified the way of negative sampling, that is, I sampled different kinds of edges: the ___original bipartite edges, the homogeneous edges of users, the homogeneous edges of items___, so there are in total 4 types of negative sampling methods. On link prediction task, it provide a good result in a Wikipedia dataset, outperforming the original LINE on AUC-ROC/-PR by approximately 1%.  The results are listed below.

| metric       |LINE(original)|LINE(only heter)|B-LINE|
| ---------- | ------------------ |---|---|
| **AUC-ROC** | 0.9329263444610261 |0.9323199284520142|**0.9482962797963796**|
| **AUC-PR** | 0.939179432498551  |0.9249040457791543|**0.9461344054322227**|

Further investigation about the sampling method is still in progress.

