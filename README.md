# graph-attempts
This repo will include some experimental models on heterogeneous network embedding (At this period, to be precise, on bipartite network embedding). ___"Experimental"___ means these models are somehow immature, and maybe cannot provide with good results yet.



Till now, I'v finished two models. They are both designed for bipartite networks. A common idea is that they both build two homogeneous graph from the original bipartite graph, which follows the way introduced in **BiNE**(Sigir18). They are:
* A **GCN+LINE** model. The original random-created node embeddings are fed into a shallow GCN(in this implementation, only one layer),  and then back prop with a loss function similar to LINE(MXNET Gluon written). 
  It is not a good implementation, and I am considering re-implement it in TensorFlow.

* A revised version of LINE. I call it *B-LINE* since it is a LINE model built specifically for a bipartite network. I modified the way of negative sampling, that is, I sampled different kinds of edges: the ___original bipartite edges, the homogeneous edges of users, the homogeneous edges of items___. On link prediction task, it provide a good result in a Wikipedia dataset, outperform the orginal LINE on AUC-ROC/-PR by approximately 1%.  The results are listed below.

  | metric       |LINE(original)|LINE(only heter)|B-LINE|
  | ---------- | ------------------ |---|---|
  | **AUC-ROC** | 0.9329263444610261 |0.9323199284520142|**0.9482962797963796**|
  | **AUC-PR** | 0.939179432498551  |0.9249040457791543|**0.9461344054322227**|

  I also tried to eliminate the self-loops in the homogeneous graphs in **B-LINE**, since for a node to compare common neighbors with itself is meaningless, but only get minor improvement in the results(0.03% on AUC-ROC and 0.01% on AUC-PR). The code is now uploaded.
