# graph-attempts
This repo will include some experimental models on heterogeneous network embedding (At this period, to be precise, on bipartite network embedding). ___"Experimental"___ means these models are somehow immature, and maybe cannot provide with good results yet.

Till now, I'v finished two models:
* A **BiNE+GCN** model, using a loss function similar to LINE(MXNET Gluon written).
  It is not a good implementation, and I am considering re-implement it in TensorFlow.
* A revised version of LINE. I modified the way of negative sampling. I sampled different kinds of edges: the ___original bipartite edges, the homogeneous edges of users, the homogeneous edges of items___. On link prediction task, it provide a good result in a Wikipedia dataset, surpassing LINE on AUC-ROC/-PR by approximately 1%. (Code not uploaded yet)

