import torch
import torch.sparse as tsp


class dataset_wikics(object):
    """
    Preprocessing the raw dataset loaded from torch_geometric.datasets.WikiCS
         - converting mutiedges to simple edges;
         - augmenting the adj with self-loops;
         - appending normalized edge weights to the raw dataset;
         - computing the pos-weights for graph reconstruction loss.
    """
    def __init__(self, device='cpu'):
        super(dataset_wikics, self).__init__()
        self.device = device

    def __aug(self):
        """ adding self-loops """
        edges = self.data.edge_index
        self_loops = torch.arange(self.Nnodes).repeat(2,1)
        setattr(self.data, 'edge_index', torch.cat([edges, self_loops], 1))

    def __collapse(self):
        """ collapsing multiedges into simple edges """
        i_ = self.data.edge_index
        v_ = torch.ones(i_.shape[1])
        A  = torch.sparse_coo_tensor(i_, v_, (self.Nnodes, self.Nnodes)).coalesce()
        setattr(self.data, 'edge_index', A.indices())

    def __normalize(self):
        """ computing normalized edge weights (renormalization trick in GCN) """
        # get binary adj A
        i_ = self.data.edge_index
        v_ = torch.ones(i_.shape[1])
        A  = torch.sparse_coo_tensor(i_, v_, (self.Nnodes, self.Nnodes))
        
        # get D^{-1/2}, D is the degree mat
        dsqrt = tsp.sum(A, 1).to_dense().squeeze()
        dsqrt_inv  = dsqrt.pow(-.5) 
        self_loops = torch.arange(self.Nnodes).repeat(2,1)
        Dsqrt_inv  = torch.sparse_coo_tensor(self_loops, dsqrt_inv, (self.Nnodes, self.Nnodes))

        # D^{-1/2} * A * D^{-1/2}
        Ahat = tsp.mm(A, Dsqrt_inv)
        Ahat = tsp.mm(Dsqrt_inv, Ahat).coalesce()
        setattr(self.data, 'edge_index', Ahat.indices())
        setattr(self.data, 'edge_weight', Ahat.values())

    def __nll_adjustments(self):
        N = self.Nnodes
        E = self.data.edge_index.shape[1]
        pos_weight = torch.tensor([float((N * N - E) /E)])
        setattr(self, 'pos_weight', pos_weight)
        setattr(self, 'Legen_divisor', float(2 * (N * N - E)))

    def __call__(self, raw_ds):
        self.data = raw_ds.data
        self.Nnodes = self.data.x.shape[0]

        self.__aug()
        self.__collapse()
        self.__normalize()
        self.__nll_adjustments()

        # send to device
        self.data = self.data.to(self.device)
        self.pos_weight = self.pos_weight.to(self.device)

        return self



