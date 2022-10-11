from torch_geometric.nn import GINConv
from torch_geometric.utils import remove_self_loops



###################################### GIN conv layer ######################################
############################################################################################

class GINConv(GINConv):
    def __init__(self, nn, eps: float = 0, train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        '''
            MLP( (1+eps) x_v + \sum_{u} x_u ), as in Eq 4.1 of https://openreview.net/pdf?id=ryGs6iA5Km
            <CAVEAT> edge_index should NOT include self-loops, so is the edge_weight
        '''
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1 + self.eps) * x

        return self.nn(out)
    
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j


