import torch

class AgentVRP():
    VEHICLE_CAPACITY = 1.0

    def __init__(self, input):
        depot = input[0] # (batch_size, 2)
        loc = input[1] # (batch_size, n_nodes, 2)
        demand = input[2]

        self.batch_size, self.n_loc, _ = loc.shape

        # Coordinates of depot + other nodes -> (batch_size, 1+n_nodes, 2)
        self.coords = torch.cat((depot[:, None, :], loc), dim=-2)

        # Indices of graphs in batch
        self.ids = torch.arange(self.batch_size)[:, None] # (batch_size, 1)

        # State
        self.prev_a = torch.zeros(self.batch_size, 1)
        self.from_depot = self.prev_a == 0
        self.used_capacity = torch.zeros(self.batch_size, 1)

        # Nodes that have been visited will be marked with 1
        self.visited = torch.zeros(self.batch_size, 1, self.n_loc+1)

        # Step counter
        self.i = torch.zeros(1, dtype=torch.int64)

        # Constant tensors for scatter update (in step method)
        self.step_updates = torch.ones(self.batch_size, 1, dtype=torch.uint8)
        self.scatter_zeros = torch.zeros(self.batch_size, 1, dtype=torch.int64)

    @staticmethod
    def outer_pr(a, b):
        """ Outer product of a and b row vectors.
            result[k] = matmul( a[k].t(), b[k] )
        """
        return torch.einsum('ki,kj->kij', a, b)

    def get_att_mask(self):
        """ Mask (batchsize, n_nodes, n_nodes) for attention encoder.
            We maks alredy visited nodes except for depot (can be visited multiple times).
        """
        # Remove depot from mask (1st column)
        att_mask = torch.squeeze(self.visited, dim=-2)[:, 1:] # (batch_size, 1, n_nodes) -> (batch_size, n_nodes-1)
        
        # Number of nodes in new instance after masking
        cur_num_nodes = self.n_loc + 1 - att_mask.sum(dim=1, keepdims=True) # (batch_size, 1)

        att_mask = torch.cat((torch.zeros(att_mask.shape[0], 1), att_mask), dim=-1) # add depot -> (batch_size, n_nodes)

        ones_mask = torch.ones_like(att_mask)

        # Create square attention mask.
        # In a (n_nodes, n_nodes) matrix this masks all rows and columns of visited nodes
        att_mask = AgentVRP.outer_pr(att_mask, ones_mask) \
                    + AgentVRP.outer_pr(ones_mask, att_mask) \
                    - AgentVRP.outer_pr(att_mask, att_mask) # (batch_size, n_nodes, n_nodes)
        return att_mask == 1, cur_num_nodes

    def all_finished(self):
        """ Checks if all routes are finished
        """
        return torch.all(self.visited == 1).item()

    def partial_finished(self):
        """Checks if partial solution for all graphs has been built; i.e. all agents came back to depot
        """
        return (torch.all(self.from_depot == 1) and self.i != 0).item()

batches = 3


depot = torch.randn(batches, 2)
loc = torch.randn(batches, 4, 2)
demand = torch.randint(9, (batches, 5))

agent = AgentVRP((depot, loc, demand))

print(agent.all_finished())
print('-------------------')
print(agent.partial_finished())
print('-------------------')
print(agent.get_att_mask())