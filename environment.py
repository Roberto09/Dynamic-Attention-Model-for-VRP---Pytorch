import torch

class AgentVRP():
    VEHICLE_CAPACITY = 1.0

    def __init__(self, input):
        depot = input[0] # (batch_size, 2)
        loc = input[1] # (batch_size, n_nodes, 2)
        self.demand = input[2] # (batch_size, n_nodes)

        self.batch_size, self.n_loc, _ = loc.shape

        # Coordinates of depot + other nodes -> (batch_size, 1+n_nodes, 2)
        self.coords = torch.cat((depot[:, None, :], loc), dim=-2)

        # Indices of graphs in batch
        self.ids = torch.arange(self.batch_size) # (batch_size)

        # State
        self.prev_a = torch.zeros(self.batch_size, 1)
        self.from_depot = self.prev_a == 0
        self.used_capacity = torch.zeros(self.batch_size, 1)

        # Nodes that have been visited will be marked with 1
        self.visited = torch.zeros(self.batch_size, 1, self.n_loc+1)

        # Step counter
        self.i = torch.zeros(1, dtype=torch.int64)

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

    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
        """

        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity
        exceeds_cap = self.demand + self.used_capacity > self.VEHICLE_CAPACITY

        # Maks nodes that area already visited or have too much demand or when they arrived to depot
        mask_loc = (visited_loc == 1) | (exceeds_cap[:, None, :]) \
            | ((self.i > 0) & self.from_depot[:, None, :])
        
        # We can choose depot if we are not in depot OR all nodes are visited
        # equivalent to: we mask the depot if we are in it AND there're still mode nodes to visit 
        mask_depot = self.from_depot[:, None, :] & ((mask_loc == False).sum(dim=-1, keepdims=True) > 0)

        return torch.cat([mask_depot, mask_loc], dim=-1)

    def step(self, action):

        # Update current state
        selected = action[:, None]
        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # Shift indices by 1 since self.demand doesn't consider depot
        selected_demand = self.demand.gather(-1, (self.prev_a - 1).clamp_min(0).view(-1, 1)) # (batch_size, 1)

        # Add current node capacity to used capacity and set it to 0 if we return from depot
        self.used_capacity = (self.used_capacity + selected_demand) * (self.from_depot == False)

        # Update visited nodes (set 1 to visited nodes)
        print(self.visited[self.ids, [0], action])
        self.visited[self.ids, [0], action] = 1
        
        self.i += 1

    @staticmethod
    def get_costs(dataset, pi):
        
        # Place nodes with coordinates in order of decoder tour
        loc_with_depot = torch.cat((dataset[0][:, None, :], dataset[1]), dim=1) # (batch_size, n_nodes, 2)
        d = loc_with_depot.gather(1, pi.view(pi.shape[0], -1, 1).repeat_interleave(2, -1))

        # Calculation of total distance
        # Note: first element of pi is not depot, but the first selected node in path
        # and last element from longest path is not depot

        return ((torch.norm(d[:, 1:] - d[:, :-1], dim=-1)).sum(dim=-1) # intra node distances
            + (torch.norm(d[:, 0] - dataset[0], dim=-1))  # distance from depot to first
            + (torch.norm(d[:, -1] - dataset[0], dim=-1))) # distance from last node of longest path to depot