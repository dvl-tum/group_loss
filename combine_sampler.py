from torch.utils.data.sampler import Sampler
import random


class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl):
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []

        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        for inds in l_inds:
            n_els = self.max - len(inds) + 1  # take out 1?
            inds.extend(inds[:n_els])  # max + 1

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]

        # shuffle the order of classes
        random.shuffle(split_list_of_indices)
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)
