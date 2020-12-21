import torch
from utils.gen_data import get_gen_data


def compute_pairwise_dist(a, b):
    """
    Compute similarity matrix
    D_i,j is distance between a[:, :, i] and b[:, :, j]

    :param a:
    :param b:
    :return: dist
    """
    with torch.no_grad():
        r_a = torch.sum(a * a, dim=1, keepdim=True)  # (B,1,N)
        r_b = torch.sum(b * b, dim=1, keepdim=True)  # (B,1,M)
        mul = torch.matmul(a.permute(0, 2, 1), b)    # (B,N,M)
        dist = r_a.permute(0, 2, 1) - 2 * mul + r_b  # (B,N,M)
    return dist


def batched_index_select(x, dim, index):
    """
    Analog of index_select across batches

    :param x:
    :param dim:
    :param index:
    :return: Selected values across batches
    """
    for ii in range(1, len(x.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


def get_nn(pc1, pc2):
    """
    # Finds NN for each point in pc1 from pc2

    Args:
        pc1:
        pc2:

    Returns: nn

    """
    dist = compute_pairwise_dist(pc1, pc2)
    nn_idx = torch.argmin(dist, dim=2)
    nn_pc1 = batched_index_select(pc2, 2, nn_idx)
    return nn_pc1


def fb_consistency(model, pc1, pc2, sf_forward, data_gen):
    # Perform FB cycle
    pc2_transformed = pc1 + sf_forward

    # find NN for each point in pc2_transformed from pc2
    nn_pc2_transformed = get_nn(pc2_transformed, pc2)

    # compute averaged anchor
    pc2_transformed_nn_avg = (pc2_transformed + nn_pc2_transformed) / 2

    # compute generated data needed for the model
    aux_data = get_gen_data(pc2_transformed_nn_avg, pc1, data_gen, 3)

    # compute backward scene flow
    sf_backward, _, _, _ = model(pc2_transformed_nn_avg, pc1, aux_data, data_gen)
    pc1_cycle = pc2_transformed_nn_avg + sf_backward

    return pc2_transformed, nn_pc2_transformed, pc1_cycle


