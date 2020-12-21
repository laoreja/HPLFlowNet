import torch


def unsqueeze_generated_data(generated_data):
    """
    Add batch dimension for generated_data
    For now only supports batch size = 1

    :param generated_data:
    """
    for data_map in generated_data:
        data_map['pc1_barycentric'] = torch.unsqueeze(data_map['pc1_barycentric'], 0)
        data_map['pc2_barycentric'] = torch.unsqueeze(data_map['pc2_barycentric'], 0)
        data_map['pc1_el_minus_gr'] = torch.unsqueeze(data_map['pc1_el_minus_gr'], 0)
        data_map['pc2_el_minus_gr'] = torch.unsqueeze(data_map['pc2_el_minus_gr'], 0)
        data_map['pc1_lattice_offset'] = torch.unsqueeze(data_map['pc1_lattice_offset'], 0)
        data_map['pc2_lattice_offset'] = torch.unsqueeze(data_map['pc2_lattice_offset'], 0)
        data_map['pc1_blur_neighbors'] = torch.unsqueeze(data_map['pc1_blur_neighbors'], 0)
        data_map['pc2_blur_neighbors'] = torch.unsqueeze(data_map['pc2_blur_neighbors'], 0)
        data_map['pc1_corr_indices'] = torch.unsqueeze(data_map['pc1_corr_indices'], 0)
        data_map['pc2_corr_indices'] = torch.unsqueeze(data_map['pc2_corr_indices'], 0)
        data_map['pc1_hash_cnt'] = torch.tensor([data_map['pc1_hash_cnt']])
        data_map['pc2_hash_cnt'] = torch.tensor([data_map['pc2_hash_cnt']])


def compute_pairwise_dist(xyz1, xyz2):
    """
    Compute similarity matrix
    M_i,j is distance between xyz1[:, :, i] and xyz1[:, :, j]

    :param xyz1:
    :param xyz2:
    :return: dist
    """
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=1, keepdim=True)  # (B,1,N)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=1, keepdim=True)  # (B,1,M)
    mul = torch.matmul(xyz1.permute(0, 2, 1), xyz2)       # (B,N,M)
    dist = r_xyz1.permute(0, 2, 1) - 2 * mul + r_xyz2     # (B,N,M)
    return dist


def batched_index_select(input_tensor, dim, index):
    """
    Analog of index_select across batches

    :param input_tensor:
    :param dim:
    :param index:
    :return: Selected values across batches
    """
    for ii in range(1, len(input_tensor.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_tensor, dim, index)
