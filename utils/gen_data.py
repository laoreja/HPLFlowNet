import torch


def unsqueeze_gen_data(generated_data, dev='cuda'):
    """
    Add batch dimension for generated_data
    For now only supports batch size = 1

    :param generated_data:
    """
    for data_map in generated_data:
        for k in data_map:
            if not torch.is_tensor(data_map[k]):
                data_map[k] = torch.tensor(data_map[k])
            data_map[k] = torch.unsqueeze(data_map[k], 0).to(dev)


def get_gen_data(pc1, pc2, data_gen, max_layers=None):
    # compute generated data needed for the model
    data = data_gen.compute_generated_data(torch.squeeze(pc1).cpu(), torch.squeeze(pc2).cpu(), max_layers)
    # add batch dimension
    unsqueeze_gen_data(data)
    return data
