"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.gcn_net import TSLANet

def MultiplexedNet(configs):

    if hasattr(configs, '__dict__'):
        config_dict = vars(configs)
    else:
        config_dict = configs
    
    return TSLANet(
        input_length=config_dict.get('seq_len', 120),
        num_nodes=config_dict.get('enc_in', 116),
        nhid=config_dict.get('d_model', 128),
        nclass=config_dict.get('c_out', 2),
        dropout=config_dict.get('dropout', 0.3),
        node_feature_dim=config_dict.get('d_model', 128),
        num_layers=config_dict.get('n_layers', 2),
        leaky_slope=config_dict.get('leaky_slope', 0.1)
    )

