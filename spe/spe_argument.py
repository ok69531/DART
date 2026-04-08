import argparse


def load_spe_args():
    
    parser = argparse.ArgumentParser()

    # model attributes
    parser.add_argument('--node_emb_dims', default = 128, type = int)
    parser.add_argument('--pooling', default = 'mean', type = str)

    # phi (GIN) attributes
    parser.add_argument('--pe_dims', default = 37, type = int, help = 'The number of used eigenvalues')
    parser.add_argument('--n_phi_layers', default = 8, type = int)
    parser.add_argument('--phi_hidden_dims', default = 128, type = int)

    # psi (MLP) attributes
    parser.add_argument('--n_psis', default = 16, type = int)
    parser.add_argument('--n_psi_layers', default = 3, type = int)
    parser.add_argument('--psi_hidden_dims', default = 16, type = int)
    parser.add_argument('--psi_activation', default = 'relu', type = str)
    parser.add_argument('--pe_aggregate', default = 'add', type = str)

    # base model (GINE) attributes
    parser.add_argument('--n_base_layers', default = 4, type = int)
    parser.add_argument('--base_hidden_dims', default = 128, type = int)

    parser.add_argument('--n_mlp_layers', default = 3, type = int)
    parser.add_argument('--mlp_hidden_dims', default = 128, type = int)
    parser.add_argument('--mlp_use_bn', default = True, type = bool)
    parser.add_argument('--mlp_use_ln', default = False, type = bool)
    parser.add_argument('--mlp_activation', default = 'relu', type = str)
    parser.add_argument('--mlp_dropout_prob', default = 0.0, type = float)

    parser.add_argument('--residual', default = True, type = bool)
    parser.add_argument('--graph_norm', default = False, type = bool)
    parser.add_argument('--batch_norm', default = False, type = bool)

    # data attributes
    # parser.add_argument('--use_subset', default = True, type = bool)
    parser.add_argument('--train_batch_size', default = 128, type = int)
    parser.add_argument('--val_batch_size', default = 128, type = int)
    parser.add_argument('--tier', default = 1, type = int)
    parser.add_argument('--assay_name', default = None, type = str)

    # optimizer attributes
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--weight_decay', default = 3e-6, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--nesterov', default = False, type = bool)

    # scheduler attributes
    parser.add_argument('--n_warmup_steps', default = 100, type = int)

    # miscellaneous
    parser.add_argument('--n_epochs', default = 1000, type = int)
    
    # task
    parser.add_argument('--task', default = 'reg', type = str, help = 'reg, cls')
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    return args
