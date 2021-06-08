"""Implement resource-aware inception block selection."""
import models.modules.inception_modules as incmod


def get_bn_to_prune(model, verbose=True, logger=None, spade=False):
    """Init information for inception block selection.

    Args:
        model: A model with method `get_named_block_list` which return all
            MobileNet V2 blocks with their names in `state_dict`.
        verbose: Log verbose info.

    Returns:
        An list of weight names for pruning.
    """
    weights = []
    for name, m in model.get_named_block_list().items():
        if spade:
            if isinstance(m, incmod.SPADEInvertedResidualChannels):
                for op, (bn_name, bn) in zip(
                        m.res_ops,
                        m.get_named_first_res_bn(prefix=name).items()):
                    weights.append('{}.weight'.format(bn_name))
                for op, (bn_name, bn) in zip(
                        m.dw_ops,
                        m.get_named_first_dw_bn(prefix=name).items()):
                    weights.append('{}.weight'.format(bn_name))
                for op, (bn_name, bn) in zip(
                        m.spade.res_ops,
                        m.spade.get_named_first_res_bn(prefix=name +
                                                       '.spade').items()):
                    weights.append('{}.weight'.format(bn_name))
                for op, (bn_name, bn) in zip(
                        m.spade.dw_ops,
                        m.spade.get_named_first_dw_bn(prefix=name +
                                                      '.spade').items()):
                    weights.append('{}.weight'.format(bn_name))
        else:
            if isinstance(m, incmod.InvertedResidualChannels):
                for op, (bn_name, bn) in zip(
                        m.res_ops,
                        m.get_named_first_res_bn(prefix=name).items()):
                    weights.append('{}.weight'.format(bn_name))
                for op, (bn_name, bn) in zip(
                        m.dw_ops,
                        m.get_named_first_dw_bn(prefix=name).items()):
                    weights.append('{}.weight'.format(bn_name))

    prune_weights = weights

    if verbose:
        for name in prune_weights:
            if logger is not None:
                logger.print_info('{}\n'.format(name))
            else:
                print('{}'.format(name))

    all_params_keys = [key for key, val in model.named_parameters()]
    for name_weight in prune_weights:
        assert name_weight in all_params_keys
    return prune_weights
