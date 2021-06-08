from torch import nn

from models.modules.inception_modules import ConvBNReLU, InvertedResidualChannels, ConvSyncBNReLU, Conv, SPADEInvertedResidualChannels, InceptionSPADE
from models.modules.inception_architecture.inception_spade_generator import InceptionSPADEGenerator
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d


def transfer_Conv2d(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, nn.Conv2d) and isinstance(m2, nn.Conv2d)
    assert m1.in_channels == m1.groups or m1.groups == 1
    assert m2.in_channels == m2.groups or m2.groups == 1
    assert ((m1.in_channels == m1.groups)
            == (m2.in_channels == m2.groups)) or (m1.groups == m2.groups)
    dw = m1.in_channels == m1.groups
    if m1.out_channels == 3:
        assert input_index is not None
        if dw:
            m2.weight.data = m1.weight.data[input_index].clone()
        else:
            m2.weight.data = m1.weight.data[:, input_index].clone()
        if m2.bias is not None:
            m2.bias.data = m1.bias.data.clone()
        return None
    else:
        if m1.in_channels == 3:
            assert input_index is None
            input_index = [0, 1, 2]
        p = m1.weight.data

        if input_index is None:
            q = p.abs().sum([0, 2, 3])
            _, idxs = q.topk(m2.in_channels, largest=True)
            if dw:
                p = p[idxs]
            else:
                p = p[:, idxs]
        else:
            if dw:
                p = p[input_index]
            else:
                p = p[:, input_index]

        if output_index is None:
            q = p.abs().sum([1, 2, 3])
            _, idxs = q.topk(m2.out_channels, largest=True)
        else:
            idxs = output_index

        m2.weight.data = p[idxs].clone()
        if m2.bias is not None:
            m2.bias.data = m1.bias.data[idxs].clone()

        return idxs


def transfer_ConvTranspose2d(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, nn.ConvTranspose2d) and isinstance(
        m2, nn.ConvTranspose2d)
    assert m1.out_channels == m1.groups or m1.groups == 1
    assert m2.out_channels == m2.groups or m2.groups == 1
    assert ((m1.out_channels == m1.groups)
            == (m2.out_channels == m2.groups)) or (m1.groups == m2.groups)
    dw = m1.out_channels == m1.groups
    assert output_index is None
    p = m1.weight.data
    if input_index is None:
        q = p.abs().sum([1, 2, 3])
        _, idxs = q.topk(m2.in_channels, largest=True)
        p = p[idxs]
    else:
        p = p[input_index]
    q = p.abs().sum([0, 2, 3])
    _, idxs = q.topk(m2.out_channels, largest=True)
    if dw:
        m2.weight.data = p[idxs].clone()
    else:
        m2.weight.data = p[:, idxs].clone()
    if m2.bias is not None:
        m2.bias.data = m1.bias.data[idxs].clone()
    return idxs


def transfer_Norm(m1, m2, input_index=None, output_index=None):
    assert isinstance(
        m1, (nn.InstanceNorm2d,
             nn.BatchNorm2d, SynchronizedBatchNorm2d)) and isinstance(
                 m2,
                 (nn.InstanceNorm2d, nn.BatchNorm2d, SynchronizedBatchNorm2d))
    assert type(m1) == type(m2)
    if m1.weight is not None and m2.weight is not None:
        m2.weight.data = m1.weight.data[input_index].clone()
    if m1.bias is not None and m2.bias is not None:
        m2.bias.data = m1.bias.data[input_index].clone()
    return input_index


def transfer_ConvBNReLU(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, ConvBNReLU) and isinstance(m2, ConvBNReLU)
    idxs = transfer(m1[0], m2[0], input_index=input_index)
    idxs = transfer(m1[1], m2[1], input_index=idxs)
    return idxs


def transfer_InvertedResidualChannels(m1,
                                      m2,
                                      input_index=None,
                                      output_index=None):
    assert isinstance(m1, InvertedResidualChannels) and isinstance(
        m2, InvertedResidualChannels)
    assert output_index is None
    for res_op1, res_op2 in zip(m1.res_ops, m2.res_ops):
        idxs = input_index
        for layer1, layer2 in zip(res_op1, res_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvBNReLU):
                idxs = transfer(layer1, layer2, input_index=idxs)
            if isinstance(layer2, nn.Conv2d):
                idxs = transfer(layer1,
                                layer2,
                                input_index=idxs,
                                output_index=input_index)
    for dw_op1, dw_op2 in zip(m1.dw_ops, m2.dw_ops):
        idxs = input_index
        for layer1, layer2 in zip(dw_op1, dw_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvBNReLU):
                idxs = transfer(layer1, layer2, input_index=idxs)
            if isinstance(layer2, nn.Conv2d):
                idxs = transfer(layer1,
                                layer2,
                                input_index=idxs,
                                output_index=input_index)
    idxs = transfer(m1.pw_bn, m2.pw_bn, input_index=idxs)
    return idxs


def transfer_ConvSyncBNReLU(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, ConvSyncBNReLU) and isinstance(m2, ConvSyncBNReLU)
    idxs = transfer(m1.conv, m2.conv, input_index=input_index)
    idxs = transfer(m1.norm, m2.norm, input_index=idxs)
    return idxs


def transfer_Conv(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, Conv) and isinstance(m2, Conv)
    idxs = transfer(m1.conv, m2.conv, input_index=input_index)
    return idxs


def transfer_SPADEInvertedResidualChannels(m1,
                                           m2,
                                           input_index=None,
                                           output_index=None):
    assert isinstance(m1, SPADEInvertedResidualChannels) and isinstance(
        m2, SPADEInvertedResidualChannels)
    assert output_index is None
    idxs_first = transfer(m1.spade, m2.spade, input_index=input_index)
    for res_op1, res_op2 in zip(m1.res_ops, m2.res_ops):
        idxs = idxs_first
        for layer1, layer2 in zip(res_op1, res_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvBNReLU):
                idxs = transfer(layer1, layer2, input_index=idxs)
            if isinstance(layer2, nn.Conv2d):
                idxs = transfer(layer1, layer2, input_index=idxs)
    for dw_op1, dw_op2 in zip(m1.dw_ops, m2.dw_ops):
        idxs = idxs_first
        for layer1, layer2 in zip(dw_op1, dw_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvBNReLU):
                idxs = transfer(layer1, layer2, input_index=idxs)
            if isinstance(layer2, nn.Conv2d):
                idxs = transfer(layer1, layer2, input_index=idxs)
    if m1.shortcut is not None:
        assert m2.shortcut is not None
        idxs = transfer(m1.shortcut[0],
                        m2.shortcut[0],
                        input_index=input_index)
        idxs = transfer(m1.shortcut[1], m2.shortcut[1], input_index=idxs)
    else:
        assert m2.shortcut is None
    return idxs


def transfer_InceptionSPADE(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, InceptionSPADE)
    assert isinstance(m2, InceptionSPADE)
    idxs = transfer(m1.param_free_norm,
                    m2.param_free_norm,
                    input_index=input_index)
    for res_op1, res_op2 in zip(m1.res_ops, m2.res_ops):
        for layer1, layer2 in zip(res_op1, res_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvSyncBNReLU):
                idxs = transfer(layer1,
                                layer2,
                                input_index=list(range(
                                    layer1.conv.in_channels)))
            if isinstance(layer1, nn.Conv2d):
                idxs = transfer(layer1, layer2, idxs, input_index)
    for dw_op1, dw_op2 in zip(m1.dw_ops, m2.dw_ops):
        for layer1, layer2 in zip(dw_op1, dw_op2):
            assert type(layer1) == type(layer2)
            if isinstance(layer1, ConvSyncBNReLU):
                idxs = transfer(layer1,
                                layer2,
                                input_index=list(range(
                                    layer1.conv.in_channels)))
            if isinstance(layer1, nn.Conv2d):
                idxs = transfer(layer1, layer2, idxs, input_index)
    return input_index


def transfer(m1, m2, input_index=None, output_index=None):
    if isinstance(m1, nn.Conv2d):
        return transfer_Conv2d(m1, m2, input_index, output_index)
    elif isinstance(m1, nn.ConvTranspose2d):
        return transfer_ConvTranspose2d(m1, m2, input_index, output_index)
    elif isinstance(
            m1, (nn.InstanceNorm2d, nn.BatchNorm2d, SynchronizedBatchNorm2d)):
        return transfer_Norm(m1, m2, input_index, output_index)
    elif isinstance(m1, ConvBNReLU):
        return transfer_ConvBNReLU(m1, m2, input_index, output_index)
    elif isinstance(m1, InvertedResidualChannels):
        return transfer_InvertedResidualChannels(m1, m2, input_index,
                                                 output_index)
    elif isinstance(m1, ConvSyncBNReLU):
        return transfer_ConvSyncBNReLU(m1, m2, input_index, output_index)
    elif isinstance(m1, Conv):
        return transfer_Conv(m1, m2, input_index, output_index)
    elif isinstance(m1, InceptionSPADE):
        return transfer_InceptionSPADE(m1, m2, input_index, output_index)
    elif isinstance(m1, SPADEInvertedResidualChannels):
        return transfer_SPADEInvertedResidualChannels(m1, m2, input_index,
                                                      output_index)
    else:
        raise NotImplementedError('Unknown module [%s]!' % type(m1))


def load_pretrained_weight(model1, model2, netA, netB, ngf1, ngf2):
    assert ngf1 >= ngf2

    if isinstance(netA, nn.DataParallel):
        net1 = netA.module
    else:
        net1 = netA
    if isinstance(netB, nn.DataParallel):
        net2 = netB.module
    else:
        net2 = netB

    index = None
    if model1 == 'inception_9blocks':
        assert len(net1.down_sampling) == len(net2.down_sampling)
        assert len(net1.features) == len(net2.features)
        assert len(net1.up_sampling) == len(net2.up_sampling)
        for m1, m2 in zip(net1.down_sampling, net2.down_sampling):
            if isinstance(m1, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d,
                               nn.InstanceNorm2d)):
                index = transfer(m1, m2, index)
        for m1, m2 in zip(net1.features, net2.features):
            if isinstance(m1, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d,
                               nn.InstanceNorm2d, InvertedResidualChannels)):
                index = transfer(m1, m2, index)
        for m1, m2 in zip(net1.up_sampling, net2.up_sampling):
            if isinstance(m1, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d,
                               nn.InstanceNorm2d)):
                index = transfer(m1, m2, index)
    elif model1 == 'inception_spade':
        assert isinstance(net1, InceptionSPADEGenerator)
        assert isinstance(net2, InceptionSPADEGenerator)
        idxs = transfer(net1.fc, net2.fc, list(range(netA.fc.in_channels)))
        idxs = transfer(net1.fc_norm, net2.fc_norm, idxs)
        idxs = transfer(net1.head_0, net2.head_0, idxs)
        idxs = transfer(net1.G_middle_0, net2.G_middle_0, idxs)
        idxs = transfer(net1.G_middle_1, net2.G_middle_1, idxs)
        idxs = transfer(net1.up_0, net2.up_0, idxs)
        idxs = transfer(net1.up_1, net2.up_1, idxs)
        idxs = transfer(net1.up_2, net2.up_2, idxs)
        idxs = transfer(net1.up_3, net2.up_3, idxs)
        if hasattr(net1, 'up_4'):
            assert hasattr(netB, 'up_4')
            idxs = transfer(net1.up_4, net2.up_4, idxs)
        else:
            assert not hasattr(netB, 'up_4')
        idxs = transfer(netA.conv_img, net2.conv_img, idxs)
    else:
        raise NotImplementedError('Unknown model [%s]!' % model1)
