from torch import nn
from torch.nn import functional as F

from models.modules.spade_modules.base_spade_distiller_modules import BaseSPADEDistillerModules
from utils import util
from utils.weight_transfer import load_pretrained_weight
from utils.common import KA


class SPADEDistillerModules(BaseSPADEDistillerModules):
    def __init__(self, opt):
        super(SPADEDistillerModules, self).__init__(opt)

    def profile(self, input_semantics):
        raise NotImplementedError('The distiller is only for training!!!')

    def calc_distill_loss(self, Tacts, Sacts):
        losses = {}
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, nn.Conv2d)
            layer = self.mapping_layers[i]
            Tact, Sact = Tacts[layer], Sacts[layer]
            if self.opt.distill_G_loss_type == 'mse':
                Sact = netA(Sact)
                loss = F.mse_loss(Sact, Tact)
            elif self.opt.distill_G_loss_type == 'ka':
                loss = -KA(Sact, Tact)
            else:
                raise NotImplementedError
            losses['G_distill%d' % i] = loss
        return sum(losses.values()) * self.opt.lambda_distill, losses

    def load_networks(self,
                      verbose=True,
                      teacher_only=False,
                      restore_pretrain=True):
        if self.opt.restore_pretrained_G_path is not None and restore_pretrain:
            util.load_network(self.netG_pretrained,
                              self.opt.restore_pretrained_G_path, verbose)
            load_pretrained_weight(self.opt.pretrained_netG,
                                   self.opt.student_netG, self.netG_pretrained,
                                   self.netG_student, self.opt.pretrained_ngf,
                                   self.opt.student_ngf)
            del self.netG_pretrained
        super(SPADEDistillerModules,
              self).load_networks(verbose, teacher_only=teacher_only)
