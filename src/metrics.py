
import torch.nn.functional as F

class PSNR:
    def __call__(self, output, targ):
        rmse = F.mse_loss(output, targ).sqrt()
        return 20 * (targ.max() / rmse).log10()
