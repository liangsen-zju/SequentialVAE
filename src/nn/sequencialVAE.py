import torch 
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from common.losses import WingLoss, L1Loss, SmoothL1Loss
from common.debug import debug

class SequencialVAE4(nn.Module):

    def __init__(self, opt):
        super(SequencialVAE4,self).__init__()

        self.opt = opt
        self.dim_output =  opt.MODEL.dim_output  # 68*2
        self.dim_latent =  opt.MODEL.dim_latent  # 32

        self.nseqR = self.opt.MODEL.n_lookprevs
        self.nseqM = self.nseqR + 1

        # encoder for x_prev
        # self.encoder_c = nn.Sequential(   # B x nseq x 136 x 1
        #     nn.Conv2d(self.nseqR, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 16 x 136 x 1
        #     #nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 68 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 34 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 17 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 9 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 128 x 5 x 1
        #     #nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 256 x 3 x 1
        #     #nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(256, self.dim_latent, kernel_size=(3,1), stride=(3,1), padding=(1,0), bias=True), # 256 x 1 x 1
        #     #nn.BatchNorm2d(self.dim_latent),
        #     nn.LeakyReLU(0.02, True)
        # )

        # # encoder for x_prev + xt
        # self.encoder_x = nn.Sequential(   # B x nseq x 136 x 1
        #     nn.Conv2d(1, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 16 x 136 x 1
        #     #nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 68 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 34 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 17 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 9 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 128 x 5 x 1
        #     #nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 256 x 3 x 1
        #     #nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(256, self.dim_latent, kernel_size=(3,1), stride=(3,1), padding=(1,0), bias=True), # 256 x 1 x 1
        #     #nn.BatchNorm2d(self.dim_latent),
        #     nn.LeakyReLU(0.02, True)
        # )

        # # encoder for x_prev
        # self.encoder_xp = nn.Sequential(  # B  x 2 x 68 x 1
        #     nn.Conv2d(2, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 16 x 68 x 1
        #     #nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 32 x 68 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 34 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 17 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 9 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 128 x 5 x 1
        #     #nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 256 x 3 x 1
        #     #nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(256, self.dim_latent, kernel_size=(3,1), stride=(3,1), padding=(1,0), bias=True), # 256 x 1 x 1
        #     #nn.BatchNorm2d(self.dim_latent),
        #     nn.LeakyReLU(0.02, True)
        # )
        # # encoder for xt
        # self.encoder_xt = nn.Sequential(  # B  x 2 x 68 x 1
        #     nn.Conv2d(2, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 16 x 68 x 1
        #     #nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),  # 32 x 68 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 32 x 34 x 1
        #     #nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 17 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 9 x 1
        #     #nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True),  # 128 x 5 x 1
        #     #nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 256 x 3 x 1
        #     #nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.02, True),
        #     nn.Conv2d(256, self.dim_latent, kernel_size=(3,1), stride=(3,1), padding=(1,0), bias=True), # 256 x 1 x 1
        #     #nn.BatchNorm2d(self.dim_latent),
        #     nn.LeakyReLU(0.02, True)
        # )

        self.encoder_xt = nn.Sequential(                  # (B, 1, 136)
            nn.Linear(136, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )
        
        self.encoder_xp = nn.Sequential(                  # (B, 1, 136)
            nn.Linear(136, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )
        
        # decoder
        self.encoder_z = nn.Sequential(
            nn.Linear(self.dim_latent, out_features=128, bias=True),
            nn.LeakyReLU(0.02),

            nn.Linear(128, out_features=64, bias=True),
            nn.LeakyReLU(0.02),

            nn.Linear(64, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )

        # latent space
        self.prior_mu  = nn.Linear(self.dim_latent,   self.dim_latent)
        self.prior_var = nn.Linear(self.dim_latent,   self.dim_latent)
        self.encoder_mu  = nn.Linear(self.dim_latent, self.dim_latent)
        self.encoder_var = nn.Linear(self.dim_latent, self.dim_latent)

        # update
        self.GRU = nn.GRU(self.dim_latent, hidden_size=self.dim_latent, num_layers=3, batch_first=True)
        self.hidden = None

        self.fusion_prior = nn.Sequential(
            nn.Linear(2*self.dim_latent, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(2*self.dim_latent, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )

        self.fusion_update = nn.Sequential(
            nn.Linear(2*self.dim_latent, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=self.dim_latent, bias=True),
            nn.LeakyReLU(0.02)
        )

        self.fusion_decoder = nn.Sequential(
            nn.Linear(2*self.dim_latent, out_features=128, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(128, out_features=64, bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(64, self.dim_output, bias=True),
            nn.Tanh()
        )


        self.loss_func_wing = WingLoss(reduction='mean')
        self.loss_func_L1 = SmoothL1Loss(reduction='mean')


    def reset_hidden(self, nbatch, device):
        self.hidden = torch.zeros((3, nbatch, self.dim_latent), device=device)
        self.feat_h = torch.zeros((nbatch, 1, self.dim_latent), device=device)


    def generate(self, x_prev):
        """Forward function
        Args:
            x_prev: landmark, [b x 1 x 68 x 2]
        """  
        B  = x_prev.size(0)

        # x_prev encoder
        xp = x_prev.view(B, -1,  68*2)                                     # B x 2 x 68 x 1
        debug(self.opt, f"[Generate] xp permute shape = {xp.shape}")
        feat_xp = self.encoder_xp(xp).view(B, 1, -1)                       # B x 1 x dim_latent 
        debug(self.opt, f"[Generate] xp encode shape = {feat_xp.shape}")

        # prior
        pmu, plogvar = self.PriorNet(feat_xp, self.feat_h)              # B x 1 x dim_latent
        debug(self.opt, f"[Generate] pmu[0] = {pmu[0, 0, ::30]}", verbose=True)
        debug(self.opt, f"[Generate] pva[0] = {plogvar[0, 0, ::30]}", verbose=True)
        

        # sampling
        z = self.Reparameterize(pmu, plogvar)                   # B x 1 x dim_latent
        x, feat_z = self.Decoder(z, self.feat_h)                # B x 1 x 68 x 2


        # x encoder
        feat_x = x.view(B, -1,  68*2)                                     # B x 1 x 136
        feat_x = self.encoder_xt(feat_x).view(B, 1, -1)                    # B x 1 x dim_latent 
        debug(self.opt, f"[Generate] xt encode shape = {feat_x.shape}")

        # GRU, get hidden h_{t-1}
        self.feat_h, self.hidden = self.UpdateNet(feat_x, feat_z, self.hidden)   # 


        return x


    def forward(self, xt, x_prev, use_pt=False):
        """Forward function
        Args:
            xt: landmark, [b x 1 x 68 x 2]
            x_prev: landmark, [b x 1 x 68 x 2]
        """    

        B  = xt.size(0)
        self.targt = xt.detach()                  # (B, 1, 68, 2)

        # prior
        xp = x_prev.view(B, -1,  68*2)                                     # B x 2 x 68 x 1
        feat_xp = self.encoder_xp(xp).view(B, 1, -1)                       # B x 1 x dim_latent 
        self.pmu, self.plogvar = self.PriorNet(feat_xp, self.feat_h)              # B x 1 x dim_latent
        # debug(self.opt, f"[Forward] pmu[0] = {self.pmu[0, 0, ::25]}", verbose=False)
        # debug(self.opt, f"[Forward] pva[0] = {self.plogvar[0, 0, ::25]}", verbose=False)

        # encoder
        xt = xt.view(B, -1,  68*2)                                         # B x 1 x 136
        feat_xt = self.encoder_xt(xt).view(B, 1, -1)                       # B x 1 x dim_latent 
        # debug(self.opt, f"[Encoder] xt encode shape = {feat_xt.shape}")
        self.xmu, self.xlogvar = self.Encoder(feat_xt, self.feat_h)      # B x 1 x dim_latent
        # debug(self.opt, f"[Forward] xmu[0] = {self.xmu[0, 0, ::25]}", verbose=False)
        # debug(self.opt, f"[Forward] xva[0] = {self.xlogvar[0, 0, ::25]}", verbose=False)


        if torch.rand(1)[0] < self.opt.MODEL.p_use_pt:
            mu, logvar = self.pmu, self.plogvar
        else:
            mu, logvar = self.xmu, self.xlogvar

        # sampling and reparameter
        z = self.Reparameterize(mu, logvar)                     # B x 1 x dim_latent
        self.predt, feat_z = self.Decoder(z, self.feat_h)       # B x 1 x 68 x 2


        if torch.rand(1)[0] < self.opt.MODEL.p_use_pt:
            with torch.no_grad():
                feat_xp = self.predt.view(B, -1,  68*2)                  # B x 1 x 136
                feat_xp = self.encoder_xt(feat_xp).view(B, 1, -1)        # B x 1 x dim_latent 
            feat_xt = (feat_xt + feat_xp) - feat_xt                      # skip grad
                
        #  recurrence
        self.feat_h, self.hidden = self.UpdateNet(feat_xt, feat_z, self.hidden)   # 


        return self.predt


    def PriorNet(self, feat_xp, feat_h):
        """Pior Network.
            feat, the output of GRU,  [B, 1, dim_latent]
        Return:
            mu, logvar, [B x 1 x dim_latent]
        """  
        feat = torch.cat([feat_xp, feat_h], dim=-1)  # B x 1 x 2*dim_latent
        feat = self.fusion_prior(feat)
        debug(self.opt, f"[PriorNet] feat fusion shape = {feat.shape}")

        # feat = self.fusion_prior(feat_h)                       # (B, 1, 128)
        # debug(self.opt, f"[PriorNet] x prior_net shape = {feat.shape}", verbose=False)

        mu     = self.prior_mu(feat)            # B x 1 x dim_latent
        logvar = self.prior_var(feat)           # B x 1 x dim_latent
        debug(self.opt, f"[PriorNet] mu, logvar shape = {mu.shape}", verbose=False)

        return mu, logvar

    def Encoder(self, feat_xt, feat_h):
        """Encoder function.
        Args:
            x:,     [B x 1 x dim_latent]
            hidden: [B x 1 x dim_latent]
        Return:
            mu, logvar,  [B x 1 x dim_latent]
        """
        # concate with hidden
        feat = torch.cat([feat_xt, feat_h], dim=-1)  # B x 1 x 2*dim_latent
        feat = self.fusion_encoder(feat)
        debug(self.opt, f"[Encoder] feat fusion shape = {feat.shape}")

        # inference
        mu     = self.encoder_mu(feat)                    # B x 1 x dim_latent
        logvar = self.encoder_var(feat)                   # B x 1 x dim_latent
        debug(self.opt, f"[Encoder] mu, logvar shape = {mu.shape}", verbose=False)

        return mu, logvar

    def Reparameterize(self, mu, logvar):
        """Reparameterize Trick function.
        sample from N(mu, var) from N(0, 1)
        Args:
            mu(torch.Tensor, [B x D]): The mean of the latent Gaussian
            logvar(torch.Tensor, [B x D]): The Log Standard deviation of the latent Gaussian
        Return:
            torch.Tensor, [B x D]: the reparameter data z 
        """
        std = torch.exp( 0.5 * logvar)       # B x seqlen x dim_latent
        eps = torch.randn_like(std)          # B x seqlen x dim_latent
        z = mu + eps * std                   # B x seqlen x dim_latent

        return z 

    def Decoder(self, z,  feat_h):
        """Decoder function. mapping the given latent code
        Args:
            z,        [B x 1 x dim_latent]
            feat_h:  [B x 1 x dim_latent]

        Return:
            output:  [B x 1 x 136]
        """
        # decoder
        feat_z = self.encoder_z(z)                              # B x 1 x dim_latent
        debug(self.opt, f"[Decoder] feat_z shape = {feat_z.shape}")
        

        # concate with hidden
        x = torch.cat([feat_z, feat_h], dim=-1)               # B x 1 x 2*dim_latent
        x = self.fusion_decoder(x)                             # B x 1 x dim_output
        x = x.reshape(-1, 1, 68, 2)                            # B x 1 x 68 x 2
        debug(self.opt, f"[Decoder] x output shape = {x.shape}")

        return x, feat_z

    def UpdateNet(self, feat_x, feat_z, hidden):
        """
        feat_x, the encoder feature of x_prev,  [B, 1, dim_latent]
        hidden, the hidden state of h_(t-1), [B, D*nlyaers, dim_latent]

        return:
        hidden state of h_t, [B, nlayer, dim_latent]
        """
        feat = torch.cat([feat_x, feat_z], dim=-1)         # (B, 1, 2*dim_latent)
        feat = self.fusion_update(feat)                  # (B, 1, dim_latent)
        debug(self.opt, f"[UpdateNet] x fusion shape = {feat.shape}")


        feat, hidden = self.GRU(feat, hidden)               # (B, 1, dim_latent)
        debug(self.opt, f"[UpdateNet] update hidden shape = {hidden.shape}")
        return feat, hidden


    def eval_losses(self):
        return self.loss_function_(self.predt, self.targt)

    def loss_function_(self, predt, targt, **kwargs):
        """
        Computes the VAE loss function.
        """
        loss_dict = OrderedDict()

        # reconstruction loss
        rec = self.loss_func_wing(predt.reshape(-1, 136), targt.reshape(-1, 136))
        rec = rec * self.opt.LOSS.lambda_rec

        loss_dict["loss_rec"] = rec
       

        # KL loss
        # loss_kld = - 0.5 * torch.sum(1 + self.logvar - self.mu ** 2 - self.logvar.exp(), dim = 2)
        xstd = torch.exp(0.5 * self.xlogvar)   # 
        pstd = torch.exp(0.5 * self.plogvar)
        kld = - 0.5 * (1 + self.xlogvar - self.plogvar - xstd**2 - (self.xmu - self.pmu)**2 / pstd**2 )
        kld = torch.sum(kld, dim=-1)   # (B, 1)
        kld = torch.mean(kld) * self.opt.LOSS.lambda_kld

        loss_dict["loss_kld"] = kld

        loss = rec + kld

        # return
        return loss, loss_dict
