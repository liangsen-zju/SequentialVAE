# implemented by p0werHu
# 11/15/2019

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Norm
import torch.distributions.kl as KL
from collections import OrderedDict

from common.debug import debug
from common.losses import WingLoss, L1Loss, SmoothL1Loss

class ConditionalVRNN(nn.Module):

    def __init__(self, opt):
        super(ConditionalVRNN, self).__init__()

        self.opt = opt
        self.dim_x = opt.MODEL.dim_x
        self.dim_h = opt.MODEL.dim_h  # 32
        self.dim_z = opt.MODEL.dim_z
        self.dim_c = opt.MODEL.dim_c
        self.n_rnn_layers = opt.MODEL.n_rnn_layers

        # feature extractors of x and z
        # paper: We found that these feature extractors are crucial for learnting complex sequences
        # paper: 'all of phi_t have four hidden layers using rectificed linear units ReLu'

        self.encode_c = nn.Sequential(
            nn.Linear(self.dim_c, self.dim_h),
            nn.ReLU()
        )

        self.encode_x = nn.Sequential(
            nn.Linear(self.dim_x, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU()
        )

        self.encoder_z = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU()
        )
        self.encoder_p = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU()
        )


        # encoder
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.dim_h*3, self.dim_h),
            nn.ReLU()
        )
        # VRE regard mean values sampled from z as the output
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_z),
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_z),
            nn.Softplus()
        )


        # prior
        self.fusion_prior = nn.Sequential(
            nn.Linear(self.dim_h*2, self.dim_h),
            nn.ReLU()
        )
        self.prior_mu = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_z),
        )
        self.prior_var = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_z),
            nn.Softplus()
        )

        # decoder
        self.fusion_decoder = nn.Sequential(
            nn.Linear(self.dim_h*3, self.dim_h),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_x),
            nn.Sigmoid()
        )
        
        # rnn
        self.fusion_update = nn.Sequential(
            nn.Linear(self.dim_h*3, self.dim_h),
            nn.ReLU()
        )

        # using the recurrence equation to update its hidden state
        self.RNN = nn.GRU(self.dim_h, hidden_size=self.dim_h, 
                          num_layers=self.n_rnn_layers, batch_first=True)
        
        self.loss_func_wing = WingLoss(reduction='mean')
        self.loss_func_L1 = SmoothL1Loss(reduction='mean')


    def reset_hidden(self, nbatch, device, random=False):
        if random:
            self.hidden = torch.rand((self.n_rnn_layers, nbatch, self.dim_h), device=device)
            self.feat_h = torch.rand((nbatch, 1, self.dim_h), device=device)
        else:
            self.hidden = torch.zeros((self.n_rnn_layers, nbatch, self.dim_h), device=device)
            self.feat_h = torch.zeros((nbatch, 1, self.dim_h), device=device)


    def generate(self, c, nsteps, device, nbatch=1):
        """
        c: [nbatch, N, dim_c]
        """
        
        toatl_predt = torch.zeros(nbatch, nsteps, self.dim_x)             # (B, N, dim_x)

        # reset hidden
        self.reset_hidden(nbatch, device, random=True)

        for t in range(nsteps):
            icond = c[:, t:t+1, :]                                 # B x 1 x  dim_c

            # print(f"cond ", icond[0, :, :])

            feat_c = self.encode_c(icond)
            debug(self.opt, f"[Genearate] feat_c = {feat_c.shape}", verbose=False) 

            # prior   
            pmu, plogvar = self.Prior(feat_c, self.feat_h)                 # B x 1 x dim_h

            # decoder
            z = self.Reparameterize(pmu, plogvar)

            feat_z = self.encoder_z(z)
            pred = self.Decoder(feat_z, feat_c, self.feat_h)               # B x 1 x dim_x

            # encoder x 
            feat_x = self.encode_x(pred)                           # B x dim_h

            #  recurrence
            self.feat_h, self.hidden = self.UpdateNet(feat_x, feat_z, feat_c, self.hidden)   # 

            # sample 
            toatl_predt[:, t:t+1, :] = pred.detach()


        return toatl_predt


    def forward(self, x, c):
        """
        x: [B, N, dim_x]
        c: [B, N, dim_c]

        :return:
        """

        B, nsteps, _ = x.size()                     # batch size
        self.targt = x.detach()                      # (B, N, dim_x)
        self.predt = torch.zeros_like(x)             # (B, N, dim_x)
        self.reset_hidden(B, x.device, random=True)

        self.total_prior_mu = []      # with gradient graph
        self.total_prior_va = []
        self.total_encod_mu = []
        self.total_encod_va = []
        self.total_predt = []


        for t in range(nsteps):
            # feature extractor:
            idata = x[:, t:t+1, :]                        # B x dim_x 
            icond = c[:, t:t+1, :]                        # B x dim_c

            # print(f"train cond ", icond[0, :, :])

            debug(self.opt, f"[Forward] x = {idata.shape}", verbose=False) 

            feat_x = self.encode_x(idata)             # B x dim_h
            feat_c = self.encode_c(icond)
            debug(self.opt, f"[Forward] feat_x = {feat_x.shape}, feat_c = {feat_c.shape}", verbose=False) 
            
            # prior
            pmu, plogvar = self.Prior(feat_c, self.feat_h)                 # B x dim_h

            xmu, xlogvar = self.Encoder(feat_x, feat_c, self.feat_h)          # B x dim_h
            debug(self.opt, f"[Forward] Prior pmu = {pmu.shape}", verbose=False) 
            debug(self.opt, f"[Forward] Encod xmu = {xmu.shape}", verbose=False) 
            

            # decoder
            z = self.Reparameterize(xmu, xlogvar)
            feat_z = self.encoder_z(z)
            pred = self.Decoder(feat_z, feat_c, self.feat_h)                  # B x 1 x dim_x
            debug(self.opt, f"[Forward] pred = {pred.shape}", verbose=False)    
            
            #  recurrence
            self.feat_h, self.hidden = self.UpdateNet(feat_x, feat_z, feat_c, self.hidden)   # 

            # save to vector 
            self.total_prior_mu.append(pmu)
            self.total_prior_va.append(plogvar)
            self.total_encod_mu.append(xmu)
            self.total_encod_va.append(xlogvar)
            self.total_predt.append(pred)

            self.predt[:, t:t+1, :]  = pred.detach()


        return self.predt


    def Prior(self, feat_c, feat_h):
        """Prior Network
            feat_h, the output of GRU,  [B, 1, dim_h]
        Return:
            mu, logvar, [B x 1 x dim_h]
        """
        
        feat_p = self.encoder_p(feat_h)
        debug(self.opt, f"[PriorNet] encoder shape = {feat_p.shape}")
        
        feat = torch.cat([feat_c, feat_p], dim=-1)        # B x 1 x 2*dim_h
        feat = self.fusion_prior(feat)
        debug(self.opt, f"[PriorNet] feat fusion shape = {feat.shape}")

        mu = self.prior_mu(feat)
        logvar = self.prior_var(feat)
        debug(self.opt, f"[PriorNet] mu, logvar shape = {mu.shape}", verbose=False)
        
        return mu, logvar


    def Encoder(self, feat_x, feat_c, feat_h):
        """Encoder function.
        Args:
            x:,     [B x 1 x dim_h]
            hidden: [B x 1 x dim_h]
        Return:
            mu, logvar,  [B x 1 x dim_h]
        """
        # concate with hidden
        feat = torch.cat([feat_x, feat_c, feat_h], dim=-1)        # B x 1 x 2*dim_h
        feat = self.fusion_encoder(feat)
        debug(self.opt, f"[Encoder] feat fusion shape = {feat.shape}")

        # inference
        mu     = self.encoder_mu(feat)                    # B x 1 x dim_h
        logvar = self.encoder_var(feat)                   # B x 1 x dim_h
        debug(self.opt, f"[Encoder] mu, logvar shape = {mu.shape}", verbose=False)

        return mu, logvar
   

    def Decoder(self, feat_z,  feat_c, feat_h):
        """Decoder function. mapping the given latent code
        Args:
            z,        [B x 1 x dim_h]
            feat_h:  [B x 1 x dim_h]

        Return:
            output:  [B x 1 x 136]
        """

        # concate with hidden
        feat = torch.cat([feat_z, feat_c, feat_h], dim=-1)               # B x 1 x 2*dim_h
        feat = self.fusion_decoder(feat)                         # B x 1 x dim_output

        x = self.output_layer(feat)                              # B x dim_x
        debug(self.opt, f"[Decoder] x output shape = {x.shape}")

        return x

    def UpdateNet(self, feat_x, feat_z, feat_c, hidden):
        """
        feat_x, the encoder feature of x_prev,  [B, 1, dim_h]
        hidden, the hidden state of h_(t-1), [B, D*nlyaers, dim_h]

        return:
        hidden state of h_t, [B, nlayer, dim_h]
        """
        feat = torch.cat([feat_x, feat_z, feat_c], dim=-1)         # (B, 1, 2*dim_h)
        feat = self.fusion_update(feat)                    # (B, 1, dim_h)
        debug(self.opt, f"[UpdateNet] x fusion shape = {feat.shape}")


        feat, hidden = self.RNN(feat, hidden)               # (B, 1, dim_h)
        debug(self.opt, f"[UpdateNet] update hidden shape = {hidden.shape}")
        return feat, hidden


    def sampling(self, seq_len, device):

        sample = torch.zeros(seq_len, self.dim_x, device=device)

        # h = torch.zeros(1, self.dim_h, device=device)
        h = torch.rand(1, self.dim_h, device=device)

        for t in range(seq_len):
            # prior
            feat = self.encoder_p(h)
            prior_means_ = self.prior_mu(feat)
            prior_var_ = self.prior_var(feat)

            # decoder
            z_t = self.Reparameterize(prior_means_, prior_var_)
            phi_z_t = self.encoder_z(z_t)
            decoder_fea_ = self.fusion_decoder(torch.cat([phi_z_t, h], dim=1))
            decoder_means_ = self.output_layer(decoder_fea_)

            phi_x_t = self.encode_x(decoder_means_)
            # rnn
            h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)

            sample[t] = decoder_means_.detach()

        return sample

    def Reparameterize(self, mu, logvar):
        """Reparameterize Trick function.
        sample from N(mu, var) from N(0, 1)
        Args:
            mu(torch.Tensor, [B x D]): The mean of the latent Gaussian
            logvar(torch.Tensor, [B x D]): The Log Standard deviation of the latent Gaussian
        Return:
            torch.Tensor, [B x D]: the reparameter data z 
        """
        std = torch.exp( 0.5 * logvar)       # B x seqlen x dim_h
        eps = torch.randn_like(std)          # B x seqlen x dim_h
        z = mu + eps * std                   # B x seqlen x dim_h

        return z 

    def eval_losses(self):
        return self.loss_function_(self.targt)

    def loss_function_(self, targt, **kwargs):
        """
        Computes the VAE loss function.
        """
        B, nsteps, _ = targt.size()      # batch size
        loss_dict = OrderedDict()

        loss = 0.
        loss_kld = 0.
        loss_rec = 0.

        for i in range(nsteps):
            # KL loss
            norm_pior = Norm.Normal(self.total_prior_mu[i], torch.exp(0.5 * self.total_prior_va[i]))  # (B, dim_z)
            norm_encd = Norm.Normal(self.total_encod_mu[i], torch.exp(0.5 * self.total_encod_va[i]))  # (B, dim_z)
            kld = torch.mean(KL.kl_divergence(norm_pior, norm_encd))
            loss_kld += kld

            # rec loss
            # rec = torch.mean(F.l1_loss(self.targt[:, i, :], self.total_predt[i]))
            rec = torch.mean(F.binary_cross_entropy(self.total_predt[i][:, 0, :], targt[:, i, :], reduction='none'))

            # rec = self.loss_func_wing(self.targt[:, i:i+1, :], self.total_predt[i])

            # print(f"self.targt[:, i, :] {self.targt[:, i:i+1, :].shape}, self.total_predt[i],{self.total_predt[i].shape}")
            loss_rec += rec

            
        # loss_kld = loss_kld / nsteps
        # loss_rec = loss_rec / nsteps 
        loss = loss_kld * self.opt.LOSS.lambda_kld + loss_rec * self.opt.LOSS.lambda_rec
 

        loss_dict["loss_rec"] = loss_rec.detach() / nsteps
        loss_dict["loss_kld"] = loss_kld.detach() / nsteps
        loss_dict["loss_all"] = loss.detach() / nsteps

        return loss, loss_dict
