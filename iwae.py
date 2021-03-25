import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class IWAE(nn.Module):
    def __init__(self, x_dim=784, h_dim=400):
        super(IWAE, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        # encoder network for computing mean and std of a Gaussian proposal q(h|x)
        self.encoder_base = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh())
        self.encoder_q_mean = nn.Sequential(
            self.encoder_base, 
            nn.Linear(200, h_dim))
        self.encoder_q_logvar = nn.Sequential(
            self.encoder_base,
            nn.Linear(200, h_dim))

        # decoder network for computing mean of a Bernoulli likelihood p(x|h)
        self.decoder_p_mean = nn.Sequential(
            nn.Linear(h_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, x_dim),
            nn.Sigmoid())

    def forward(self, x, num_samples):
        
        # computing mean and std of Gaussian proposal q(h|x)
        q_mean = self.encoder_q_mean(x)
        q_logvar = self.encoder_q_logvar(x)
        q_std = torch.exp(q_logvar / 2)

        # replicating mean and std to generate multiple samples. Unsqueezing to handle batch sizes bigger than 1.
        q_mean = torch.repeat_interleave(q_mean.unsqueeze(1), num_samples, dim=1)
        q_std = torch.repeat_interleave(q_std.unsqueeze(1), num_samples, dim=1)

        # generating proposal samples
        # size of h: (batch_size, num_samples, h_size)
        h = q_mean + q_std * torch.randn_like(q_std)
        
         # computing mean of likelihood Bernoulli p(x|h)
        likelihood_mean = self.decoder_p_mean(h)

        # log p(x|h)
        x = x.unsqueeze(1) # unsqueeze for broadcast
        log_px_given_h = torch.sum(x * torch.log(likelihood_mean) + (1-x) * torch.log(1 - likelihood_mean), dim=-1) # sum over num_samples

        # gaussian prior p(h)
        log_ph = torch.sum(-0.5* torch.log(torch.tensor(2*np.pi)) - torch.pow(0.5*h,2), dim=-1) # sum over num_samples

        # evaluation of a gaussian proposal q(h|x)
        log_qh_given_x = torch.sum(-0.5* torch.log(torch.tensor(2*np.pi))-torch.log(q_std) - 0.5*torch.pow((h-q_mean)/q_std, 2), dim=-1)
        
        # computing log weights 
        log_w = log_px_given_h + log_ph - log_qh_given_x
       
        # normalized weights through Exp-Normalization trick
        M = torch.max(log_w, dim=-1)[0].unsqueeze(1)
        normalized_w =  torch.exp(log_w - M)/ torch.sum(torch.exp(log_w - M), dim=-1).unsqueeze(1) # unsqueeze for broadcast

        # loss signal        
        loss = torch.sum(normalized_w.detach().data * (log_px_given_h + log_ph - log_qh_given_x), dim=-1) 
        loss = -torch.mean(loss) # mean over batchs

        # computing log likelihood through Log-Sum-Exp trick
        log_px = M + torch.log((1/num_samples)*torch.sum(torch.exp(log_w - M), dim=-1))
        log_px = torch.mean(log_px) # mean over batches
        
        return likelihood_mean, log_px, loss
 
def main():
    batch_size = 250
    x_dim = 28*28
    h_dim = 50
    num_samples = 5
    num_epochs = 35
    lr = 10e-4

    train_dataset = torchvision.datasets.MNIST(root='C:/Users/Andre/Dropbox/iwae',train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=batch_size,shuffle=True)

    model = IWAE(x_dim, h_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for _, (images,_) in enumerate(train_loader):
            optimizer.zero_grad()
            
            x = images.cuda().view(batch_size, x_dim)
            reconstructed_x, log_px, loss = model(x, num_samples)           
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}],  loss: {:.3f}'.format(epoch + 1, num_epochs, loss.item()))
        print('Epoch [{}/{}],  negative log-likelihood: {:.3f}'.format(epoch + 1, num_epochs, - log_px.item()))

    plt.imshow(reconstructed_x[0,0].detach().cpu().numpy().reshape(28,28))
    plt.show()

if __name__ == '__main__':
    main()