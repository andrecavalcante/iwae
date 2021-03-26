# Importance weighted autoencoder (IWAE)

A minimum Pytorch implementation of importance weighted autoencoder from Burda et al. (2016).
https://arxiv.org/abs/1509.00519

The code is commented regarding the equations in the paper. 

## Results

Negative log-likelihood of test data using k importance samples. 

 k  | Negative Log-Likelihood 
----| -----------------------
1   | 86.886
5   | 81.439
50  | 79.008
1000| 77.317

## Notes: 
1. I used the original MNIST dataset. In order to have a proper application of the discrete likelihood used this model, you may want to download the binarized version of MNIST referenced in the paper.
2. At this time torchvision cannot automatically download MNIST dataset. Apparently this is a server side issue. You can however download it manually and then set the root arg of torchvision.datasets.MNIST function with the proper local dataset directory. Keep the download=True flag.
3. I did not use the schedule for the Beta coefficient of ADAM optimizer used in the paper.  


