# Importance weighted autoencoder (IWAE)

A minimum implementation of importance weighted autoencoder by Burda et al. (2016) in Pytorch.
https://arxiv.org/abs/1509.00519

The code is commented regarding the equations in the paper. 


## Results

Negative log-likelihood of test data using k importance samples. 

 k | Negative Log-Likelihood 
---| -----------------------
1  | 86.886
5  | 81.439
50 | 79.008

# Notes: 
1. I use the original MNIST dataset. You may want to download the binarized version of MNIST referenced in the paper in order to compare the likelihood values with those from other methodologies.
2. At this time torchvision cannot automatically download MNIST dataset. Apparently this is a server side issue. You can however download it manually and then set the root arg of torchvision.datasets.MNIST function with the proper local dataset directory. Keep the download=True flag.
3. I did not use the schedule for the Beta coefficient for the ADAM optimizer. 
