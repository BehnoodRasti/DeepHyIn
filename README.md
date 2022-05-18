# DeepHyIn
UNSUPERVISED DEEP HYPERSPECTRAL INPAINTING

We propose a novel model for hyperspectral inpainting in which the degraded hyperspectral image is a linear mixture of endmembers and degraded abundances. The proposed model is subjected to abundance sum to one and nonnegativity constraints. We further assume that the endmembers are known. Then, we propose an optimization problem to estimate the unknown abundance using a deep image prior. We shift the optimization problem to optimize the parameters of a deep network.


Paper: UNSUPERVISED DEEP HYPERSPECTRAL INPAINTING USING A NEW MIXING MODEL (IGARSS 2022)


![image](https://user-images.githubusercontent.com/61419984/169041279-63e3d5ed-cde7-4f93-9743-06e231311fb3.png)
![image](https://user-images.githubusercontent.com/61419984/169041386-8eee408a-fc82-4137-b07b-d84d971dce64.png)

Color image of the abundances; blue: water, red: tree, green: soil. From left to right: inpainted, downgraded, and ground truth abundances.

![image](https://user-images.githubusercontent.com/61419984/169041449-22c1652f-3cf1-46a5-97a9-27998a08a509.png)
