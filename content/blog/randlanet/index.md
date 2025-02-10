---
date: 2022-05-12T00:00:00Z
author: "Mohammad Sadil Khan"
external_link: ""
summary: Large-Scale Point Cloud Segmentation Network
categories:
- Deep Learning
- Point Cloud
- Segmentation
- Encoder
- Decoder
title: RandLaNet
subtitle: Large-Scale Point Cloud Segmentation Network
layout: single-sidebar # single or single-sidebar
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
share: false
commentable: true
show_related: true
---

## 1. Point Cloud
  ### A. Introduction
  A Point Cloud is a set of points in 3D space which can represent the boundary or the whole object (including inside points). In a point cloud, the points are unordered and are not restricted by any grid which means a point cloud can be expressed in an infinite way (using translation). Each point can have 3D coordinates and feature vectors. $$P=\{(X_i,F_i)\}^{i=N}_{i=1}, X_i\in\mathbb{R}^3,F_i\in\mathbb{R}^d$$
  
  ### B. Properties of Point Cloud in $\mathbb{R}^3$
  
  <p>
 <ul>
    <li> $\textit{Unordered:}$ Unlike images or arrays, point cloud is unordered. It has no restriction to be confined within a boundary. This causes a problem for CNN type architecture to learn since CNN uses convolutional operations which requires ordered and regular array like representation of the input. Point cloud networks are generally invariant to the $N!$ number of permutations in input.</li>
   <li> $\textit{Irregularity:}$ Points are not sampled uniformly from an image which means different objects can have dense points while others sparse [1, 2]. This sometimes causes class imbalance problems in point cloud dataset.</li>
    <li> $\textit{Connectedness:}$ Since points are not connected like graph structure and neighbouring points contain meaningful spatial and geometry information of the object, networks must learn to pass information from points to points.</li>
</ul>
  </p>
  
## 2. RandLaNet - Architecture
  <p>
  Large-scale point cloud segmentation is a challenging task because of huge computational requirements and effective embedding learning. RandLa-Net[3] is an efficient and lightweight neural architecture that segments every point in large-scale point clouds. It is an encoder-decoder-like architecture that uses random sampling to downsample the input point cloud in the encoder and upsample the point cloud in decoder blocks. It uses random sampling compared to other sampling methods because of faster computation. Although random sampling can discard key points necessary for efficient point cloud segmentation, RandLa-Net implements attention-based local feature aggregation to effectively share features of points that are removed into the neighbor points. Figure[1] is the architecture of RandLa-Net. 
  <figure>
					<center><img src="sidebar-featured.png" width="800" /> </center>
					<figcaption class="figure-caption text-center">Figure 1:  RandLa-Net Architecture.  FC is the fully connected layer, LFA is the localfeature aggregation, RS is random sampling, MLP is shared multilayer perceptron,US is upsampling and DP is dropout.  (Image from [3])
					</figcaption>
				</figure>
  </p>
  
### A. Random Sampling

  <p>
   Compared to other sampling methods, Random sampling is extremely fast (time complexity $O(N)$). It is invariant to any changes to the points as well as the permutation of points. The random-sampling block is added in encoder part. To compensate for the loss of information, the author has added LFA module.
   <figure>
					<center><img src="randlanet_sampling.png" width="800" /> </center>
					<figcaption class="figure-caption text-center">Figure 2: Random Sampling in RandLa-Net.  The downsampling rate is a hyperparameter and has significant influence on model performance (Image from [3])
					</figcaption>
				</figure>
   
  </p>
  
### B. Architecture

  <p>
  RandLa-Net consists of 4 encoder and 4 decoder layers (Figure 1). Each encoder layer consists of LFA modules (which is shown in the bottom panel of Figure 3). LFA modules aggregate the local features and gradually expands the receptive field to perform global feature passing. Every LFA module is followed by a random sampling step. Let the input shape be $N\times d_n$, where $N$ is the number of points in the point clouds ($N\approx 10^6 - 10^7$) and $d_n \in \mathbb{R^d},d\geq3$). $d_n$ can contain the coordinates with other features like intensity, gradient or normal.
  </p>
  <p>
  $\textbf{Positional Encoding:}$Since point clouds are unstructured, positional encoding layer embeds the positional information in an 8 dimensional vector ($3\rightarrow 8$). This layer describes the location of a point by mapping the position/index of a point into a vector and assigning unique representation for every point. In this way, positional encoding layer makes the network more permutation-invariant.
  <p>
  <p>
  $\textbf{Encoding Layer:}$ The encoding layer progressively reduces the number of points and increases the point features. The point cloud is downsampled at each encoding layer after the dilated residual block by downsampling factor 4. $$N\rightarrow \frac{N}{4} \rightarrow \frac{N}{4^2} \rightarrow \frac{N}{4^3} \rightarrow \frac{N}{4^4}$$ The per-point feature dimension is increased gradually. $$8 \rightarrow 32 \rightarrow 128 \rightarrow 256 \rightarrow 512$$
  <p>
  <p>
  $\textbf{Decoding Layer:}$ In each decoder layer, points are upsampled. In each encoder layer, when a point is removed, it is stored as a reference. In subsequent decoding layer, (i.e the layer with which a skip connection is added from an encoder in Figure 1 for each query reference point, KNN is used to find the one nearest neighbor in the input set of points. Afterwards, feature of the nearest point is copied to the target point. Subsequently, the feature maps are concatenated with the feature maps produced by corresponding encoding layers through skip connections. Then a shared MLP is applied to the concatenated feature maps. Shared MLP means same MLP network for every point in the input point cloud.
  </p>
  <p>
  $\textbf{Final Output Layer:}$ The segmentation label is predicted through three fully connected layers $(N,64) \rightarrow (N,32) \rightarrow (N,C)$, where $C$ is the number of classes.
  </p>
  
## 3. RandLaNet - LFA

  <p>The Local Feature Aggregation follows a three-step message passing system. Since point cloud don't have connectivity information, LFA ensures features are shared between points. In Figure 1, the LFA module in the first encoder transforms the feature vector ($8 \rightarrow 32$) and random sampling removes 75% of the points. Let's take a point in the first encoder $(p,f),p\in \mathbb{R}^3,f\in \mathbb{R}^8$.
  <figure>
					<center><img src="randlanet_lfa.png" width="800" /> </center>
					<figcaption class="figure-caption text-center">Figure 3: RandLaNet Feature Sharing
					</figcaption>
	</figure>
	 Let's take an overview of how this happens before diving deep into it. 
   <ul>
  <li>$\textbf{1. Sampling:}$ The first step in message passing system is from which points we want to pass a message to the red point $p$ in Figure 3. K-Nearest Neighbor is used to find $K$ neighbor points (blue points) which will share its features with red point $p$.</li>
  <li> $\textbf{2. Message Generation:}$ Once we choose the points, we need to generate the message to send from blue points to red point. For every point, $p_i$, we will generate a message $f_i$ by incorporating the distance and spatial information using an MLP. This MLP will give us the desired dimension of feature vector for $f_i,\forall i=1,2,\cdots,K$.</li>
  <li> $\textbf{3. Message Passing:}$ There are several ways to share features from neighbor points. We can use MAX, AVG or SUM function. But the best method is use linear sum of the features $$f=\sum\limits_{i=1}^{6}\alpha_if_i$$, with $\alpha_i$ as learnable by the model. This $\alpha_i$ is the attention score. It makes sure to give more weights during aggregation to points of similar nature or belonging to the same object.
  </li>
  </ul>
  <figure>
					<center><img src="randlanet_lfav2.png" width="800" /> </center>
					<figcaption class="figure-caption text-center">Figure 4: LFA Module
					</figcaption>
	</figure>
	Figure 4 is the detailed view of LFA module. It consists of three neural units (1) Local Spatial Encoding(LocSE) (2) Attentive Pooling (3) Dilated Residual Block.
  </p>
  <p>$\textbf{A. Local Spatial Encoding}$</p>
  <p>
  Let $P=\{p_1,p_2,\cdots,p_n\},p_i \in \mathbb{R}^3 \text{ and } F=\{f_1,f_2,\cdots,f_n\}, f_i \in R^d$ be the point set and feature set accordingly. LSE units embed the features and the spatial information from the neighbourhood points. This helps the network learn the complex local geometrical structures with as increasing receptive field.
For every point $p_i$, first K-Nearest Algorithm is used for finding $K$ neighbor points. Let the set of neighbor points, $N(p_i)=\{p_1^{(i)},p_2^{(i)},\cdots,p_K^{(i)}\}$ and the set of features for the neighbor points be $N(f_i)=\{f_1^{(i)},f_2^{(i)},\cdots,f_K^{(i)}\}$. At first positional features for every point in $N(p_i)$ is encoded as follows. (Figure 4)
\begin{equation}
    r_k^{(i)}=MLP\bigg(p_i;p_k^{(i)};(p_i-p_k^{(i)});||p_i-p_k^{(i)}||\bigg), r_k^{(i)} \in \mathbb{R}^r
\end{equation}
$;$ is the concatenation layer and $||\cdot||$ is the $l_2$ distance between neighbor and center points. $r_k^{(i)}$ not only just concatenates two positions but also the effect of one point on another point in terms of distance is also added. Once $r_k^{(i)} ,\forall k=1,2,\cdots,K$ is computed it is concatenated with corresponding features in $N(f_i)$.
$$\hat{F}=\{\hat{F}_1,\hat{F}_2,\cdots,\hat{F}_i\},\hat{F_i}=\{\hat{f}_k^{(i)}\}_{k=1}^{k=K},
\hat{f}_k^{(i)}=\{r_k^{(i)};f_k^{(i)}\}$$
  </p>
  <p>
  $\textbf{B. Attentive Pooling}$</p>
  <p>
Attentive pooling aggregates the set of neighboring point features $\hat{F}$ with adaptive weights. Existing methods use mean or max pooling, resulting in the loss of important information. Attention mechanism will automatically learn important features. Given $\hat{F_i}=\{\hat{f}_1^{(i)},\cdots,\hat{f}_k^{(i)}\}$, first attention scores are computed using a shared MLP, $g$ such that
\begin{equation}
    s_k^{(i)}=g(\hat{f}_k^{(i)},W)
\end{equation}
where $W$ is the weight of the MLP. After learning the attention scores feature for point $p_i$, $f_i$ is updated with concatenated neighbor features. (Figure 4)
\begin{equation}
    \hat{f}_i=MLP(\sum\limits_{k=1}^{K}(\hat{f}_k^{(i)} \odot s_k^{(i)}))
\end{equation}
Together with LSE and Attentive pooling, the model learns informative features with geometric patterns for point $p_i$.
  </p>
  <p>
  $\textbf{C. Dilated Residual Block}$
  </p>
  <p>
Since the point cloud is downsampled, it is necessary to expand the receptive field to preserve geometric details. Inspired by Resnet architecture, the author stacks several LSE and attentive pooling in one block before downsampling. In Figure 6, the red points observe $K$ features from neighboring points after the first LSE and Attentive Pooling layer and then in the next step it learns from $K^2$ features (See Figure 5). However, the more layers are added, the more the model is likely to be over-fitted. In the original paper (Figure 5), only two layers of LSE and Attentive pooling are used.
<figure>
					<center><img src="randlanet_drb.png" width="800" /> </center>
					<figcaption class="figure-caption text-center">Figure 5: Dilated Residual Block
					</figcaption>
	</figure>
  
  <figure>
					<center><img src="randlanet_drb2.png" width="400" /> </center>
					<figcaption class="figure-caption text-center">Figure 6: Illustration of dilated residual block which expands the receptive field at each step.
					</figcaption>
	</figure>
  
  </p>
  
## 4. Conclusion

  <p>$\textbf{Advantages:}$</p>
  <p>
The main advantages of RandLa-Net are 
<ul>
  <li>It is lightweight and achieves state-of-the-art results compared to existing methods. The random sampling method reduces the computation.</li>
  <li> The proposed attention-based Local Feature Aggregation (LFA) can expand into larger receptive fields using Local Spatial Encoding (LSE) with attentive pooling of point and neighbor features.</li> 
  <li> The network consists of Shared MLP without any need of graph reconstruction or voxelization. </li>
  <li> The encoder-decoder architecture with downsampling aims to generate discriminative latent vectors using small samples which represent the objects of interest. </li>
  </ul>
  </p>
  <p>$\textbf{Disadvantages:}$</p>
  <p>
  <ul>
  <li>The random downsampling rate can influence the performance of the model. Reducing too many points will prevent the model from learning rich latent representations. </li>
  <li> Even though RandLaNet input allows addition of other features such as intensity, gradient, etc, it fails to learn local geometrical information. It learns the average shape of the object which causes over-segmentation. For more information, <a href="/publications/masterthesis/report.pdf">Thesis Report.</a> ( Look at the Modified RandLa-Net with Feature Extractor and Voxel Segmentation Results)

</li> 
  </ul>
  
  </p>
  
  
## 5. Bibliography

<ol>
<li>
  <p>Anh Nguyen, Bac Le, <a href="https://ieeexplore.ieee.org/document/6758588"><i>3D Point Cloud Segmentation - A Survey</i></a>, 2013 6th IEEE Conference on Robotics, Automation and Mechatronics (RAM), 2013, pp. 225-230.</p>
</li>
<li>
  <p>Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas, <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf"><i>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</i></a>, 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 77-85.</p>
</li>
<li>
  <p>Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, A. Trigoni, A. Markham, <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf"><i>RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds</i></a>,  2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).</p>
</li>
</ol>
