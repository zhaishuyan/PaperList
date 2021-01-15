#  PaperList
 Organize a list of papers 🎈

[中国计算机学会推荐国际学术会议和期刊目录-2019](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=144845) [[PDF]](./paper/中国计算机学会推荐国际学术会议和期刊目录-2019.pdf)



####  Talking face generation

- 3D based approaches

  - [ ] [Real-time facial animation with image-based dynamic avatars.]() Transactions on Graphics, 2016

  - [ ] [pagan: real-time avatars using dynamic textures.]() SIGRAPH Asia, 2018. (generates key face expression textures that can be deformed and blended in real-time.)

- 2D landmark based approaches
  
  - [x] [Hierarchical Cross-Modal Talking Face Generation with Dynamic Pixel-Wise Loss.](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Hierarchical_Cross-Modal_Talking_Face_Generation_With_Dynamic_Pixel-Wise_Loss_CVPR_2019_paper.html) [[PDF]](./paper/Hierarchical_Cross_Modal_Talking_Face_Generation_with_Dynamic_Pixel_Wise_Loss.pdf) CVPR, 2019.
  - [x] [MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation.](https://wywu.github.io/projects/MEAD/MEAD.html) [[PDF]](./paper/MEAD.pdf) ECCV, 2020.
  - [x] [Talking Face Generation with Expression-Tailored Generative Adversarial Network.](https://dl.acm.org/doi/abs/10.1145/3394171.3413844) [[PDF]](./paper/Talking_Face_Generation_with_Expression_Tailored_Generative_Adversarial_Network.pdf) ACM MM, 2020. (emotion, identity, speech, 多模态合成)
  - [ ] [Fast face-swap using convolutional neural networks.](https://arxiv.org/abs/1611.09577) ICCV, 2017.
  - [ ] [X2face: A network for controlling face generation using images, audio, and pose codes.](https://openaccess.thecvf.com/content_ECCV_2018/html/Olivia_Wiles_X2Face_A_network_ECCV_2018_paper.html) [[PDF]](./paper/X2Face.pdf) ECCV, 2018.
  - [ ] [FSGAN: Subject agnostic face swapping and reenactment.](https://openaccess.thecvf.com/content_ICCV_2019/html/Nirkin_FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment_ICCV_2019_paper.html) [[PDF]](./paper/FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment.pdf) ICCV, 2019.
- Vid2vid approaches
  - [ ] [Few-shot video-to-video synthesis.]() NeurIPS, 2019.
  - [ ] [Video-to-video synthesis.]() NeurIPS, 2018.
  - [x] [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild.](https://dl.acm.org/doi/10.1145/3394171.3413532) [[PDF]](./paper/A_Lip_Sync_Expert_Is_All_You_Need_for_Speech_to_Lip_Generation_In_The_Wild.pdf) ACM Multimedia 2020. 
  - [ ] [Towards Automatic Face-to-Face Translation.](https://dl.acm.org/doi/10.1145/3343031.3351066) [[PDF]](./paper/Towards_Automatic_Face_to_Face_Translation.pdf) ACM Multimedia 2020. (LipGAN, face2face translation)
  
- Disentanglement based approaches

  - [x] [Talking Face Generation by Adversarially Disentangled Audio-Visual Representation]() [[PDF] ](./paper/Talking Face Generation by Adversarially Disentangled Audio-Visual Representation.pdf)CVPR, 2019. 

  - [x] [Mittal_Animating_Face_using_Disentangled_Audio_Representations.](https://openaccess.thecvf.com/content_WACV_2020/papers/Mittal_Animating_Face_using_Disentangled_Audio_Representations_WACV_2020_paper.pdf)  WACV, 2020. (解耦emotion和content)

    

#### Lip reading

- [x] (SyncNet) [Out of Time: Automated Lip Sync in the Wild.](https://www.researchgate.net/publication/315311266_Out_of_Time_Automated_Lip_Sync_in_the_Wild) [[PDF]](./paper/SyncNet.pdf) ACCV, 2016.



#### Landmark processing

- [x] [Low Bandwidth video-chat compression using deep generative models.](https://arxiv.org/abs/2012.00328) [[PDF]](./paper/Low_Bandwidth_Video_Chat_Compression_using_Deep_Generative_Models.pdf) arxiv, 2020.
- [ ] [Lifting 2D styleGan for 3D-aware face generation.](https://arxiv.org/abs/2011.13126) [[PDF]](./paper/Lifting_2D_StyleGAN_for_3D_Aware_Face_Generation.pdf) arxiv, 2020.
- [ ] Fast bi-layer neural synthesis of oneshot realistic head avatars. ECCV, 2020.
- [ ] Few-shot adversarial learning of realistic neural talking head models. ICCV, 2019.
- [ ] Semantic image synthesis with spatially-adaptive normalization. CVPR, 2019.
- [ ] First order motion model for image animation. NeurIPS, 2019.
- [ ] face reenactment



#### Emotion animation

- [x] [GANimation: Anatomically-aware Facial Animation from a Single Image.]() [[PDF]](./paper/GANimation.pdf) CVPR, 2018. (Generates attention mask and color mask by a conditional GAN)



#### Video prediction / generation



#### Emotion recognition

- [ ] [M3ER: Multiplicative Multimodal Emotion Recognition using Facial, Textual, and Speech Cues.](https://ojs.aaai.org//index.php/AAAI/article/view/5492) [[PDF]](./paper/https://ojs.aaai.org//index.php/AAAI/article/view/5492.pdf) AAAI, 2020. (从三个模态提取信息识别一个视频的emotion)



#### GAN

- [x] (Pix2PixGAN) [Image-to-Image Translation with Conditional Adversarial Networks.](https://ieeexplore.ieee.org/document/8100115/) [[PDF]](./paper/pix2pixgan.pdf) [[Github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) CVPR, 2017.

- [x] (CycleGAN) [Unpaired Image-to-Image Translationusing Cycle-Consistent Adversarial Networks.](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) [[PDF]](./paper/cyclegan.pdf) [[Github]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) ICCV, 2017.

- StyleGAN and its derivative

  - [x] (StyleGAN) [A Style-Based Generator Architecture for Generative Adversarial Networks.](https://arxiv.org/pdf/1812.04948.pdf) [[Which face is real?]](http://www.whichfaceisreal.com/learn.html)[[PDF]](./paper/StyleGAN.pdf)[[GitHub]](https://github.com/NVlabs/stylegan)[[中文Blog]](https://zhuanlan.zhihu.com/p/63230738)[[YouTube]](https://www.youtube.com/watch?v=dCKbRCUyop8) [[Bilibili]](https://www.bilibili.com/video/BV1ME411d7Y7?from=search&seid=346330266334742148) CVPR, 2019. Nvidia.
  - [x] (StyleGAN2) [Analyzing and Improving the Image Quality of StyleGAN.](https://ieeexplore.ieee.org/document/9156570) [[PDF]](./paper/StyleGAN2.pdf) [[GitHub]](https://github.com/NVlabs/stylegan2)[[翻译]](http://www.gwylab.com/pdf/stylegan2_chs.pdf) CVPR, 2020. Nvidia.
  - [x] [GAN-Control: Explicitly Controllable GANs.](https://arxiv.org/abs/2101.02477) [[PDF]](./paper/GAN-Control.pdf) arXiv, 2021. Amazon.
  - [ ] (InterFaceGAN) [Interpreting the Latent Space of GANs for Semantic Face Editing.](https://arxiv.org/abs/1907.10786) [[PDF]](./paper/InterFaceGAN.pdf) [[Github]](https://github.com/genforce/interfacegan) CVPR, 2020.

  

#### 3DMM

- [ ] 3DMM
- [ ] 3DFFA





#### Pretraining

- [ ] [Robust One Shot Audio to Video Generation](https://ieeexplore.ieee.org/document/9150729)

- [ ] [HeadGAN: Video-and-Audio-Driven Talking Head Synthesis](https://arxiv.org/abs/2012.08261)

- [ ] [Towards Real-World Blind Face Restoration with Generative Facial Prior.](https://arxiv.org/pdf/2101.04061.pdf) [[PDF]](./paper/Towards_Real_World_Blind_Face_Restoration_with_Generative_Facial_Prior.pdf)

  (Blind face restoration, 用到pretrained GAN as prior)



#### Survey

- [ ] [The Creation and Detection of Deepfakes A Survey](https://arxiv.org/abs/2004.11138) [[PDF]]()
- [ ] [Transformers in Vision: A Survey]()
- [ ] [3D Morphable Face Models—Past, Present, and Future](https://dl.acm.org/doi/abs/10.1145/3395208)



#### Extension

- [ ] [VinVL: Making Visual Representations Matter in Vision-Language Models]()

- [x] [Anomaly Detection in Video Sequence with Appearance-Motion Correspondence.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf) [[PDf]](./paper/Anomaly_Detection_in_Video_Sequence_with_Appearance_Motion_Correspondence.pdf) [[Github]](https://github.com/nguyetn89/Anomaly_detection_ICCV2019)

  (其中用到gradient loss，原因是L2 reconstruction loss会使边缘模糊，而image gradient loss可以锐化细节，gradient loss实现代码在GAN_tf.py的224行。用到了optical flow做motion prediction)

  

#### Blogs

- [ ] [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names](https://gombru.github.io/2019/04/03/ranking_loss/)
- [ ] [常用数学符号的LaTeX表示方法](https://www.mohu.org/info/symbols/symbols.htm)
