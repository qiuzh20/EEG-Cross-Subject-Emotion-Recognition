# EEG-Cross-Subject-Emotion-Recognition

预处理部分：
- [ ] 去处问题数据，保留 （80，28 ，30 * 125） 的训练数据和 (80, 28) 的 label
	- [ ] 对应的功率谱密度
	- [ ] (optional) 逐个体 通道标准化
- [ ] 剔除眼动信号等无关信号，保留同上信号
	- [ ] 对应的功率谱密度

后续实验：
- [ ] 网络框架：原始EEGNet 与 STNet（22年文章中框架）
- [ ] 学习方法：加/不加 contrastive learning
- [ ] 输入信息：
	- [ ] 剔除/不剔除 无关信号 
	- [ ] 加/不加 功率谱密度信息
- [ ] 增强方法（可以分别在contrastive training 和 标准训练中加入）：
	- [ ] 增加 noise
	- [ ] 改变 window 范围
