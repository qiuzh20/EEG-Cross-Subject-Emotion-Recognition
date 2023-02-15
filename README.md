# EEG-Cross-Subject-Emotion-Recognition

实现了基于EEGNet的Cross Subject Emotion Recognition

数据预处理请参考 `preprocess/`

**Directory structure:**
```
./
├── preprocess
├── report
│   ├── pic/         # 相关图片
│   └── report.md    # 实验报告
├── runs
│   └── ....         # 按路径名称记录实验结果
├── basic_demo.ipynb # 展示了基本数据处理过程和模型输入输出
├── dataset.py       # 用于监督学习和对比学习的数据集 
├── main.py          # 主函数
├── model.py         # 实现了EEGNet和EEGNet2D
└── utils.py         # contrastive loss，logger等功能函数
```

基础命令
`
python main.py --model EEG1D --F1 16 --D 16
`

更多请参考 `main.py` 中的 parser。
