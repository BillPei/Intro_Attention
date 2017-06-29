A Decomposable Attention Model for Natural Language Inference
---
- tensorflow实现: python intro_attention.py train
- data里面应该包含train.pair和test.pair
- 2个epoch的测试准确度约为82%
- 跟论文一样，分为Attend-Compare-Aggregate三个阶段
- Attend:
	- Attend可以看作点积化的attention
	- 论文里面似乎没有提，但是我将所有padding部分都设置成了0
- Compare：
	- 这一步用来将原来的词向量跟attention处理之后的词向量联合
	- 这里concat之后的向量理应用G函数转换一下，但是我发现好像直接用训练更快
	- 用G函数转换之后的最终效果尚未测试；
- Aggregate
	- 这里先将句子按照列reduce_sum， 一个句子转换成一个固定维度的向量
	- 这一步原文是feed forward network + linear layer
	- 原文没有说明这个feed forward网络用什么结构或者用什么activation函数
	- 我发现直接用最终向量乘以一个矩阵再做点积效果更好；


simi_cnn是将问答句子做相似度矩阵计算之后做cnn的模型；
---
1：问答句子做词向量表示之后，分别成32*256的矩阵，按照单词做相似度计算，成一个32*32的对称矩阵；
2：词向量后的句子用双向LSTM跑一遍，得到2*32*256矩阵，按1的方法做相似度计算，成两个32*32的矩阵；
3：根据1，2得到32*32*3的矩阵表示问题答案的关系，再做convolution，最后跟结果进行比较；

