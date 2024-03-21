# Brief Description of PEFT

## Adapter-based

### Parameter-Efficient Transfer Learning for NLP (13 June 2019)

#### Abstract

Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. **As an alternative, we propose transfer with adapter modules.** Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter’s effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.

#### Notes

![](adapter.png)

- A skip-connection is employed inside the adapter network such that if the parameters of the projection layers are near zeros, the adapter module approximates an identity function.
- During adapter tuning, only the parameters of the adapters, the normalization layers, and the final classification layer are updated.
- During training, the adapters may then be activated to change the distribution of activations throughout the network.

#### Computational Cost

- classification with BERT: All runs are trained on 4 Google Cloud TPUs with a batch size of 32.
- The following command provides an example of tuning with adapters on GLUE. Fine-tuning may be run on a GPU with at least 12GB of RAM, or a Cloud TPU
- For each task, we run AutoML for one week on CPUs, using 30 machines. In this time the algorithm explores over 10k models on average per task.

## Prompt-based

**离散的 prompt --> 连续的 prompt，以降低离散的 prompt 造成的不稳定性**

### GPT Understands, Too (18 Mar 2021)

> https://kexue.fm/archives/8295

#### Abstract

Prompting a pretrained language model with natural language patterns has been proved effective for natural language understanding (NLU). However, our preliminary study reveals that manual discrete prompts often lead to unstable performance, e.g., changing a single word in the prompt might result in substantial performance drop. We propose a novel method P-Tuning that employs trainable continuous prompt embeddings in concatenation with discrete prompts. Empirically, P-Tuning not only stabilizes training by minimizing the gap between various discrete prompts, but also improves performance by a sizeable margin on a wide range of NLU tasks including LAMA and SuperGLUE. P-Tuning is generally effective for both frozen and tuned language models, under both the fully-supervised and few-shot settings.

#### Notes

![](p-tuning-v1.png)

前向过程：
1. 输入一个句子，以及预先设计的一个离散的模板：`The Disney film is good! It was [MASK].`
2. 先使用 BERT 的分词工具分词，并获得 input ids、position ids、attention masks 等；
3. 对输入的 template 中，挑选一个（或多个）token 作为 pseudo token：`[pseudo] Disney film [pseudo] good! [pseudo] was [MASK].` 其初始化可以直接使用原本的 token embedding；
4. 对所有的 pseudo token 喂入一层 LSTM，并获得每个 pseudo token 输出的隐状态向量 $h_i$
5. 将整个句子喂入 BERT embedding layer，对于 pseudo token 部分的 token embedding，则使用 $h_i$ 进行替换，最后喂入 MLM 中获得 [MASK] 位置的预测结果。

- Specifically, given a discrete prompt as the input, P-Tuning concatenates continuous prompt embeddings with the discrete prompt tokens and feeds them as the input to the language model. To further improve performance, we employ a prompt encoder using LSTMs or MLPs to model the dependency between continuous prompt embeddings.
- We can also concatenate discrete prompts with continuous prompts, which performs better and is adopted throughout our experimentWs.
- Prompt Encoder: long short-term memory (LSTM) networks, multi-layer perceptrons (MLPs).

![](kexue_fm.png)

### Prefix-Tuning: Optimizing Continuous Prompts for Generation (1 Jan 2021)



### The Power of Scale for Parameter-Efficient Prompt Tuning (18 Apr 2021)



### Visual Prompt Tuning (23 Mar 2022)

#### Abstract

The current modus operandi in adapting pre-trained models involves updating all the backbone parameters, i.e., full fine-tuning. This paper introduces Visual Prompt Tuning (VPT) as an efficient and effective alternative to full fine-tuning for large-scale Transformer models in vision. Taking inspiration from recent advances in efficiently tuning large language models, VPT introduces only a small amount (less than 1% of model parameters) of trainable parameters in the input space while keeping the model backbone frozen. Via extensive experiments on a wide variety of downstream recognition tasks, we show that VPT achieves significant performance gains compared to other parameter efficient tuning protocols. Most importantly, VPT even outperforms full fine-tuning in many cases across model capacities and training data scales, while reducing per-task storage cost.

#### Figures

![](vpt-general.png)

![](vpt-training.png)

#### Computational Cost

- We use PyTorch to implement all experiments on NVIDIA A100-40GB GPUs.
![](vpt-computational-cost.png)

## LoRA
