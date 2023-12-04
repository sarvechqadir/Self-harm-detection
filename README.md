# Self-harm/Suicide-detection

![image](https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/71eec72b-971a-4dcc-8a3f-515b443856f2)

Suicidal intention or ideation detection is one of the evolving research fields in social media. People use Instagram, Twitter, and Reddit platforms to share their thoughts, tendencies, opinions, and feelings toward suicide, especially among youth. Detection of suicide and self-harm therefore becomes a challenging one due to the unstructured and noisy texts and data. Although engaging with people having similar experiences warrants peer-to-peer support, exposure to such content on these platforms can pose a threat of an increase in these behaviors or normalizing the use of this content. Moreover, suicide is a growing public health concern and has been ranked the second leading cause of death among youth (10-24) in the United States. Therefore, it becomes vital to research how youth talk and discuss these sensitive topics on social media. Several studies conducted research identifying different mechanisms for early and onset detection of suicide and self-harm content for mitigation.   

# Problem

1. It is vital to detect online indicators of mental health issues to mitigate harm.
2. There is immense reliance on publicly available digital trace data or self-reported survey data.
3. Survey-based studies are prone to recall biases.
4. There is a dire need to study how self-harm or suicide discussions unravel in private conversations of youth.


# Data Cart

This study was conducted on two different datasets. One was the Suicide and Depression Detection dataset from Kaggle (which is an openly used dataset for public use) focused on Reddit posts. The second suicide/self-harm annotated dataset is part of my study. Since the dataset is not public it cannot be shared for public use. 

## Suicide and Depression Detection

[Kaggle Link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

The dataset is a collection of posts from the "SuicideWatch" and "depression" subreddits of the Reddit platform. The posts were collected using Pushshift API. All posts that were made to "SuicideWatch" from Dec 16, 2008(creation) till Jan 2, 2021, were collected while "depression" posts were collected from Jan 1, 2009, to Jan 2, 2021. All posts collected from SuicideWatch are labeled as suicide, While posts collected from the depression subreddit are labeled as depression. Non-suicide posts are collected from youth data. 

<img width="903" alt="Screen Shot 2023-12-03 at 5 47 51 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/54ab5954-7874-45e2-9e7c-674e51ef5767">

For the dataset, I extracted 22211 rows of conversations containing both suicide and non-suicide-labeled data. 

<img width="461" alt="Screen Shot 2023-12-03 at 6 00 14 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/884e2e3e-40c5-48d3-8b75-dd1b4ebee154">


## Annotated Dataset of Instagram private conversations

This dataset is a suicide/self-harm annotated by researchers and youth which contains sub-conversations/snippets from Instagram private conversations. Since the dataset is not public it cannot be shared for public use. The data contains 3518 entries on suicide and self-harm in private conversations containing both suicide and non-suicide labeled data. 


# Approach


1. **BERT Encoder**: My choice of the model is to use BERT encoder, which in this case is "base" and "uncased". "Base" means it's the smaller version of BERT (as opposed to "large"), with 12 layers (or transformer blocks), 768 hidden units, and 12 self-attention heads. "Uncased" means that it does not differentiate between uppercase and lowercase letters, treating all text as lowercase. The encoder receives tokenized text as input, which is then passed through multiple layers of transformer blocks that process the text bi-directionally, capturing context from both the left and right sides of each token.

![image](https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/f4a216df-555e-483d-9dbf-033934231e9b)


   - **Inputs**: The inputs to the BERT model are typically token embeddings, segment embeddings, and position embeddings. The token embeddings are representations of the tokens in the input text. Segment embeddings are used to distinguish between different sentences or segments of text. Position embeddings provide information about the position of each token in the sequence. In our case, we are working with token embeddings. 
   - **Weights \( W_i \)**: Trainable parameters of the BERT model that are adjusted during the training process. Each layer \( i \) has its own set of weights.
   - **Layer \( L_i \)**: Each layer of the transformer model performs self-attention and other transformations to process the input sequence into a higher-level representation.

```
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```
<img width="722" alt="Screen Shot 2023-12-03 at 7 32 52 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/99d16610-0e28-42fc-9032-529c9fa153bc">



2. **Meta Encoder**: This part of the diagram indicates there is an additional encoding layer that is specific to the task or domain. This "Meta Encoder" takes meta-information (like the coolness, usefulness, funniness, etc., of a product) which may be encoded separately from the main text input. This meta-information is likely represented as a one-hot encoding or some other form of numerical representation that can be understood by the model.

3. **Representation Concatenation**: The outputs of the BERT encoder and the Meta Encoder are concatenated to form a combined representation. This suggests that the model is designed to use both the contextual information from the text and the meta-information to make its predictions.

4. **Classification and Regression Heads**: On top of the concatenated representations, there are different "heads" for different tasks:
   
   - **Classification Head**: This part of the model takes the concatenated representation and outputs probabilities for different classes, such as "sentiment score" or "star rating". This is typically done using a softmax function that converts raw model outputs into probabilities.
   - **Regression Head**: In addition to classification, the model can also do regression, predicting continuous values (like an overall "sentiment score"). This is usually done with a simple linear layer with one output, which predicts the score directly without any activation function like softmax.

Each head uses the same underlying BERT representations but is trained for different tasks. The presence of multiple heads indicates that the model is multi-tasking — it is trained to perform both classification (categorical outputs) and regression (continuous outputs) simultaneously.

In summary, this BERT-based model has been adapted to incorporate additional meta-information alongside the text input to perform both classification and regression tasks. The model leverages the powerful contextual embeddings generated by the BERT encoder, enriched with task-specific metadata, to perform its predictions.

## Why choose this approach?
1. The base model is faster to train and to use for inference, making it more practical for many applications, especially when resources are limited.
2. The choice between BERT-base-uncased and other variants like BERT-base-cased, BERT-large-uncased, or BERT-large-cased depends on the specifics of the task at hand, the computational resources available, and the characteristics of the text data you are working with. Based on the available resources, size, and data especially those where casing is not critical, BERT-base-uncased is a solid default choice due to its balance of size, speed, and performance.
3. Uncased vs. Cased: An "uncased" model does not make a distinction between uppercase and lowercase letters and treats them as the same. This is useful in cases where case information is not important, such as when you are dealing with text where capitalization is used inconsistently, or when you want to reduce the model's vocabulary size and complexity.
5. Performance: For many tasks, the uncased variant of BERT performs comparably to the cased variant and sometimes even better, especially if the case information in the training data is noisy or if the downstream task does not rely heavily on the case.
6. Simplicity: When using the uncased model, you don't need to worry about the correct capitalization of your input text, simplifying preprocessing steps. This can be particularly useful when dealing with user-generated content like social media posts, where capitalization may be irregular.
7. Vocabulary Size: The uncased model has a smaller vocabulary since it doesn't need separate tokens for cased and uncased versions of words, which can slightly reduce the computational requirements.


## Training

```
history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=val_dataset
)
```
```
Epoch 1/2
8884/8884 [==============================] - 2654s 293ms/step - loss: 0.1425 - accuracy: 0.9451 - val_loss: 0.0914 - val_accuracy: 0.9667
Epoch 2/2
8884/8884 [==============================] - 2594s 292ms/step - loss: 0.0582 - accuracy: 0.9800 - val_loss: 0.1474 - val_accuracy: 0.9685

```

## Testing

```
model.evaluate(test_dataset)
```
```
1111/1111 [==============================] - 113s 102ms/step - loss: 0.2261 - accuracy: 0.9577
```

## Sample Input

[**Interactive demonstration - HF Space**](https://huggingface.co/spaces/Sarvech/BERTSelf-Harm)

<img width="726" alt="Screen Shot 2023-12-03 at 8 04 05 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/a0f7353d-f1dd-4c66-a292-2ee19b3991bf">



# Dataset2 - Effectiveness
![Screen Shot 2023-12-04 at 8 51 59 AM](https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/9302cb35-d4f0-457e-86f8-58dc51193405)

This shows that both datasets have given high accuracy on both datasets with >95% accuracy on test data.  



# Critical Analysis
The current solution is not perfect but has high accuracy and works well on many specific types of texts in private conversations.

Persisting issues: The model still makes mistakes because it has not been trained on much data regarding suicide and self-harm language usage of youth. More annotated data by youth can be better for the model to perform in-context learning. 

## Next steps:

1. Work on more ground truth annotations by youth to include more data.
2. Perform with different BERT models as suggested by different research i.e DistilBERT, ALBERT, RoBERTa, and DistilRoBERTa, and check for performance.
3. Research and work on more approaches for in-context learning.
4. Comparative analysis of working different models on different kinds of datasets such as posts, tweets, comments, and private conversations to finalize a perfect model for each specific situation.
5. A BERT "cased" model preserves the original case of the text, which can be beneficial for tasks where case information carries important signals, such as Named Entity Recognition (NER), look at Capitalized texts to understand the context and responses.

# Resources
**Dataset**: [https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)

**Papers**:
1. [Suicidal Intention Detection in Tweets Using BERT-Based Transformers](https://ieeexplore.ieee.org/abstract/document/10037677?casa_token=Z_ouS0P8gPAAAAAA:jsSd5ZnZ6qMD9yboUK228eNUwsfWR78l8ZEAkd-wgCHXTiSArOQB1FL37BJnyu5aJ5VWM7hssQ)
2. [Towards Suicide Ideation Detection Through Online Conversational Context
](https://dl.acm.org/doi/abs/10.1145/3477495.3532068?casa_token=Xlm5arI4a5sAAAAA:5kDhl-4Er6xfK020NKoYNeL06nI2cj5PucseObQ7c1OwM5oi0B4TGYoiujr4rWtjsR1xsjYEiwLuyw)

**BERT-base-uncased**: [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)


