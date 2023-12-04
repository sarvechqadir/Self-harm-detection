# Self-harm-detection

![image](https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/71eec72b-971a-4dcc-8a3f-515b443856f2)

Suicidal intention or ideation detection is one of the evolving research fields in social media. People use Instagram and Twitter platforms to share their thoughts, tendencies, opinions, and feelings toward suicide, especially among youth. Therefore, this task becomes a challenging one due to the unstructured and noisy texts. Although engaging with people having similar experiences warrants peer-to-peer support, exposure to such content on these platforms can pose a threat of an increase in these behaviors or normalizing the use of this content. Moreover, suicide is a growing public health concern and has been ranked the second leading cause of death among youth (10-24) in the United States. Therefore, it becomes vital to research how youth talk and discuss these sensitive topics on social media. Several studies conducted research to   

# Problem

1. It is vital to timely detect online indicators of mental health issues to mitigate harm.
2. There is immense reliance on publicly available digital trace data or self-reported survey data.
3. Survey-based studies are prone to recall biases
4. There is a dire need to study how self-harm or suicide discussions unravel in private conversations of youth.


# Data Cart
## Suicide and Depression Detection

[Kaggle Link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

The dataset is a collection of posts from the "SuicideWatch" and "depression" subreddits of the Reddit platform. The posts are collected using Pushshift API. All posts that were made to "SuicideWatch" from Dec 16, 2008(creation) till Jan 2, 2021, were collected while "depression" posts were collected from Jan 1, 2009, to Jan 2, 2021. All posts collected from SuicideWatch are labeled as suicide, While posts collected from the depression subreddit are labeled as depression. Non-suicide posts are collected from r/teenagers.

<img width="903" alt="Screen Shot 2023-12-03 at 5 47 51 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/54ab5954-7874-45e2-9e7c-674e51ef5767">





2. Annotated Dataset from private conversations. 


# Approach

<img width="461" alt="Screen Shot 2023-12-03 at 6 00 14 PM" src="https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/884e2e3e-40c5-48d3-8b75-dd1b4ebee154">




![image](https://github.com/sarvechqadir/Self-harm-detection/assets/78235308/f4a216df-555e-483d-9dbf-033934231e9b)


1. **BERT Encoder**: The core of the model is the BERT encoder, which in this case is "base" and "uncased". "Base" means it's the smaller version of BERT (as opposed to "large"), with 12 layers (or transformer blocks), 768 hidden units, and 12 self-attention heads. "Uncased" means that it does not differentiate between uppercase and lowercase letters, treating all text as lowercase. The encoder receives tokenized text as input, which is then passed through multiple layers of transformer blocks that process the text bidirectionally, capturing context from both left and right sides of each token.

   - **Inputs**: The inputs to the BERT model are typically token embeddings, segment embeddings, and position embeddings. The token embeddings are representations of the tokens in the input text. Segment embeddings are used to distinguish between different sentences or segments of text. Position embeddings provide information about the position of each token in the sequence.
   - **Weights \( W_i \)**: These are the trainable parameters of the BERT model that are adjusted during the training process. Each layer \( i \) has its own set of weights.
   - **Layer \( L_i \)**: Each layer of the transformer model performs self-attention and other transformations to process the input sequence into a higher-level representation.

2. **Meta Encoder**: This part of the diagram indicates there is an additional encoding layer that is specific to the task or domain. This "Meta Encoder" takes meta-information (like the coolness, usefulness, funniness, etc., of a product) which may be encoded separately from the main text input. This meta-information is likely represented as a one-hot encoding or some other form of numerical representation that can be understood by the model.

3. **Representation Concatenation**: The outputs of the BERT encoder and the Meta Encoder are concatenated to form a combined representation. This suggests that the model is designed to use both the contextual information from the text and the meta-information to make its predictions.

4. **Classification and Regression Heads**: On top of the concatenated representations, there are different "heads" for different tasks:
   
   - **Classification Head**: This part of the model takes the concatenated representation and outputs probabilities for different classes, such as "sentiment score" or "star rating". This is typically done using a softmax function that converts raw model outputs into probabilities.
   - **Regression Head**: In addition to classification, the model can also do regression, predicting continuous values (like an overall "sentiment score"). This is usually done with a simple linear layer with one output, which predicts the score directly without any activation function like softmax.

Each head uses the same underlying BERT representations but is trained for different tasks. The presence of multiple heads indicates that the model is multi-tasking — it is trained to perform both classification (categorical outputs) and regression (continuous outputs) simultaneously.

In summary, this BERT-based model has been adapted to incorporate additional meta-information alongside the text input to perform both classification and regression tasks. The model leverages the powerful contextual embeddings generated by the BERT encoder, enriched with task-specific metadata, to perform its predictions.


