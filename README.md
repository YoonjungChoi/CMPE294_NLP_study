# CMPE 297 Natural Language Processing

- Project "Question Answering Task"
- Title "Kids Education Assistant"


# Project Report


 ## 1. Overview

 The cutting edge techniques are so prevalent everywhere and contemporary children are fluent in manipulating devices since they have been exposed to devices when they are growing up. Variety educational applications give children a chance to learn by themselves. With the advent of the COVID-19, many children have to stay at home and need contactless education. Most educational applications are based on non-language operations. For example, “OSMO” is one of the kids educational applications being able to interact with users by using image recognition. If applications have the ability to communicate with users, the effect of learning would improve significantly. There are definite demands on applications handling natural language processing. To make educational applications having semantically natural conversations with children, language models should be trained. We proposed building a language model of an educational application for children like virtual teaching assistants. 


Question-Answering, which is a computer science discipline within the fields of information retrieval and natural language processing [wiki](https://en.wikipedia.org/wiki/Question_answering), enables us to have more semantic conversations between devices and humans. By using Question-Answering techniques in Natural Language Processing, children can question things and get answers by virtual teaching assistants and keep expanding their knowledge without losing their interests. 

To aid children to understand knowledge on science topics like Space, we will collect information from [ESA-Space for Kids](https://www.esa.int/kids/en/learn). This site has information about scientific topics. The dataset we will be using should be structured with text, questions and answers. We will create questions and answers from the text. 


## 2. Dataset description and characteristics

Dataset is csv format and answers are extracted from exactly the same words or sentences from text. We will find a ‘start_char_index’ and ‘end_char_indet’ through code (note: not generative or paraphrasing answers). There are a total of 150 samples for the training set with 19 topics. We used 20 samples for a test set. Test set has the same text as the train set but has different question and answer pairs. The below Table 1 shows the number of tokens at each set. 


|              |  Training Set (Text, Question)  |  Test Set(Text, Question)  |
|--------------|---------------------------------|----------------------------|
| Total Tokens |  44607                          |  5570                      |
| Unique Tokens|  1265                           |  699                       |
| Mean Tokens of Questions | 10                  |  11                        |
| Mean Tokens of Text  |  287                    |  268                       |


## 3. NLP models and techniques utilized

**BERT**, which stands for Bidirectional Encoder Representations from Transformers. By concurrently conditioning on both left and right context in all layers, BERT is aimed to pre-train deep bidirectional representations from unlabeled text, in contrast to recent language representation models.

**BERT Tokenizer  [Preprocess](https://huggingface.co/docs/transformers/preprocessing#natural-language-processing)**

Tokenizer is the main preprocessing phase, which splits text into tokens based on rules. The tokens are converted into numbers and then, tensors, which are the expected model inputs. In detail, texts are converted into a sequence of tokens, and creates a numerical representation of the tokens, and assembles them into tensors.  The tokenizer returns three important values: 

- **input_ids** are the indices corresponding to each token in the sentence

- **attention_mask** indicates whether a token should be attended to or not

- **token_type_ids** identifies which sequence a token belongs to when there is more than one sequence. The first sequence, the “context” used for the question, has all its tokens represented by a 0, whereas the second sequence, corresponding to the “question”, has all its tokens represented by a 1.


**Diverse pre-trained BERT models**

_'Bert-large-uncased-whole-word-masking-finetuned-squad’_

We used pre-trained tokenizer with 'bert-large-uncased-whole-word-masking-fine tuned-squad’ dataset SQuAD dataset, which is a reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles. The texts are lowercase and tokenized using WordPiece and a vocabulary size of 30,000. This model was trained with Whole Word Masking, which means that all of the tokens corresponding to a word are masked at once.

_‘Distilbert-base-uncased-distilled-squad’_

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT's performances. This model is fine-tuned using knowledge distillation on SQuAD v1.1. [Distilbert] 

_‘Bert-base-cased-squad2’_

The model distinguishes lowercase and uppercase. Difference between BERT base and BERT large is on the number of encoder layers. BERT base model has 12 encoder layers stacked on top of each other whereas BERT large has 24 layers of encoders stacked on top of each other. [bert-base vs bert-large]

_‘bert-medium-squad2-distilled’_

This model is distilled from its teacher model ‘bert-large-uncased-whole-word-masking-squad2’ and uses haystack’s distillation feature for training. This model has an overall f1-score of 72.76. Some of the hyperparameters are as follows: batch_size = 6, epochs = 2, learning rate = 3e-5, temperature = 5



**ReTraining-FineTune**

There are variables for fine tuning; learning rate, optimizer, batch size.

 **Optimizer**
 
 **Adam** (Adaptive Moment Estimation) combines RMSProp and Momentum and computes adaptive learning rates. We can obtain a group of parameters from ‘param_optimizer = list(model.named_parameters())’ and use these parameters to optimize. The ‘betas’ option is coefficients used for computing running averages of gradient and its square. The ‘lr’ option stands for learning rate and the ‘eps’ option is a term added to the denominator to improve numerical stability. [Pytorch Adam Ref] (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) AdamW implements Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay Regularization.[Hugging Face](https://huggingface.co/transformers/v3.3.1/main_classes/optimizer_schedules.html#transformers.AdamW)
 
**Evaluation**

**Loss Function**

The loss from start and end logits are weighted equally in the loss function.

**Evaluation Method**

This below is extracted from the [Paper : Question and Answering on SQuAD 2.0](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15812785.pdf)

  To evaluate our models we use the standard SQuAD performance metrics: Exact Match (EM) score and F1 score.  For our project, we focus on the EM and F1 scores with respect to the dev set.
  - **Exact Match**: A binary measure of whether the system output matches the ground truth answer exactly.
  - **F1**: Harmonic mean of precision and recall, where precision = (true positives) / (true positives + false positives) and recall = true positives / (false negatives + true positives). F1 score = (2×prediction× recall) / (precision + recall). 
  (Note: In SQuAD, **precision measures to what extent the predicted span is contained in the ground truth span, and recall measures to what extent the ground truth span is contained in the predicted span.**)


**BLEU [Paper: Evaluating Question Answering Evaluation](https://mrqa.github.io/2019/assets/papers/45_Paper.pdf)**

BLEU stands for BiLingual Evaluation Understudy. A BLEU score is a quality metric assigned to a text which has been translated by a Machine Translation engine. BLEU scores a candidate by computing the number of n-grams in the candidate that also appear in a reference. 

**BERTScore**

[BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore) is an automatic evaluation metric for text generation that computes a similarity score for each token in the candidate sentence with each token in the reference sentence. It leverages the pre-trained contextual embeddings from BERT models and matches words in candidate and reference sentences by cosine similarity.

**Additional Interface - Pipelines**

The pipelines offer an excellent and simple method for using models for inference. Pipelines are classes that encapsulate the majority of the library's sophisticated logic. Pipelines are made of a tokenizer mapping raw textual input to the token and a model to make predictions from the inputs and some post-processing for enhancing the model’s output.

**GPT-3**

GPT stands for Generative pre-trained Transformer, which is an autoregressive language model that uses deep learning to produce human-like text. Given an initial text as a prompt, it will produce text that continues the prompt. [Wiki](https://en.wikipedia.org/wiki/GPT-3). Researchers at OpenAI described the development of GPT-3 as third-generation "state-of-the-art language model". Instead of fine-tuning the model, we decided to experiment prompt engineering, which is another approach to improve pre-trained models. Prompt Engineering adds informative context into the prompt, helping GPT-3 better understand the scenario and background of the questions. We compared the result using prompt engineering to that without prompt engineering, and we uses two approaches to evaluate the result. The first one is BLEU score, the second one is GPT-3 similarity embedding with cosine similarity.

## 4. Experiments conducted

4.1 Experiment on BertQuestionAnswering Models for fine-tuning

**Training workflow:**
  - Set GPU environment.
  - Prepare data input and output.
  - Load model
  - Forward pass by feeding input data through the model with model.train()
  - Backward pass to update parameters with optimizer.step()
  - Log variables(steps, loss) for monitoring progress
  - Save fine-tuned model

**Evaluation workflow:**
  - Prepare data input and output
  - Load fine-tuned model
  - Forward pass by feeding input data through the model with model.eval()
  - Log results
  - Save and compare result

_'bert-large-uncased-whole-word-masking-finetuned-squad'_

This Bert Question Answering model with uncased squad set shows the decrease in training loss of the below plots. Based on initial experiments, We modified a few values for fine-tuning like batch size, learning rate, number of epochs. We used batch size 5 because we faced the issue “CUDA out of memory”,  used learning late 1e-5 is a very small number to prevent overshooting, and used 5 epochs.

Adam: y = [2.3990, 1.1283, 0.6045, 0.3451, 0.1929] | AdamW: y = [2.4794, 1.1356, 0.5683, 0.3663, 0.1738]

_‘distilbert-base-uncased-distilled-squad’_

When fine tuning on this model, training is very fast and finished quickly rather than the Bert Large model. However, this model cannot respond to some questions.

## 5. Insights

### Insights from BertQuestionAnswering experiment

From the initial experiment, We realized that QuestionAnsweringModel is a factoid model to extract start and end span from the texts and needed to use a tokenizer and model from the same dataset. We fixed some issues on the dataset where there could be typo or whitespace in question and answers. The training loss with 2 epochs quickly reduced from 2.5549 and to 0.9266. When we take into account that we did an experiment within only 2 epochs, which means there is a possibility of overshooting in case of feeding more data. Thus, we should take a look into the loss to be reduced stably. Also, we found that prediction is short words rather than expressing one sentence. From experiments with changing parameters on model, We learned how to train with optimizer, batch size, epoch and as well as converting raw data into question answering formats. For evaluation, we should declare model.eval() not to update via backpropagation. We compared two different Bert models before and after fine tuning, and evaluated them using BLEU and BERTScore’s F1 Average. The result shows the table below. The DistillBert model was really fast trained rather than the Bert Large model, but it cannot answer some questions. 

| Model                                                  | BLEU   | BERTScore (F1 Average) |
|--------------------------------------------------------|--------|------------------------|
| bert-large-uncased-whole-word-masking-finetuned-squad  | 20.2   |  0.89                  |
| distilbert-base-uncased-distilled-squad                | 21     |  0.80                  |

