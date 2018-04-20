""" 
                RATIONALE FOR THE PERPLEXITY:
  We can basically adhere to the perplexity as 2^{1/M*\sum^{M}_{i} {\log{p(w_i | w_1, ..., w_{i-1})}}}.
As we compute this perplexity over a known test set, we just need to know how likely is our model to
predict correctly the following word; so basically we are gonna take the probability (from the softmax)
that our model spits out for our [target] word of the test set, that'll be p(w_i | w_1, ..., w_{i-1}). 
On the practical level, at the end of the day is doing the cross entropy, as we'd in our loss, and then
base 2 exponent of the average of those cross entropies to obtain the perplexities. 
"""

import numpy as np

def calculate_cross_entropy(pred, target):
    cross_entropy = -np.log2(pred[target])
    if cross_entropy == 'nan':
        cross_entropy =1e30 
    return cross_entropy

def perplexity(sentence, target_sentence):
    """
    Funtion to compute perplexity.
        sentence: outputs of the softmax layer,
        target_sentence: target sentence with the indices of the words
    """
    exp = 0
    for i in range(len(sentence)):
        exp += calculate_cross_entropy(sentence[i], target_sentence[i])
    exp *= 1/len(sentence)
    return 2**exp
