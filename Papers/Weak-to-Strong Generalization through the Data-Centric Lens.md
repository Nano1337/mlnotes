
### Definitions
- Patterns: sets of features or mechanisms for prediction per sample
- Weak models: can label sample with easy and hard patterns, but not samples with ONLY hard patterns
- Strong models: can label all types of samples no matter weak or hard
- Overlap density: hard patterns present within samples with both weak and hard patterns
Thesis: since weak models can label data with both weak and hard patterns, the hope is that the stronger model being trained can learn the harder patterns more effectively and generalize to points ONLY containing the hard patterns. 


### Setup
- Data: Have supervised labeled dataset $D_{train}$ of $n$ samples and $D_{w2s}$ unlabeled dataset with $m$ samples. 
- Models: $f_{weak}$ is trained/fine-tuned on $D_{train}$ ; $f_{w2s}$ is trained on $D_{w2s}$ with labels provided by $f_{weak}$. 

Question:
- What do we know about $D_{w2s}$? Seems like it contains all kinds of samples? It doesn't specify the quality of that dataset it only says it has an extra $m$ samples and isn't labeled

### Thoughts
- This theoretical framework might be able to explain why the templated recaptioning works since we're basically transforming the giant unsupervised dataset into a supervised dataset and classification evals are basically a supervised task. 

### TODO
- After writing up the templated recaptioning results and report, throw that report + this paper and ask R1 to reason and draw connections and conclusions

