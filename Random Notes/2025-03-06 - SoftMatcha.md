
Website: https://softmatcha.github.io/
Super fast `grep` but using fast text word embedding so it scales really well 

I was thinking, maybe we can use this as a target domain text mining? Maybe we can concat all the {original caption, synthetic caption, templated caption} per row and use this to filter for all directly relevant classes post-curation?
- Or we could do this earlier in the curation pipeline to prevent extensive computational cost downstream to more expensive filtering algorithms but we would have to guarantee that the text quality is relatively high and descriptive of the image or else we risk losing high quality data. 

This would actually be especially great for reasoning-based RAG systems (as Josh McGrath said reasoning model + grep is better than any RAG system)