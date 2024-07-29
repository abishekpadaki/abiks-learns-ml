Link: https://youtu.be/eMlx5fFNoYc?si=AWOMa2ZsYAvnRs4j

Words/Tokens are first converted to embeddings and then place vectors in a latent space
Transformers can change or adjust the direction of vectors in latent space

'Mole' has 3 diff meanings but have the same embedding
Attention is used to provide direction of this vector based on the context in which the word mole is used

Attention blocks refine meaning of a word in a sentence

## Computation done bts in an attention block
- words are embedded (with their position too) enough to tell you what the word is and where it is in the context (position)
- Goal is to refine the embeddings such that the nouns ingest their meanings based on their corresponding adjectives
- The nouns ask the question if there are any adjectives in front of me, and this question is also embedded as a vector called the 'query vector'. (Much smaller than a regular embedding vector ~128 dimensions). The query vector is computed by taking a matrice (query matrix) Wq which has tunable parameters and matmuling it with the word embeddings.
- We then have another set of vectors called 'Keys' that we matmul with the word embeddings. This can be thought of as keys to potentially answer the queries. The key vector is computed by taking a matrice (key matrix) Wk which has tunable parameters and matmuling it with the word embeddings.
- Both the query vector and key vectors are embedded into the small query/key space (~128 dimensional)
- The key and queries must ideally almost align with each other in the latent space.

Example: "a fluffy blue creature"

Fluffy and Blue align well with Creature (Large dot product). Low dot product means they don't really matter to the noun.
Here Query corresponds to  Creature (noun)
Fluffy and blue correspond to Keys (adjectives)

In ML terms: Fluffy and Blue "Attend to" Creature

- Then, along each column in the table of dot products between queries and keys, we add softmax to normalize the values (bw 1 and 0)
- ![[Screenshot 2024-07-11 at 10.22.52 AM.png]]This grid is called "Attention pattern"

Another technique used during training sometimes is "masking"
where future words in the sentence are hidden (values in grid set to -Inf which becomes 0 after softmax) such that they don't influence previous words and can also act as a test data.

The original transformer paper has the following attention equation:
![[Screenshot 2024-07-11 at 10.32.07 AM.png]]
Here QK is the are the full array of vectors that you get by multiplying the embeddings by the key and query matrices. QK^T represents the grid (all possible values of dot prod bw pairs of keys and queries)

For numeric stability, all the dot product values are divided by the square root of the dimension in that key-query space (~128) (sqrt(dk))

Then the softmax is applied to each column, as shown in the expression

## What is the V?

Single Head Attention: Refining the embeddings
Here, we take a 3rd Matrix called the Value Matrix (Wv) which we use to get a Value vector by multiplying Wv with the first word (adjective/key) and is then added to the word that needs to be refined (query/noun).
This is similar to saying in simple terms, "if something is to be added to a word in order to refine it or change it's meaning, what exactly should that 'something' that's added to the embedding?"

Adding this value (V * dotproduct of QK) we get deltaE
We add this deltaE to the original embedding (delE + E) in hope to add some context/meaning to the word/embedding

Note: The value vector is embedded in the larger latent space, the same as the embeddings of the words.

![[Screenshot 2024-07-11 at 11.08.37 AM.png]]We add this weighted sum across all columns in the grid causing a sequence of changes to all embeddings, thus providing an output of refined embeddings from the Attention block.

This whole process is far is called "One head of attention".

---

Number of parameters added with Q,K and V matrices

Ideally, # Value Params = (# query params + # key params)

(Key + Query + Value params) almost 6.3Mil params

The next token prediction here in this transformer uses "Self-attention"
Another variation is "Cross-attention" which is usually used to map translations bw 2 languages or 2 different formats (audio and transcriptions) where Key and Queries are from different data sets.


GPT-3 uses around 96 attention blocks (multi-headed attention). Which simply means you have 96 different Key and Query matrices and each head has its own distinct value matrices and hence 96 different Value sequences.
The final refined embedding is computed by summing all deltaE embeddings from each head.

So in simple terms - by having many/multiple heads working in parallel, you are understanding different ways in which each word affects the context or meaning of a word in a sentence.

In total, in GPT-3, there are around 58Billion parameters just for Key, Query, Value and Output parameters.

---

## Up next- Multilayer Perceptrons

"Attention is only 1/3 of the bigger picture"
OR
"Attention is only 1/3 of what you need" :)

---

## Some more complex things I haven't noted down:

 - Value down and up matrices ('Low rank' transformation) to ensure # Value Params = (# query params + # key params) which is efficient for multi-head attention.


![[Screenshot 2024-07-11 at 11.31.49 AM.png]]