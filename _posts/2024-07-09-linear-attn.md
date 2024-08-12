---
layout: distill
title: Linear Attention Fundamentals
description: The basics of linear attention in sub-quadratic language model architectures. 
tags: ML, Architectures, Linear-Attention
giscus_comments: false
date: 2024-07-09
featured: false

authors:
  - name: Hailey Schoelkopf
    url: "https://haileyschoelkopf.github.io"
    affiliations:
      # name: IAS, Princeton

bibliography: 2024-07-09-linear-attn.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Linear(ized) Attention
  - name: Recurrent Form
  - name: Parallelization
    subsections:
    - name: Chunkwise Form
  - name: Conclusion 

---

## Introduction

This post will be a short overview recapping key formulas and intuitions around the increasingly-popular family of methods under the umbrella of Linear Attention, first introduced by Katharopoulos et al. (2020) <d-cite key="katharopoulos2020transformersrnnsfastautoregressive"></d-cite>. The material in this post is also covered excellently by Yang, Wang et al. (2023)<d-cite key="yang2024gatedlinearattentiontransformers"></d-cite> and Yang et al. (2024).

*This post will assume familiarity with the transformer architecture and softmax attention, and with KV caching for decoding.*

---

The fundamental building block of the transformer is the scaled dot product / softmax self-attention, defined as the following:<d-footnote>We elide the scaling by $\sqrt d$ and the causal mask for simplicity, for now.</d-footnote>

 $$ \text{Attention}(Q,K,V) = \text{Softmax}\left(QK^T\right)V $$ 

Softmax attention is very powerful, but is not especially friendly to fast inference, especially when operating on very long sequences: it requires  $$ O(L^2d) $$  FLOP<d-footnote>We write "FLOP" to indicate a quantity of floating point operations and "FLOPs" to indicate "FLOP per second". </d-footnote> and (although circumvented by efficient, modern implementations such as Flash Attention<d-cite key="dao2022flashattentionfastmemoryefficientexact"></d-cite><d-cite key="dao2023flashattention2fasterattentionbetter"></d-cite>) naively  $$ O(L^2) $$  memory. 


For this reason, many attempts have been made to recover softmax attention performance at  $$ < O(L^2) $$  complexity--for example, an operation that scales linearly in complexity as  $$ O(L) $$  as sequence length increases. While many of the methods have commonalities or shared foundations, we'll focus on one particular family of methods that fall under the header of "Linear Attentions". 


## Linear(ized) Attention


Katharopoulos et al. (2020) introduce the foundation of linear attention. We assume as input  $$ Q, K, V \in \mathbb{R}^{L \times d} $$ . The core idea is that, as we can write scaled dot product / softmax attention as

 $$ \text{Softmax}\left(QK^T\right)V = \frac{ \text{exp}(QK^T) } {\sum_{i=1}^L \text{exp}(QK_i^T)} V $$ 

, we can view this as a special case of

 $$ 
\frac{ \text{sim}(Q, K) } {\sum_{i=1}^L \text{sim}(Q, K_i)} V
 $$ 

for any (non-negative) similarity function  $$ \text{sim}(\cdot, \cdot) $$  between keys and queries,  $$ \text{sim}(Q, K) $$ . 

Katharopoulos et al. note that this applies to any *kernel*  $$ \text{sim}: \mathbb{R}^d \times \mathbb{R}^{d'} \rightarrow \mathbb{R}_{+} $$  (applied to each position in our sequence independently), where 

 $$ \text{sim}(Q, K) = \phi(Q) \cdot \phi(K) = \phi(Q)\phi(K)^T $$ 

--that is, the similarity metric is induced by applying a "feature map"  $$ \phi $$  independently to both Q and K.  $$ \phi $$  can optionally change dimensionality of our queries and keys from $d$ to $d'$, though we will often see  $$ d' = d $$ .  $$ \text{exp}() $$  cannot be expressed exactly using any particular  $$ \phi $$ <d-footnote>Although we can approximate it with polynomial-sized feature maps in the limit</d-footnote>, but a number of other useful functions including simply the identity can.


Now that we've switched from  $$ \text{exp}(QK^T) $$  to  $$ \text{sim}(Q, K) = \phi(Q)\phi(K)^T $$ , we can leverage the fact that **matrix multiplication is associative** to rearrange our order of operations:


 $$ \frac{ \phi(Q)\phi(K)^T } {\sum_{i=1}^L \phi(Q)\phi(K_i)^T} V = \frac{ \phi(Q)(\phi(K)^T V)} {\phi(Q)\sum_{i=1}^L \phi(K_i)^T} $$ 

Now, instead of calculating  $$ \phi(Q)\phi(K)^T $$  (size  $$ L \times L $$ , and thus  $$ O(L^2) $$  FLOP and memory)--we can first calculate  $$ \phi(K)^TV \in \mathbb{R}^{d' \times d} $$  , and then calculate the product of  $$ \phi(Q) \in \mathbb{R}^{L \times d'} $$  with  $$ \phi(K)^TV $$ . 

This gives us a way to compute this kernel-based "linear attention" in  $$ O(Ldd') $$  (typically  $$ O(Ld^2) $$  in particular) FLOP and space--we are now *linear* in complexity with respect to sequence length!

## Recurrent Form


The above equations are perhaps easier to understand when written position-wise in terms of our outputs. Let  $$ O \in \mathbb{R}^{L \times d} $$  be our attention mechanism output, and let  $$ o_l $$  be the output at position  $$ l $$ . Another caveat we have not yet dealt with is that GPT-style training requires a lower-triangular mask matrix  $$ M \in \mathbb{R}^{L \times L} $$  to prevent attending to tokens that are in the "future" from our current one. 


We can then write our linear attention output for sequence position  $$ l $$ ,  $$ o_l $$ , as 

 $$ o_l = \frac{\phi(q_l)\sum_{i=1}^l \phi(k_i)^Tv_i}{\phi(q_l) \sum_{i=1}^l \phi(k_i)^T} $$ 

The equivalent for standard softmax attention is given by

 $$ o_l = \frac{\sum_{i=1}^l \text{exp}(q_lk_i^T)v_i}{\sum_{i=1}^l\text{exp}(q_lk_i^T)} $$ 

We can observe 2 characteristics here: 
1. At every timestep, softmax attention requires computing  $$ \text{exp}(q_lk_i^T) $$  between our current query  $$ q_l $$  and every prior key  $$ k_i $$ . **A single decoding step at timestep  $$ l $$  with softmax attention is  $$ O(l) $$  complexity, because the transformer's KV cache of past  $$ k_i,v_i $$  for  $$ i \in [1, l] $$  (its "state") grows linearly with sequence length--as do the FLOP required! 
2. For linear attention, we must only compute a denominator  $$ \sum_{i=1}^l \phi(k_i)^T $$  and  $$ \sum_{i=1}^l \phi(k_i)^Tv_i $$  **which are independent of the current query  $$ q_l $$ **. This means we can *reuse* the previously computed values from up to step  $$ l-1 $$  !

Specifically, if we let  $$ Z_{l-1} = \sum_{i=1}^{l-1} \phi(k_i)^T $$  , and  $$ S_{l-1} = \sum_{i=1}^{l-1} \phi(k_i)^Tv_i $$ , then 

 $$ o_l = \frac{\phi(q_l) (S_{l-1} + \phi(k_l)^Tv_l)}{\phi(q_l)(Z_{l-1} + \phi(k_l)^T)} $$ 

Then, for calculating  $$ o_{l+1} $$ , we can let  $$ S_l = (S_{l-1} + \phi(k_l)^Tv_l) $$  and  $$ Z_l = (Z_{l-1} + \phi(k_l)^T) $$ , and repeat this same calculation!

This gives us a *recurrent view* of linear attention--if we maintain a constant-size *"state"*  $$ S_l $$  and "normalizer"  $$ Z_l $$ , we can perform each decoding timestep in  $$ O(1) $$  time and memory! Katharopoulos et al. (2020) also show that this can be formally viewed as an RNN with matrix-valued state  $$ S_l \in \mathbb{R}^{d' \times d} $$ .

Some work <d-cite key="yang2024gatedlinearattentiontransformers"></d-cite><d-cite key="sun2023retentivenetworksuccessortransformer"></d-cite> drops the  $$ Z_l $$  normalizer due to numerical instabilities, and empirically doesn't observe any problems. Additionally, using  $$ \phi $$  equal to the identity also appears to work, interestingly <d-cite key="sun2023retentivenetworksuccessortransformer"></d-cite>.

This gives us a clean recurrent form for computing  $$ o_l $$  from  $$ o_{l-1} $$ :

 $$ S_l = S_{l-1} + k_l^Tv_l $$ 

 $$ o_l = q_l S_l $$ 

Assuming Linear Attention is both 1) actually in practice faster to compute on hardware than softmax attention and 2) equally or nearly as performant downstream as the vanilla Softmax-attention transformer, this is very promising!

## Parallelization

The above *recurrent form* we've described is efficient at inference time, because we only have to make one update step and get to do this in fewer FLOP and lower memory overhead than in typical attention. 


However, when we want to actually train these models, we run into problems: computing  $$ S_l = S_{l-1} + k_l^Tv_l $$  then  $$ o_l = q_l S_l $$  via looping over our entire sequence length ( $$ l \in [1, L] $$ ) (full "recurrent mode") is generally prohibitively slow, despite costing  $$ O(Ldd') $$  FLOP. We are forced to compute each of our  $$ L $$  timesteps sequentially (instead of in parallel as in softmax attention), and must save our potentially-large state  $$ S_l $$  to memory and read it back from memory at each timestep.<d-footnote>Approaches like the custom kernel for performing a parallel scan in Mamba-1<d-cite key="gu2024mambalineartimesequencemodeling"></d-cite> can mitigate this by keeping the state in higher-bandwidth SRAM, but this imposes limitations on how large we can make our state without being forced to incur IO costs.</d-footnote>


**Does our linear attention permit a parallelizable form for training?** Recall that in our original derivation, we wrote

 $$ O = \frac{ \phi(Q)(\phi(K)^T V)} {\phi(Q)\sum_{i=1}^L \phi(K_i)^T} $$ 

to indicate the computation of our entire output in parallel. However, when performing attention with a *causal mask* as done in GPT-style autoregressive language modeling, we are forced to compute

 $$ O = \frac{(\phi(Q)\phi(K)^T \odot M) V}{\phi(Q)\sum^L_{i=1}\phi(K_i)^T} $$ 

to obtain the right answer to avoid "seeing into the future". This pointwise multiplication by our causal mask  $$ M \in \mathbb{R}^{L \times L} $$  prevents us from using associativity to compute  $$ \phi(K)^TV $$  first--losing the better complexity of linear attention we claimed. We need to compute  $$ \phi(Q)\phi(K)^T $$ , requiring  $$ O(L^2) $$  time. 


### Chunkwise Form

Luckily, not all is lost. Hua et al. (2022)<d-cite key="hua2022transformerqualitylineartime"></d-cite> propose a solution they term the "chunkwise parallel form" for linear attention training, which is later extended by others<d-cite key="yang2024fla"></d-cite><d-cite key="sun2023retentivenetworksuccessortransformer"></d-cite> for even better efficiency.

In particular, we can find a middle ground for efficient training of these models by striking the right balance between the  $$ O(L) $$  recurrent and  $$ O(L^2) $$  parallel forms. We can do this by performing computation in *chunks* across the sequence length, where we will use the parallel form for computing results within a chunk, and the recurrent form for transmitting information across chunks.

We will split our sequence of length  $$ L $$  into  $$ C $$  chunks of length  $$ L // C $$ . Following [RetNet<d-cite key="sun2023retentivenetworksuccessortransformer"></d-cite>, Gated Linear Attention (GLA)<d-cite key="yang2024gatedlinearattentiontransformers"></d-cite>, and DeltaNet<d-cite key="yang2024parallelizinglineartransformersdelta"></d-cite>] we will adopt the notation that  $$ \cdot_{[c]} $$  denotes the given variable's value for the  $$ c $$ -th chunk.

Now, we need to define a few components: first, our new update rule for going from the state  $$ S_{[c]} $$  at the start of chunk  $$ c $$  to the next chunk  $$ c+1 $$ 's starting state  $$ S_{[c+1]} $$  is as follows:<d-footnote>As is convention in the rest of this post, we assume that chunk indices are 1-indexed: $c \in [1, L//C]$. So $S_{[1]}$ corresponds to tokens indexed between $[1, C]$.</d-footnote>

 $$ S_{[c+1]} = S_{[c]} + \sum_{i=((c-1)C + 1)}^{cC}\phi(k_i)^Tv_i = S_{[c]} + \phi(K_{[c]})^T V_{[c]} $$  

And to compute the output for chunk  $$ c+1 $$ ,  $$ O_{[c+1]} \in \mathbb{R}^{C \times d} $$  , we compute

 $$ 
O_{[c+1]} = \phi(Q)_{[c+1]}S_{[c]} + (\phi(Q)_{[c+1]}\phi(K_{[c+1]})^T \odot M)V_{[c+1]}
 $$ 

the first portion of the equation is the contribution across previous chunks, computed using our recurrent mode ( $$ O(Cdd') $$ ), while the latter term is the current chunk's contribution to its output computed using the parallel mode ( $$ O(C^2d') $$ ).

To compute the entire output  $$ O $$  for all chunks, we have two options<d-cite key="yang2024gatedlinearattentiontransformers"></d-cite><d-cite key="yang2024fla"></d-cite>:

1) Precompute and *materialize* each chunk's starting state: save  $$ S_{[c]} $$ ,   $$ \forall c \in [1, L//C] $$ . This can be done by starting with $$ S_{[0]} = O \in \mathbb{R}^{d' \times d} $$, then sequentially calculating and storing $$ S_{[c+1]} = S_{[c]} + \phi(K_{[c]})^T V_{[c]} $$. 

2) Save no intermediate  $$ S_{[c]} $$  aside from (optionally, during the prefill stage for inference)  $$ S_{L//C} $$ , our final state.

If we precompute and materialize our  $$ L//C $$  per-chunk starting states, then we can **calculate all  $$ O_{[c]} $$  simultaneously**, since each  $$ O_{[c+1]} $$  depends only on  $$ S_{[c]} $$ . During training, we can also maintain these per-chunk states in order to more quickly perform the backward pass. However, we do pay a memory and IO cost: we must store  $$ C $$  chunks of size  $$ d' \times d $$ , resulting in  $$ O(Cdd') $$  memory overhead.


Alternately, we can avoid materializing any states  $$ S_{[c]} $$ . This will force us to compute each  $$ O_{[c+1]} $$  sequentially:  for  $$ c \in [1, L//C] $$  , once  $$ S_{[c-1]} $$  has been computed, we can calculate  $$ O_{[c]} $$ , and subsequently update  $$ S_{[c-1]} $$  to  $$ S_{[c]} $$  using our chunkwise update rule and subsequently compute  $$ O_{[c+1]} $$ , and so on, until we have computed our full output  $$ O $$  across all chunks. We pay no memory overhead due to not storing any intermediate states, but in the backward pass we will have to recompute these per-chunk states, requiring extra FLOP.


This chunkwise formulation allows us to interpolate between the parallel and recurrent forms, choosing  $$ C $$  based on which is fastest, and ends up being faster than full recurrence because we can take advantage of fast matrix multiplications without paying a cost quadratic in  $$ L $$  !

This chunkwise formulation is also adopted by the Mamba-2 / SSD<d-cite key="dao2024transformersssmsgeneralizedmodels"></d-cite> architecture--chunkwise parallelism is very hardware-friendly. This chunked algorithm is sometimes called "Flash Linear Attention" for this reason <d-cite key="yang2024gatedlinearattentiontransformers"></d-cite><d-cite key="yang2024fla"></d-cite>.

## Conclusion

In this post, we've seen:

- How "linear attention" is derived and originally motivated
- How this can be viewed as an RNN with a matrix-valued state
- How to make training these theoretically-linear-complexity models efficient on hardware

Again, this post is based off of the excellent exposition in Gated Linear Attention<d-cite key="yang2024gatedlinearattentiontransformers"></d-cite>, Parallel DeltaNet<d-cite key="yang2024parallelizinglineartransformersdelta"></d-cite>, and the original Linear Attention paper. If you're interested in this topic, I'd highly recommend checking them out as a reference point!


## Acknowledgements

Thanks to Arun Kumar for reading an early version of this blog post!
