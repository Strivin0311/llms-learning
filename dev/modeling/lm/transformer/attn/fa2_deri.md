# Flash Attention 2 Math Derivation

## Forward

### For standard attn forward:

$$
\begin{cases}
\begin{align} 
&P = \mathrm{mask}(QK^{\mathrm{T}} + bias)  \in \mathbb{R}^{N\times N} \\
&A = \mathrm{softmax}_{row\text{-}wise}(P) = \mathrm{diag}(l)^{-1}S  \in \mathbb{R}^{N\times N}, \quad \text{where}\quad l = \mathrm{rowsum}(S) \in \mathbb{R}^{N}, \space S = \exp{(P  - \mathrm{rowmax}(P))} \in \mathbb{R}^{N\times N} \\ 
&O = AV \in \mathbb{R}^{N\times d}
\end{align}
\end{cases}
$$

given $Q,K,V \in \mathbb{R}^{N\times d}$

### For flash-attn forward:

#### step0. the basic attention row-decomposition:

$$
\begin{cases}
\begin{aligned}
&P = \left[ P_1\quad P_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\quad \text{where}\quad P_i = \mathrm{mask}(QK_i^{\mathrm{T}} + \text{bias}) \in \mathbb{R}^{B_q\times B_k},\ Q \in \mathbb{R}^{B_q\times d},\ K_i \in \mathbb{R}^{B_k\times d},\ i \in \{1,2\}, \\
&m = \max\left( \mathrm{rowmax}(P_1), \mathrm{rowmax}(P_2) \right) \in \mathbb{R}^{B_q}, \\
&S = \left[ S_1\quad S_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\quad \text{where}\quad S_i = \exp(P_i - m) \in \mathbb{R}^{B_q\times B_k},\ i \in \{1,2\}, \\
&l = \mathrm{rowsum}(S_1) + \mathrm{rowsum}(S_2) \in \mathbb{R}^{B_q}, \\
&A = \left[ A_1\quad A_2 \right] = \mathrm{diag}(l)^{-1} \left[ S_1\quad S_2 \right] \in \mathbb{R}^{B_q\times 2B_k}, \\
&O = \left[ A_1\quad A_2 \right] \left[
\begin{matrix}
V_1 \\ 
V_2 
\end{matrix}
\right] = \mathrm{diag}(l)^{-1} \left( S_1V_1 + S_2V_2 \right) \in \mathbb{R}^{B_q\times d}.
\end{aligned}
\end{cases}
$$


#### step1. the online-softmax attention:


$$
\text{base}: 
\begin{cases}
\begin{align} 
&m_1 = \mathrm{rowmax}(P_1) \in \mathbb{R}^{B_q},\quad S_1 = \exp(P_1 - m_1) \in \mathbb{R}^{B_q\times B_k}\notag\\
&l_1 = \mathrm{rowsum}(S_1)\in \mathbb{R}^{B_q},\quad A_1 = \mathrm{diag}(l_1)^{-1}S_1\in \mathbb{R}^{B_q\times B_k}  \notag\\
&O_1 = A_1V_1\in \mathbb{R}^{B_q\times d} \notag
\end{align}\\
\end{cases}
$$

$$
\text{update}: 
\begin{cases}
\begin{align} 
&m_2 = \max(m_1, \mathrm{rowmax}(P_2)) \in \mathbb{R}^{B_q},\quad S_2 = \exp(P_2 - m_2) \in \mathbb{R}^{B_q\times B_k}\notag\\
&l_2 = \delta_m l_1 + \mathrm{rowsum}(S_2)\in \mathbb{R}^{B_q},\quad A_2 = \mathrm{diag}(l_2)^{-1}S_2\in \mathbb{R}^{B_q\times B_k}  \notag\\
&O_2 = \mathrm{diag}(l_1/l_2)^{-1}\delta_m O_1 + A_2V_2 \in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
$$
$$
\begin{align}
&\text{where}\quad \delta_m := \exp(m_1 -m_2)
\end{align}
$$


#### step2: flash-attn forward algorithm with tiling (double-loop):
* the outer loop runs through $i := 1 \rightarrow N_q$ for each block of $Q_i$ to compute $O_i$,  where $N_q = \lceil\frac{N}{B_q}\rceil$

$$
\text{in one i-th outer iteration}: 
\begin{cases}
\begin{align} 
&\text{load}\space  Q_i \in \mathbb{R}^{B_q\times d}\space  \text{from HBM to SRAM}\notag\\
&\text{initialize}\space \tilde{O_{i}}^{(0)} = 0_{ B_q\times d },\space  l_i^{(0)} = 0_{B_q} \in \mathbb{R}^{B_q},\space  m_i^{(0)} = -\infty_{B_q} \in \mathbb{R}^{B_q}  \notag\\
&\text{loop over}\space  j := 1 \rightarrow N_k\space  \text{for each j-th inner iteration} \notag\\
&\text{compute}\space  O_i = \mathrm{diag}(l_{i}^{(N_k)})^{-1} \tilde{O_i}^{(N_k)}\in \mathbb{R}^{B_q\times d}\space  \text{and write it to HBM to return as output} \notag\\
&\text{compute}\space  \mathrm{LSE_i} = m_i^{(N_k)} + \log(l_i^{(N_k)})\in \mathbb{R}^{B_q} \space  \text{and write it to HBM to save for backward} \notag
\end{align}
\end{cases}
$$
$$
\begin{align}
&\text{where}\quad \text{LSE}( \mathbf{x}) := \log\left(\sum\limits_{i=1}^n \exp(x_i)\right) = \max( \mathbf x) + \text{LSE}( \mathbf{x}-\max( \mathbf x)),\space   \mathbf x \in \mathbb{R}^{n},\\
&\text{and}\space \tilde{O_i} \space\text{is the un-normalized} \space O_i, \space\text{i.e.}\space O_i = \mathrm{diag}(l_{i})^{-1}\tilde{O_i}
\end{align}
$$

* in which each inner loop goes across $j := 1 \rightarrow N_k$ for each block of $K_j,V_j$ to update $\tilde{O_i}^{(j)}, l_i^{(j)}, m_i^{(j)}$, where $N_k = \lceil\frac{N}{B_k}\rceil$

$$
\text{in one j-th inner iteration}: 
\begin{cases}
\begin{align} 
&\text{load}\space  K_j, V_j \in \mathbb{R}^{B_k\times d}\space  \text{from HBM to SRAM} \notag\\
&\text{compute}\space  P_{i}^{(j)} = \text{mask}(Q_iK_j^{\mathrm T} + bias) \in \mathbb{R}^{B_q\times B_k} \notag\\
&\text{update}\space  m_i^{(j)} = \max\big(m_i^{(j-1)}, \mathrm{rowmax}(P_{i}^{(j)})\big) \in \mathbb{R}^{B_q} \notag\\
&\text{compute}\space S_i^{(j)} = \exp(P_i^{(j)} - m_i^{(j)}) \in \mathbb{R}^{B_q\times B_k} \notag\\
&\text{update}\space  l_i^{(j)} = \delta_{m_i^{(j)}}l_i^{(j-1)} + \mathrm{rowsum}(S_i^{(j)})\in \mathbb{R}^{B_q}  \notag\\
&\text{update}\space  \tilde{O_i}^{(j)} = \mathrm{diag}(\delta_{m_i^{(j)}})^{-1}\tilde{O_i}^{(j-1)} + S_i^{(j)}V_j\in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
$$
$$
\begin{align}
&\text{where}\quad \delta_{m_i^{(j)}} := \exp(m_i^{(j-1)} -m_i^{(j)})
\end{align}
$$

## Backward

### For standard attn backward:

$$
\begin{cases}
\begin{align}
&\mathrm{d}{V} = A^{\mathrm T} \mathrm{d}{O} \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{A} = \mathrm{d}{O}V^{\mathrm T} \in \mathbb{R}^{N\times N} \notag \\
&\mathrm{d}{P_{i:}} = \cfrac{\partial A_{i:}}{\partial P_{i:}}\cdot\mathrm{d}{A_{i:}}\in \mathbb{R}^{N}, \quad where\space  \cfrac{\partial A_{i:}}{\partial P_{i:}} = J_{softmax} = \mathrm{diag}(A_{i:}) - A_{i:}A_{i:}^{\mathrm T} \in \mathbb{R}^{N\times N} \notag \\
&\mathrm{d}{Q} = \mathrm{d}{P}K \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{K} = \mathrm{d}{P}^{\mathrm T}Q \in \mathbb{R}^{N\times d} \notag
\end{align}
\end{cases}
$$
$$
\begin{align}
&\text{where}\space\space \mathrm{d}X \space\space\text{denotes}\space \cfrac{\partial{\mathbb{loss}}}{\partial{X}}, \space\text{and}\space X_{i:} \space\text{denotes the column vector made of the $i$-th row of}\space X, \space\text{for any matrix}\space X
\end{align}
$$

given $\mathrm{d}{O} \in \mathbb{R}^{N\times d}$


### For flash-attn backward:


#### step0. store LSE during forward to save memory:

$$
\text{for i-th row}: 
\begin{cases}
\begin{align} 
&\text{since}\space  A_{i:} = \cfrac{S_{i:}}{l_{i:}} \in \mathbb{R}^{B_k}, \quad l_{i} = \mathrm{sum}(S_{i:}) \in \mathbb{R}, \quad S_{i:} = \exp(P_{i:} - m_{i}) \in \mathbb{R}^{B_k}, \quad m_{i} = \max(P_{i:})\in \mathbb{R} \notag\\
&\text{therefore}\space  A_{i:} = \cfrac{\exp(P_{i:} - m_{i})}{\mathrm{sum}(\exp(P_{i:} - m_{i}))} = \cfrac{\exp(P_{i:} - m_{i})}{\exp(\mathrm{LSE}(P_{i:} - m_{i}))} = \exp(P_{i:} - (m_{i} + \mathrm{LSE}(P_{i:} - m_i))) \notag\\
&\text{and according to}\space  \text{LSE}( \mathbf{x}) = \max( \mathbf x) + \text{LSE}( \mathbf{x}-\max( \mathbf x)) \notag\\
&\text{therefore}\space  A_{i:} = \exp(P_{i:} - (m_{i} + \mathrm{LSE}(P_{i:} - m_i))) = \exp(P_{i:} - \mathrm{LSE}(P_{i:})) = \exp(P_{i:} - \mathrm{LSE_i})\notag
\end{align}
\end{cases}
$$

so we can jump storing $m_i, l_i$ to compute $S_{i:}$, but computing $A_{i:}$ from $P_{i:}$ directly with only $\mathrm{LSE_i}$


#### step1. compute Delta during preprocessing to save memory:

$$
\text{for i-th row}: 
\begin{cases}
\begin{align} 
&\text{since}\space  \mathrm{d}{P_{i:}} = \cfrac{\partial A_{i:}}{\partial P_{i:}}\cdot\mathrm{d}{A_{i:}} = (\mathrm{diag}(A_{i:}) - A_{i:}A_{i:}^{\mathrm T} )\cdot\mathrm{d}{A_{i:}} = A_{i:}\odot\mathrm{d}{A_{i:}} - (A_{i:}A_{i:}^{\mathrm T})\mathrm{d}{A_{i:}}  \in \mathbb{R}^{B_k}\notag\\
&\text{then}\space  \mathrm{d}{P_{i:}} = A_{i:}\odot\mathrm{d}{A_{i:}} - A_{i:}(A_{i:}^{\mathrm T}\mathrm{d}{A_{i:}}) = A_{i:}\odot\mathrm{d}{A_{i:}} - (A_{i:}^{\mathrm T}\mathrm{d}{A_{i:}})A_{i:}\notag\\
&\text{define}\space  \Delta_{i} = A_{i:}^{\mathrm T}\mathrm{d}{A_{i:}}  \in \mathbb{R}, \space  \text{and because}\space  \mathrm{d}{A_{i:}} = (\mathrm{d}{O_{i:}}^{\mathrm T}V^{\mathrm T})^{\mathrm T} = VdO_{i:}  \in \mathbb{R}^{B_k}\notag\\
&\text{so}\space  \Delta_{i} = A_{i:}^{\mathrm T}\mathrm{d}{A_{i:}} = A_{i:}^{\mathrm T}(VdO_{i:}) = (A_{i:}^{\mathrm T}V)dO_{i:} = O_{i:}^{\mathrm T}dO_{i:}\notag\\
\end{align}
\end{cases}
$$
$$
\begin{align}
&\text{then for all rows, we compute }\space  \Delta = \mathrm{rowsum}(O\odot dO)\in \mathbb{R}^{B_q}\space  \text{during preprocessing} \notag
\end{align}
$$

so we can avoid massive matrix computing like $A_{i:}A_{i:}^{\mathrm T} \in \mathbb{R}^{B_k\times B_k}$


#### step2. flash-attn backward algorithm with recomputation (double-loop):

* the outer loop runs through $j := 1 \rightarrow N_k$ for each block of $K_j, V_j$ to compute $dK_j, dV_j$,  where $N_k = \lceil\frac{N}{B_k}\rceil$

$$
\text{in one j-th outer iteration}: 
\begin{cases}
\begin{align} 
&\text{load}\space  K_j, V_j \in \mathbb{R}^{B_k\times d}\space  \text{from HBM to SRAM, and initialize}\space  dK_j^{(0)}, dV_j^{(0)} = (0)_{B_c\times d} \in \mathbb{R}^{B_k\times d} \notag \\
&\text{loop over}\space  i := 1 \rightarrow N_q\space  \text{for each i-th inner iteration} \notag \\
&\text{write}\space  dK_j = dK_j^{(N_q)}, dV_j = dV_j^{(N_q)} \space \text{back to HBM to return as output} \notag
\end{align}
\end{cases}
$$


* in which each inner loop goes across $i := 1 \rightarrow N_q$ for each block of $Q_i, dO_i$ to update $dQ_i, dK_j^{(i)}, dV_j^{(i)}$, where $N_q = \lceil\frac{N}{B_q}\rceil$

$$
\text{in one i-th inner iteration}: 
\begin{cases} 
\begin{align} 
&\text{load}\space  Q_i, dO_i, \mathrm{LSE_i}, \Delta_i\space  \text{from HBM to SRAM} \notag \\
&\text{recompute}\space  P_j^{(i)} = Q_iK_j^{\mathrm T} \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{recompute}\space  A_j^{(i)} = \exp(P_j^{(i)} - \mathrm{LSE_i}) \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{update}\space  dV_j^{(i)} = dV_j^{(i-1)} + (A_j^{(i)})^{\mathrm T} dO_i \in \mathbb{R}^{B_k\times d} \notag \\
&\text{compute}\space  dA_j^{(i)} = dO_iV_j^{\mathrm T} \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{compute}\space  dP_j^{(i)} = A_j^{(i)}\odot (dA_j^{(i)} - \Delta_i) \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{update}\space  dK_j^{(i)} = dK_j^{(i-1)} + (dP_j^{(i)})^{\mathrm T} Q_i \in \mathbb{R}^{B_k\times d} \notag \\
&\text{update}\space dQ_i \stackrel{atomic\space add}\longleftarrow dP_j^{(i)}K_j \in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
$$