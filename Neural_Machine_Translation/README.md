Part 1:
1) NMT without attention (nmt_dynet.py) 
   I increased the size of hidden state from 50 to 65 (50 * 65), then bleu score can be larger than 22.
   Best bleu score is around 22.5662890561.

2) NMT using beam search without attention (nmt_dynet_beam.py)
   I tried several K vlaues(2 to 5) and found when K equals to two, the bleu score was the best.
   K = 2: Best bleu score is 23.5878756893.
   Beam search is in the generate function.

Part 2:
1) NMT with attention (nmt_dynet_attention.py)
   I increased the size of gru_d from 50 to 125 and the size of word_d from 50 to 100 (100 * 125), then bleu score can be larger than 25.
   Best bleu score is 25.8072987796.


2) NMT using beam search with attention (nmt_dynet_attention_beam.py)
   I tried several K values(2 to 5) and found when K equals to 2 and kept same layer size as part2.1, the bleu score was the best.
   Best bleu score is 26.1133077647.
   Beam search is in the generate function and I chose this model to join the competition.

     |no beam & no attend|beam search & no attend|no beam & attend|beam search & attend|
   --|-------------------|-----------------------|----------------|--------------------|
   0 |   12.6149992763   |     14.6881169907     | 16.6456853685  |    17.0766251936   |
   1 |   16.5359460474   |     14.5018859406     | 18.3096242419  |    18.9231562344   |
   2 |   17.2295831365   |     16.3963507909     | 21.8552133277  |    22.7143655712   |
   3 |   16.6491853486   |     18.3246806525     | 23.1212081679  |    22.7932652372   |
   4 |   18.9201624112   |     19.6419909801     | 24.3245674312  |    22.8203346621   |
   5 |   19.5441103419   |     18.2875366954     | 23.4856846252  |    23.7039265225   |
   6 |   20.4625531159   |     20.062649607      | 25.2321400568  |    24.5430701511   |
   7 |   20.2039835984   |     19.7205699216     | 23.4122692027  |    24.9397950268   |
   8 |   19.1743324613   |     20.8953036417     | 23.7853937999  |    24.5024257929   |
   9 |   21.7036874223   |     19.7773937203     | 25.0577763786  |    24.2914469766   |
   10|   20.5514482032   |     21.1887383488     | 25.4258748629  |    25.4036880906   |
   11|   20.8115697463   |     22.2190962515     | 25.1066768688  |    25.2082961055   |
   12|   22.4359506396   |     22.7868694602     | 25.4063846119  |    24.9542139692   |
   13|   21.7806980023   |     22.8090197341     | 24.6890560786  |    26.1133077647   |
   14|   22.5662890561   |     23.2166020453     | 24.8599039537  |    24.7708235209   |
   15|   21.6218502722   |     22.081455289      | 25.8072987796  |    25.4671510712   |
   16|   21.6541793392   |     22.5923782316     | 25.7708329885  |    25.252556962    |
   17|   21.7196090598   |     21.7050589829     | 24.9379023964  |    24.7795578804   |
   18|   22.2997945302   |     21.193379189      | 24.4903615418  |    24.6788504788   |
   19|   21.896668936    |     23.5878756893     | 24.3816464953  |    24.9961569792   |
