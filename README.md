**KECNW**

This repository implements the proposed models in the paper 


<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417417308473" >"A graph based keyword extraction model using collective node weight"</a>

---
**runing the code**

1) Install libraries in requirements.txt 

    `pip install -r requirements.txt`


2) Run below 

   `python main.py -d 'data/trump_20200530.csv' -t 'text' -w DC-SC-F-L-TF-Neigh_Imp
`
   
  
   -d: dataset file as csv file
   
   -t: text field name like "text"
   
   -w: list of weighting criterias with dash as spliter (ex: DC-SC-Neigh_Imp-F-L-TF)
   
      DC: Distance from central node
   
      SC: Selectivity Centrality:
      Neigh_Imp: Importance of neighboring nodes
   
      F :  Position of a node (nf/freq(i), where, nf is the number of times i is the first word)
      freq(i) is the frequency of the term i
   
      L :  Position of a node (nl/freq(i), where, nl is the number of times i is the last word)
      freq(i) is the frequency of the term i
   
      TF: Term frequency
the output would be list of weighted keywords:

`
[('Trump', 0.5503164870433442), ('fact', 0.42678799899425307), ('George', 0.32278482851305984), ('us', 0.32000000000000006), ('Floyd', 0.26582279995193164), ('Minneapolis', 0.2441229795476923), ('Plus', 0.22967080346226096), ('check', 0.17492089603979788), ('Guard', 0.17088608568338465), ('National', 0.17088608568338465)]
`

