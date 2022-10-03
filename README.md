# MTDAL
Multiple-model Time-sensitive Dynamic Active Learning For RGCN Model Extraction Attacks

TGCN file presents different types of recurrent graph convolutional networks.

ModelExtraction file simulate the MTDAL strategy on DBLP5 and DBLP3 datasets, and testing other strategies on Chickenpox and EngCovid datasets

In ModelExtraction subfile,  'active learning' presents serveral active learning strategies on dynamic graphs. In other subfiles, 'attack.py' trains the attack RGCN model, 'victim.py' trains the oracle RGCN model. 'models.py' constructs all RGCN models in RGCN committee.
