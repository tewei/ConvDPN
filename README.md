# ConvDPN
Chinese Dependency Parser with Text Convolutional Neural Networks

## Intro
This is a graph-based non-projective dependency parser for traditional Chinese.
For more info check out [Universal Dependencies](http://universaldependencies.org/).

The parser consists two modified version of [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181).
These two CNNs predict edges and labels separately and the final tree is generated by a directed MST

The performance is roughly UAS = 71.87% and LAS = 68.74% on dev set (with gold standard segmentation / POS).


## Usage
Please download the zh_gsd treebank from [UD_Chinese-GSD](https://github.com/UniversalDependencies/UD_Chinese-GSD/). 
Then create two folders: UD_data and skip-gram. 
Put the treebank files in UD_data and word2vec models (dim=100 by default, please include all 3 files generated by gensim) in skip-gram.
You can find a skip-gram word2vec model [here](https://github.com/voidism/Chinese_Sentence_Dependency_Analyser).

### Train the models with 
```
python convdpn.py
```
### Test the models with 

```
python convdpn_parser.py -i test_set.csv -o result.csv 
```

Please look at the example test_set.csv for input format!

The MST algorithm is from [here](https://github.com/mlbright/edmonds/blob/master/edmonds/edmonds.py).
