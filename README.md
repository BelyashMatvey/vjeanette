# VJeaNETte

## About problem

#### The human immune system has a wide variety of receptors that allow it to recognize an almost unlimited number of antigens. This diversity is formed by the V(D) processJ-recombination is the random assembly of genetic segments V, D, and J.
#### As a result, each B- or T-cell receives a unique receptor, and the combination of such receptors forms the so-called immune repertoire.
![VJ's](pics/pic.png)

*VJ segments*

#### The analysis of this repertoire is of great practical importance.
#### It is used, for example, for:
- studying the immune response to infections,
- analyzing oncological diseases,
- development of vaccines and immunotherapy.
#### However, the key task in this analysis is to correctly identify the V, D, and J segments in the sequences. The purpose of our work was to create a model based on convolutional neural networks for detecting VJ fragments.

## Description
#### This tool allows you to find V, J and CDR3 segments in RNA-seq data by using Convolutional Neural Network.
### Preprocessing

#### Each nucleotide was encoded numerically, and a special symbol was added to align the sequences to a fixed length.
![Preprocessing](pics/prep.png)
#### In addition, we implemented multithreaded processing of FASTQ files, which significantly accelerated data preparation.

*Preprocessing scheme*
### Model
#### A fixed-length sequence obtained after preprocessing is fed to the input of the model. The model is based on the encoder–decoder principle. The encoder gradually compresses the input sequence, extracting more and more abstract features. At this stage, the model learns to recognize local and more global structures in the data.
#### This is followed by the bottleneck layer, which plays the role of a "bottleneck" in which only the most significant information remains.
#### After that, the decoder restores the spatial structure, allowing you to switch back to positional prediction.
#### An important feature is that the model does positional detection, rather than just classifying the entire sequence. That is, at the output we get a probability distribution for each position, separately for V and J segments. For this, the architecture uses two output heads, one for V and the other for J. This allows the model to simultaneously solve two related tasks and take into account their interrelationship.

![Model](pics/model.png)
*Model architecture*

## Installing

#### Just _git clone_ it

## Tutorial

```python
python run.py --in_file file.fastq --device device --logs logs.log --batch_size 32768 --n_cores 1
```

- --in_file - input file
- --device - device (cuda/cpu). Default - cuda
- --logs - log file. Default - logs/progress.log
- --out - output gile. Default - out/out.csv
- --batch_size - size of batches. Default - 32768. You can choose the best value for your video card.
- --n_cores - number of cpu cores. Default - 1. 


## Support

#### Contact with us via gmail:neonlight20006@gmail.com

## Contributing
#### Beliakov Matvei, Department of Biology, Saint-Petersburg State University
#### Vlasova Elisaveta, Immunosequencing Algorithms Laboratory, RNRMU
#### Mikhail Shugay, Immunosequencing Algorithms Laboratory, RNRMU
## Project status
#### Under Developing