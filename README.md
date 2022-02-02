# Detection of composite transposable elements:

<p align="center">
    <img src="https://github.com/DMH-dutte/Detection_of_composite_transposable_elements/blob/main/preview/frequency_landscape.png" width="400" />
</p>

## Description:
Transposable elements are sequences in genomes that can change their position in the genome. Thus, they are also called “jumping genes”. They are able to affect the composition and size of genetic replicons. Our research interest in this project are composite transposable elements, which are flanked by two inverted repeats and transposable elements. Composite transposable elements are moving as one unit within a genome and are copying and inserting genes enclosed by itself. The following traits of composite transposable elements are making their detection challenging: 

1. Sometimes terminal information such as repeats or transposable elements are missing, which would theoretically determine the boundaries of a composite transposable element.
2. Composite transposable elements are diverse in their genetic composition and size. 

Composite transposable elements are usually associated with essential and indispensable genes, which are having a high gene frequency across genomes, but also with genes of lower essentiality, which leads to significant drop in the gene frequency landscape. We hypothesize that the genetic frequency landscape of a replicon will follow a particular pattern, which can be used as a marker for putative regions of composite transposable elements. Thus, we are representing here an approach to detect regions of putative composite transposable elements using gene frequencies, protein family clusters and positions of composite transposable elements as input for a supervised LSTM-based neural network model. 

### HPC cluster:
This project was executed on the NEC HPC-System (nesh) of the University of Kiel (CAU). 4 x NVIDIA Tesla V100-GPU cards have been used for training. <br/>
Thus, it its possible that images in the notebook code/3_LSTMs.ipynb are not displayed properly. You'll find the graphs in code/graphs/ and further information on how the job was committed to the cluster in /code/HPC. 

### Participants:
Yiqing Wang<br/>Dustin Hanke


### Contact information:
dhanke@ifam.uni-kiel.de<br/>
https://www.mikrobio.uni-kiel.de/de/ag-dagan/team

### Course:
Machine Learning with TensorFlow


### Semester:
WiSe2122

### Data:
The data used in this approach corresponds to the genomic microbiology group at institute of general microbilogy (Christian-Albrechts-Universität zu Kiel/University of Kiel).
<br/>https://www.mikrobio.uni-kiel.de/de/ag-dagan

### Website:
https://github.com/DMH-dutte<br/>
https://www.mikrobio.uni-kiel.de/de/ag-dagan/team

