This is a very simple neural network made in raw Python. It's meant to recognize a number in a grayscale 28x28x1 image.

Also, I realized after making this that it is similar to the model in [3Blue1Brown's Neural Network YouTube series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (it's amazing, go check it!).
## Details
This model of neural networks is called "multilayer perceptron", aka. plain vanilla.

It was trained with the MNIST handwriting dataset with a dark foreground against a white background.

It will take a 28x28 (784 pixels) array input with values ranging 0-1.

It uses backpropagation to correct for its error over the output nodes, which identify numbers ranging 0-9.

It does NOT have biases, only weights.
## Setting up
Requires Python and ` jupyterlab`. If you don't have it:
```sh
pip install jupyterlab
```
Then, you may run `notebooks/setup.ipynb`, or follow below.
### Virtual environment
```sh
py -m venv venv
./venv/Scripts/activate
# install requirements
pip install -r requirements.txt
# create kernel with current environment
python -m ipykernel install --user --name nrnn-venv --display-name="Number recognition NN"
```
### Starting
- `jupyter-lab .` (or `jupyter notebook`)
- Select the new kernel on Jupyter.
- Run code cells.
## Training
Run up until the 2nd cell. It will train from the MNIST handwriting dataset (as a `csv`) file `data/minst_train.csv`, which is too big to upload to git so you may want to download from these links:
- http://pjreddie.com/projects/mnist-in-csv (the one I used)
- https://python-course.eu/data/mnist/mnist_train.csv

It may take up to 6 mins/epoch, based on the empirical evidence of running it on my okay-ish CPU (i5-3470 3.2GHz 4 cores).
## Testing/running
Saving and loading weight parameters is not something that I have implemented, so make sure to run the training code cell beforehand.
You can run the 3rd cell to train on some test data, or add your own image under `data/handwriting.png` and run the 4th cell.
## Going forward
This project was more so just to test the waters of ML and how it works on a low-level. I think if I were to work on this again, it would be interesting to remake it as something that's more like [this](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65?gi=bd0f27d1189b), making it modular & other upgrades.
