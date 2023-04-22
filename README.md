# TODO:
This is a neural network made in raw Python. It's meant to recognize a number in a grayscale 256x256 image, preferably against a white background.
## Setting up
Requires Python (^3.11.4) and `jupyter-labs`.
### Virtual environment
```sh
py -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name nrnn-venv --display-name="Number recognition NN"
```
### Starting
- `jupyter notebook` (use this instead of `jupyter-lab` to use this folder as the working directory)
- Select the new kernel on Jupyter.
- Run desired code blocks.