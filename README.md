# Simulating articulatory trajectories with phonological feature interpolation

## Installation

First, clone this repository:

```bash
git clone https://github.com/angelo-ortiz/articulatory-probing.git
cd articulatory-probing
```

Before installing this package, it's best to create a virtual environment.
<details open>
<summary>If you have Python 3.10 installed in your OS and you don't want to use Anaconda (or its 
derivatives), you can use <code>venv</code> for that:</summary>

```bash
python3.10 -m venv ./artprobenv
source ./artprobenv/bin/activate
```

</details>
<details>
<summary>Otherwise, you can create a conda environment:</summary>

```bash
conda create -n artprobenv python=3.10
conda activate artprobenv
```

</details>

Finally, install the package within the virtual environment:

```bash
python3.10 -m pip install -e .[plot]  # add plot support for the notebooks
```

## TODO
