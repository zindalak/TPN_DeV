## HPCC Access

Login to HPCC:

``` bash
ssh <MSU_NETID>@hpcc.msu.edu
```

Connect to a development node:

``` bash
ssh dev-amd20
```


## IceCube Environment on HPCC

Navigate to the repository:

``` bash
cd ~/projects/TPN_DeV
```

Load the IceCube environment:

``` bash
source env.sh
```

Verify IceTray:

``` bash
python -c "from icecube import dataio; print('IceCube OK')"
```

## Clone the Repository

``` bash
cd ~/projects
git clone https://github.com/rishibbdb/TPN_DeV.git
cd TPN_DeV
```

## Python Environment Options

### Conda (Recommended)

Conda is recommended because it handles scientific Python packages (JAX,
TensorFlow, NumPy, SciPy and PyArrow) more reliably across platforms.

Install Miniconda:

``` bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Initialize Conda:

``` bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Create and activate the environment:

``` bash
conda create -n tpn-dev python=3.10 -y
conda activate tpn-dev
```

Install the required dependencies following the project requirements.
```bash
python -m pip uninstall -y jax jaxlib ml-dtypes numpy scipy

python -m pip install numpy==1.26.4 scipy==1.12.0

python -m pip install --only-binary=:all: ml-dtypes==0.2.0

python -m pip install "jax[cuda11_pip]==0.4.23" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python -m pip install tensorflow-cpu==2.15.0

python -m pip install -U equinox==0.11.10 jaxtyping==0.3.3 --no-deps

python -m pip install optimistix==0.0.8 --no-deps

python -m pip install lineax==0.0.6 --no-deps

python -m pip install quadax==0.2.9 --no-deps

python -m pip install --only-binary=:all: pyarrow==14.0.2

python -m pip install feather-format
```

### Poetry

Install Poetry:

``` bash
python -m pip install --user poetry
export PATH="$HOME/.local/bin:$PATH"
poetry --version

Initialize Poetry:

``` bash
cd ~/projects/TPN_DeV
poetry init

poetry add numpy==1.26.4 scipy==1.12.0 pyarrow==14.0.2 tensorflow-cpu==2.15.0

poetry add jax==0.4.23 jaxlib==0.4.23

poetry add equinox==0.11.10

poetry add jaxtyping==0.3.3

poetry add optimistix==0.0.8

poetry add lineax==0.0.6

poetry add quadax==0.2.9

poetry add feather-format
'''
Activate:
``` bash
poetry shell
```

## Local IceTray Installation (Windows / WSL)

Install Ubuntu from the Microsoft Store.

``` bash
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential cmake git qtbase5-dev libqt5opengl5-dev \
libboost-all-dev libgsl-dev libfftw3-dev python3-dev zlib1g-dev
```

Clone and build:

``` bash
cd ~
git clone https://://github.com/icecube/icetray-public.git i3/icetray
mkdir -p ~/i3/build
cd ~/i3/build
cmake ../icetray
make -j8
source env-shell.sh
python -c "from icecube import dataio; print('IceTray OK')"
```

## Local IceTray Installation (macOS)

``` bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git cmake boost gsl fftw qt@5 python
git clone https://github.com/icecube/icetray-public.git i3/icetray
mkdir -p ~/i3/build
cd ~/i3/build
cmake ../icetray
make -j$(sysctl -n hw.ncpu)
source env-shell.sh
```

## Launching Steamshovel

``` bash
cd ~/i3/build
source env-shell.sh

./bin/steamshovel \
/path/to/GeoCalibDetectorStatus.i3.gz \
/path/to/event_file.i3
```
