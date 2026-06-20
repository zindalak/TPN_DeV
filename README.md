
Login to HPCC:

--
ssh <MSU_NETID>@hpcc.msu.edu

Connect to a development node:
-
ssh dev-amd20
--

2. ICECUBE ENVIRONMENT ON HPCC

Load IceCube CVMFS environment:

--
export SROOTBASE=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0

eval `$SROOTBASE/setup.sh`
--

Verify:

--
echo $SROOT

which python
--

Verify IceTray:

--
python -c "from icecube import dataio; print('IceTray OK')"
--

3. CLONE TPN_DeV

--
cd ~

git clone https://github.com/rishibbdb/TPN_DeV.git

cd TPN_DeV
--

3. PANDELNET / JAX ENVIRONMENT
--
python -m pip uninstall -y jax jaxlib ml-dtypes numpy scipy

python -m pip install "numpy==1.26.4" "scipy==1.12.0"

python -m pip install --only-binary=:all: "ml-dtypes==0.2.0"

python -m pip install "jax[cuda11_pip]==0.4.23" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python -m pip install tensorflow-cpu==2.15.0

python -m pip install -U "equinox==0.11.10" "jaxtyping==0.3.3" --no-deps

python -m pip install "optimistix==0.0.8" --no-deps

python -m pip install "lineax==0.0.6" --no-deps

python -m pip install "quadax==0.2.9" --no-deps

python -m pip install --only-binary=:all: "pyarrow==14.0.2"
--

Verify:

--
python -c "import jax,numpy,scipy,pyarrow; print('Environment OK')"
--
4. REQUIRED PYTHON PACKAGES

--
python -m pip install --user feather-format
--

Verify:

--
python -c "import feather; print('feather OK')"
--
6. INSTALL UBUNTU

Install Ubuntu from Microsoft Store.

Open Ubuntu terminal.

Update:

--
sudo apt update

sudo apt upgrade -y
--

Verify WSL:

--
uname -a
--
Expected:
--
microsoft-standard-WSL2
--
7. CLONE ICETRAY

--
cd ~

git clone https://github.com/icecube/icetray.git i3

cd i3
--
Create build directory:
--
mkdir build

cd build
--

8. INSTALL ICETRAY DEPENDENCIES
--
sudo apt install \
build-essential \
cmake \
git \
qtbase5-dev \
libqt5opengl5-dev \
libboost-all-dev \
libgsl-dev \
libfftw3-dev \
python3-dev \
zlib1g-dev
--

9. CONFIGURE ICETRAY
--
cd ~/i3/build

cmake ..

10. BUILD ICETRAY

--
make -j8
--

Verify:

--
ls ~/i3/build/bin/steamshovel
--

==================================================
11. DOWNLOAD EXAMPLE EVENT
==================================================

Copy from HPCC:

--
scp <NETID>@dev-amd20:/mnt/research/IceCube/Gupta-Reco/l322645/0000000-0000999/FinalLevel_NuMu_NuGenCCNC.022853.000354.i3.zst .
--
Decompress:
--
unzstd FinalLevel_NuMu_NuGenCCNC.022853.000354.i3.zst

12. TROUBLESHOOTING

Error:
--
ModuleNotFoundError: No module named 'icecube'
--

Fix:
--
export SROOTBASE=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0

eval `$SROOTBASE/setup.sh`
--


Error:

ModuleNotFoundError: No module named 'feather’

Fix:

--
python -m pip install --user feather-format
--
Error:

--
No objects to concatenate
--

Cause:

--
No files matched input pattern
--

Verify:

--
print(infiles)


see https://github.com/HansN87/TriplePandelReco_JAX/tree/main/examples/scripts for example usage.
