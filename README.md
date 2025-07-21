![dot-pe logo](dot-pe-logo-white-bg.png)

## Purpose

DOT-PE is a Python package for parameter estimation and evidence integration using data from the gravitational wave interferometer observatories LIGO, Virgo, and KAGRA. Unlike traditional approaches that rely on stochastic samplers, DOT-PE performs parameter estimation and evidence integration using matrix multiplications for fast likelihood evaluation. All interfacing with gravitational wave data, waveform generation, and sampling tools is handled through the [`cogwheel`](https://github.com/jroulet/cogwheel) package.

## Installation

### Basic Installation

Clone the repository:
```bash
git clone https://github.com/jonatahm/dot-PE.git
cd dot-pe
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dot-pe
```

### IMRPhenomXODE Installation

DOT-PE defaults to using IMRPhenomXODE for waveform generation. This requires additional installation steps (see `cogwheel.waveform_models.xode.py`):

#### 1. Install IMRPhenomXODE

Clone the IMRPhenomXODE repository:
```bash
git clone https://github.com/hangyu45/IMRPhenomXODE.git
cd IMRPhenomXODE
```

Follow the installation instructions in the IMRPhenomXODE repository. Note that IMRPhenomXODE requires `lalsimulation` version >= 5.1.0.

#### 2. Manual Cogwheel Configuration

After installing IMRPhenomXODE, you need to create a symbolic link in the cogwheel package:

```bash
# Find your cogwheel installation path
python -c "import cogwheel; print(cogwheel.__file__)"

# Create symbolic link (replace paths as needed)
cd <path_to_cogwheel>/cogwheel/waveform_models
ln -s <path_to_IMRPhenomXODE>/src/ IMRPhenomXODE
```

**Example:**
```bash
# If cogwheel is installed in your conda environment
cd ~/anaconda3/envs/dot-pe/lib/python3.9/site-packages/cogwheel/waveform_models
ln -s ~/path/to/IMRPhenomXODE/src/ IMRPhenomXODE
```

#### 3. Verify Installation

Test that IMRPhenomXODE is properly configured:
```python
import cogwheel.waveform_models.xode
print("IMRPhenomXODE installation successful!")
```

## Usage

See the notebooks in the `notebooks/` directory for examples.

## License

This project is licensed under the GNU General Public License v3.0 – see the [LICENSE](LICENSE) file for details. 
