#!/usr/bin/env bash
# Install Odin-Multi, its pinned ColabDesign fork, and AlphaFold 2 weights.

set -o pipefail

pkg_manager="conda"
cuda=""
environment_name="${ODIN_MULTI_ENV_NAME:-Odin-Multi}"

usage() {
    cat <<EOF
Usage: bash install_odin_multi.sh [OPTIONS]

Options:
  -p, --pkg-manager {conda,mamba}
      Conda-compatible package manager to use (default: conda).
  -c, --cuda VERSION
      CUDA version exposed to Conda, for example 12.4.
  -h, --help
      Show this help message and exit.
EOF
}

OPTIONS=p:c:h
LONGOPTIONS=pkg_manager:,pkg-manager:,cuda:,help

if ! PARSED=$(getopt \
    --options="${OPTIONS}" \
    --longoptions="${LONGOPTIONS}" \
    --name "$0" \
    -- "$@"
); then
    usage >&2
    exit 2
fi
eval set -- "$PARSED"

while true; do
    case "$1" in
        -p|--pkg_manager|--pkg-manager)
            pkg_manager="$2"
            shift 2
            ;;
        -c|--cuda)
            cuda="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${pkg_manager}" in
    conda|mamba) ;;
    *)
        echo "Error: --pkg-manager must be either conda or mamba." >&2
        exit 2
        ;;
esac

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not installed or cannot be found on PATH." >&2
    exit 1
fi
if ! command -v "${pkg_manager}" >/dev/null 2>&1; then
    echo "Error: ${pkg_manager} is not installed or cannot be found on PATH." >&2
    exit 1
fi

echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
install_dir="${script_dir}"
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

### Odin-Multi install begin, create base environment
echo -e "Installing ${environment_name} environment\n"
"${pkg_manager}" create --name "${environment_name}" python=3.10 -y || { echo -e "Error: Failed to create ${environment_name} conda environment"; exit 1; }
conda env list | grep -w "${environment_name}" >/dev/null 2>&1 || { echo -e "Error: Conda environment '${environment_name}' does not exist after creation."; exit 1; }

# Load the newly created environment
echo -e "Loading ${environment_name} environment\n"
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/${environment_name}" || { echo -e "Error: Failed to activate the ${environment_name} environment."; exit 1; }
[ "${CONDA_DEFAULT_ENV}" = "${environment_name}" ] || { echo -e "Error: The ${environment_name} environment is not active."; exit 1; }
echo -e "${environment_name} environment activated at ${CONDA_BASE}/envs/${environment_name}"

# install required conda packages
echo -e "Installing conda requirements\n"
if [ -n "$cuda" ]; then
    CONDA_OVERRIDE_CUDA="$cuda" "${pkg_manager}" install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib=*=*cuda* jax"<0.6.0" cuda-nvcc cudnn -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu -y || { echo -e "Error: Failed to install conda packages."; exit 1; }
else
    "${pkg_manager}" install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib jax"<0.6.0" cuda-nvcc cudnn -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu -y || { echo -e "Error: Failed to install conda packages."; exit 1; }
fi

# make sure all required packages were installed
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ "${#missing_packages[@]}" -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# install ColabDesign
echo -e "Installing ColabDesign\n"
colabdesign_dir="${install_dir}/ColabDesign"
git -C "${install_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo -e "Error: Odin-Multi must be installed from a Git clone so its ColabDesign submodule can be initialized."; exit 1; }

if [ -e "${colabdesign_dir}/.git" ]; then
    colabdesign_status=$(git -C "${colabdesign_dir}" status --porcelain --untracked-files=all) || { echo -e "Error: Could not inspect the existing ColabDesign checkout."; exit 1; }
    [ -z "${colabdesign_status}" ] || { echo -e "Error: Refusing to update a modified ColabDesign checkout."; exit 1; }
fi

git -C "${install_dir}" submodule sync -- ColabDesign || { echo -e "Error: Failed to synchronize the ColabDesign submodule URL."; exit 1; }
git -C "${install_dir}" submodule update --init --recursive --checkout -- ColabDesign || { echo -e "Error: Failed to initialize the pinned ColabDesign submodule."; exit 1; }
git -C "${install_dir}" submodule absorbgitdirs ColabDesign || { echo -e "Error: Failed to normalize the ColabDesign submodule metadata."; exit 1; }

colabdesign_commit=$(git -C "${install_dir}" rev-parse HEAD:ColabDesign 2>/dev/null) || { echo -e "Error: Could not read the pinned ColabDesign Gitlink."; exit 1; }
colabdesign_head=$(git -C "${colabdesign_dir}" rev-parse HEAD 2>/dev/null) || { echo -e "Error: Could not read the initialized ColabDesign revision."; exit 1; }
[ "${colabdesign_head}" = "${colabdesign_commit}" ] || { echo -e "Error: ColabDesign is at ${colabdesign_head}; expected pinned revision ${colabdesign_commit}."; exit 1; }
colabdesign_status=$(git -C "${colabdesign_dir}" status --porcelain --untracked-files=all) || { echo -e "Error: Could not inspect the ColabDesign submodule."; exit 1; }
[ -z "${colabdesign_status}" ] || { echo -e "Error: Refusing modified ColabDesign submodule."; exit 1; }

pip3 install --editable "${colabdesign_dir}" --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
ODIN_MULTI_COLABDESIGN_DIR="${colabdesign_dir}" python -c 'import os; from pathlib import Path; import colabdesign; import colabdesign.af.design as design; root = Path(os.environ["ODIN_MULTI_COLABDESIGN_DIR"]).resolve(); Path(colabdesign.__file__).resolve().relative_to(root); Path(design.__file__).resolve().relative_to(root)' >/dev/null 2>&1 || { echo -e "Error: Python did not import the packaged ColabDesign checkout"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2 weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
conda deactivate
echo -e "${environment_name} environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
"${pkg_manager}" clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS
echo -e "Successfully finished ${environment_name} installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate ${environment_name}\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
