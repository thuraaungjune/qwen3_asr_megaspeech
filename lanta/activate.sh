# Activate the training env in the current shell:   source lanta/activate.sh
# Used by train_common.sh too. Ignores ~/.local site-packages so a stray user-level
# pandas/numpy for another Python can't shadow the env's packages.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/config.sh"
module load Mamba/23.11.0-0
_had_u=0; [[ $- == *u* ]] && _had_u=1; set +u      # conda's activate scripts reference unset variables
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
[ $_had_u = 1 ] && set -u
export PYTHONNOUSERSITE=1
export PATH="$ENV_PREFIX/bin:$PATH"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "ERROR: $ENV_PREFIX/bin/python not found - run 'bash lanta/setup_env.sh' on the frontend first" >&2
  return 1 2>/dev/null || exit 1
fi
echo "env: $(command -v python) ($(python -V 2>&1))"
