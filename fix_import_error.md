# Fixing the Import Error

The error you're seeing is related to a corrupted or broken `mpmath`/`sympy` installation that transformers depends on.

## Quick Fix

Try reinstalling the problematic packages:

```powershell
pip uninstall sympy mpmath -y
pip install sympy mpmath
```

Or reinstall transformers and its dependencies:

```powershell
pip uninstall transformers torchvision -y
pip install transformers torchvision
```

## Alternative: Use Environment Variable

If the issue persists, you can try setting an environment variable to force CPU and potentially avoid some CUDA-related imports:

```powershell
$env:CUDA_VISIBLE_DEVICES=""
python Training_Scripts/train_pipeline.py --mode full --seed 42 `
  --clustering-algorithm spectral `
  --num-experts 8 `
  --data-percent 0.01 `
  --asr-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --log-file Runs/full/training.log `
  --resume `
  --device cpu
```

## Check Python Path Length

On Windows, very long paths can cause import issues. If your Python installation path is very long, consider:
1. Using a shorter path for your project
2. Enabling long path support in Windows

## Verify Installation

Check if the packages are properly installed:

```powershell
python -c "import sympy; import mpmath; print('OK')"
```

If this fails, the packages are corrupted and need to be reinstalled.
