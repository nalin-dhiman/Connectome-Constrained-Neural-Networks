# GitHub Push Instructions

From inside `github_release_repo/`:

```bash
git init
git checkout -b main
git remote add origin https://github.com/nalin-dhiman/Connectome-Constrained-Neural-Networks.git
git add .
git status
git commit -m "Clean repo: revision controls, reproducible pipeline, figures and tables only (no manuscript)"
git push origin main
```

If HTTPS authentication fails:

1. Configure a credential helper:

```bash
git config --global credential.helper store
```

2. Use a GitHub personal access token when prompted, or switch to SSH.

If you prefer SSH:

```bash
git remote set-url origin git@github.com:nalin-dhiman/Connectome-Constrained-Neural-Networks.git
git push origin main
```
