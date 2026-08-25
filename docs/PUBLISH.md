# Публикация на GitHub (Vizanix)

Локально уже есть `main` с первым коммитом. Remote:

```text
git@github.com:Vizanix/TideGBM.git
```

## 1. SSH-ключ на аккаунте Vizanix

На этой машине push идёт через ключ, который GitHub видит как **Kalashnkv**.
Чтобы пушить в **Vizanix/TideGBM**, добавь тот же публичный ключ:

GitHub → **Vizanix** → Settings → SSH and GPG keys → New SSH key  
(или добавь Kalashnkv как collaborator в репозиторий с правом Write.)

Публичный ключ с машины сборки:

```bash
cat ~/.ssh/id_rsa.pub
```

## 2. Создай пустой репозиторий

https://github.com/new (залогинен как **Vizanix**)

| Поле | Значение |
|---|---|
| Repository name | `TideGBM` |
| Visibility | Public (или Private) |
| Initialize | **ничего** не отмечай (без README / .gitignore) |

## 3. Push

```bash
cd /path/to/buster
git remote -v   # origin → git@github.com:Vizanix/TideGBM.git
git push -u origin main
```

Или одной командой:

```bash
bash scripts/publish_github.sh
```

## 4. (Опционально) GitHub CLI

```bash
gh auth login
gh repo create Vizanix/TideGBM --public --source=. --remote=origin --push
```

## 5. После push

- Actions: `.github/workflows/ci.yml`, `wheels.yml`, `gpu-wheel.yml`
- Topics на странице репо: `gradient-boosting`, `time-series`, `quantitative-finance`, `machine-learning`
- Release: tag `v0.4.0` от текущего `main`

## Что не коммитится

`.gitignore` исключает `.venv/`, `benchmarks/data/yahoo/`, кэши. Секреты (`.env`, ключи) не добавляй.
