
# Application de prédiction de valeur résiduelle LLD/LOA

## Finalité

Cette application Flask charge le modèle CatBoost optimisé produit par
le notebook de modélisation et propose une estimation :

- du prix de marché actuel ;
- du prix futur à marché constant ;
- de la valeur résiduelle en euros et en pourcentage ;
- de la dépréciation estimée.

La valeur résiduelle est un proxy. Elle ne constitue pas une valeur
contractuelle certifiée.

La fourchette affichée est construite à partir des modèles quantiles puis
ajustée, si nécessaire, afin de toujours contenir la prédiction centrale.
L'application signale cet ajustement à l'utilisateur.

## Résultats

Le modèle final retenu est un CatBoost optimisé.

| Métrique | Résultat |
|---|---:|
| MAE | ~2 590 € |
| RMSE | ~4 544 € |
| R² | ~0,85 |
| WAPE | ~18,35 % |

## Prérequis

- Python 3.12 recommandé ;
- modèle `models/modele_ml_valeur_residuelle_v2.joblib` ;
- mémoire suffisante pour charger le modèle CatBoost.

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:5000 --workers 1 --threads 4 --timeout 180
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:5000 --workers 1 --threads 4 --timeout 180
```

Gunicorn n'est pas pris en charge nativement sous Windows. Pour un test
Windows, utiliser :

```powershell
python app.py
```

Puis ouvrir `http://127.0.0.1:5000`.

## Exemple de démonstration

L'application génère automatiquement un véhicule de démonstration réaliste
à partir du jeu de modélisation. Elle privilégie une Volkswagen Golf récente,
puis d'autres véhicules courants si ce modèle n'est pas disponible. Le prix
d'acquisition est aligné sur le prix de marché observé dans les données.

## Tests

```bash
pytest -q
```

Les tests couvrent le chargement du modèle, le formulaire, l'API, les
validations, les fourchettes, les en-têtes de sécurité et les pages d'erreur.

## Vérification de santé

```bash
curl http://127.0.0.1:5000/health
```

## API

Point d'entrée :

```text
POST /api/v1/predict
Content-Type: application/json
```

Le corps JSON reprend les champs du formulaire.



## Protection des données

L'application ne demande pas de données personnelles et ne sauvegarde
pas les simulations dans une base. Les journaux de l'hébergeur restent
soumis à sa politique propre.

## Accessibilité

Le front-end inclut :

- des libellés explicites ;
- un lien d'évitement ;
- une navigation clavier ;
- des messages avec rôle d'alerte ;
- une structure sémantique ;
- une mise en page responsive ;
- une prise en compte de la réduction des animations.

Un audit RGAA complet reste nécessaire avant une utilisation
institutionnelle.

## Structure

```text
app.py           Serveur Flask et validation
model_utils.py   Préparation des variables et prédiction
templates/       Pages HTML
static/          CSS et JavaScript


tests/           Tests fonctionnels
models/          Modèle entraîné
data/            Options du formulaire
```
