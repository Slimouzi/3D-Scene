# Reconstruction de scènes depuis des photos 360

Étude R&D initiée le 1er octobre 2026 sur 13 panoramas Ricoh Theta Z1.
Cible confirmée : visite immersive photoréaliste sur le Web.

La [note CTO](</Users/stani/code/3D Scene/Output/RD/01_Note_CTO.md>) présente la recommandation, les constats et les jalons. Elle renvoie à l’état de l’art et à la spécification du pipeline.

La [revue critique CTO](</Users/stani/code/3D Scene/CTO/01_Revue_critique_RnD.md>) challenge ces recommandations. Les [instructions de développement](</Users/stani/code/3D Scene/CTO/02_Instructions_developpement.md>) fixent le périmètre du premier lot, les interfaces, le backlog et la recette. Ces instructions priment pour l’implémentation initiale sur les options encore ouvertes du dossier R&D.

## État actuel

- Audit local exécuté : métadonnées, lisibilité, empreintes et planches d’inspection.
- Recherche technologique et protocole expérimental rédigés.
- Reconstruction sparse CPU exécutée sur 13 panoramas : 13/13 enregistrés dans une composante, avec diagnostic des poses et nuage sparse.
- Les artefacts de reconstruction sont dans `Output/runs/<run-id>/`, notamment `poses.json`, `quality.json`, `report.md` et `sfm/sparse/<component>/`.
- La partition panorama est produite comme `split.json` provisoire; la navigation et le sol restent à revoir.
- Segmentation SAM 3 implémentée (`theta_pipeline/segmentation/`) et testée avec un segmenteur simulé ; aucune inférence SAM 3 réelle n’a encore été exécutée. Environnement GPU non qualifié tant que `tests/test_gpu_acceptance.py` n’a pas réussi sur la VM.
- Entraînement Gaussian Splatting, densification MVS et lecteur final : non exécutés.
- Les originaux restent dans `Input/`. Le dossier `outpout` mentionné initialement n’était pas présent ; les livrables sont dans `Output/RD/`.

## Segmentation SAM 3 sur VM GPU

SAM 3 s’exécute uniquement sur une VM Linux x86_64 avec CUDA, dans l’environnement figé par `requirements/gpu.lock.txt` (Python 3.12.8, PyTorch 2.14.1, CUDA 13.0, Triton 3.8.0, SAM 3 au commit `2345a4ad`). L’environnement macOS ne prouve rien sur la disponibilité GPU. Le runner refuse un code non commité et vérifie l’environnement contre le lock : tout écart produit `UNKNOWN`, jamais de masque.

Codes de sortie du runner : `0` PASS, `2` UNKNOWN, `3` FAIL, `1` refus ou erreur d’intégrité.

```sh
# Mac : publier le code, noter le commit
git push && git rev-parse HEAD
rsync -av Input/ <vm>:3D-Scene/Input/           # les originaux ne sont pas dans Git

# VM : cloner le commit exact, vérifier les entrées, créer les environnements
git clone git@github.com:Slimouzi/3D-Scene.git && cd 3D-Scene && git checkout <commit>
(cd Input && sha256sum -c ../configs/salon.inputs.sha256)
uv python install 3.12.8
uv venv --python 3.12.8 .venv-sfm && uv pip install --python .venv-sfm/bin/python -r requirements/sfm.lock.txt
uv venv --python 3.12.8 .venv-gpu && uv pip install --python .venv-gpu/bin/python -r requirements/gpu.lock.txt
hf auth login                                     # facebook/sam3 est un dépôt à accès restreint

# VM : créer le run à ce commit (ne pas supposer qu’il existe)
for a in audit prepare seg-faces; do .venv-sfm/bin/python -m theta_pipeline $a --run-id salon-sam3-001; done

# VM : essai sur un seul panorama
.venv-gpu/bin/python -m theta_pipeline.segmentation --run Output/runs/salon-sam3-001 --trial R0010004
THETA_GPU_RUN=Output/runs/salon-sam3-001 .venv-gpu/bin/python -m unittest -v tests.test_gpu_acceptance
```

Inspecter ensuite `segmentation/trial/` : `trial_gate.json`, `semantic_masks_preview/`, `mask_consistency.json`, `mask_provenance.json`. **Uniquement si l’essai renvoie PASS (code 0)** :

```sh
.venv-gpu/bin/python -m theta_pipeline.segmentation --run Output/runs/salon-sam3-001 --all
```

Ne jamais lancer `--all` après un essai UNKNOWN ou FAIL ; le runner le refuse. Corriger la cause (environnement, checkpoint, CUDA) et relancer `--trial`.

Artefacts du run : `semantic_masks.json`, `semantic_masks_preview/`, `mask_consistency.json`, `mask_provenance.json` (commit Git, SHA-256 du checkpoint, environnement qualifié), `navigation_constraints.json`, et par panorama `segmentation/fused/<id>/{geometry_mask,appearance_mask,unknown_mask,labels}.png` à la résolution d’origine.

Règles :

- UNKNOWN (modèle, checkpoint, CUDA, environnement indisponibles, ou aucun vitrage détecté à l’essai) reste distinct d’un échec (FAIL/REJECTED). Aucun des deux ne produit de masque par défaut ni n’autorise l’entraînement.
- `geometry_mask` exclut les régions validées (≥ 2 faces concordantes, majorité) de vitrage, miroir, reflet et objets dynamiques, leurs trous et les candidats qui leur sont connexes, ainsi que les pixels sans inférence ; marge de 1°. Les candidats isolés non validés restent utilisables pour la texture SfM, sont marqués dans `unknown_mask` et bloqués pour la navigation (`POLICY['geometry_policy']`).
- `navigation_constraints.json` est consultatif (`advisory`) tant qu’aucun constructeur de navigation ne le consomme ; il ne valide aucun déplacement.
- Jointure ERP traitée comme circulaire (`seg-policy-2`) : composantes connexes et trous raccordés entre les colonnes 0 et W−1 ; le post-traitement doit être invariant par rotation horizontale. Un écart à la jointure n’échoue que s’il dépasse `max_seam_mismatch` à la fois en valeur brute et au-delà de la pire paire de colonnes intérieures voisines (16 de chaque côté) : un bord de région proche de la jointure passe, un masque coupé à la jointure reste FAIL. `mask_consistency.json` conserve la valeur observée, la référence et le seuil. Diagnostic en lecture seule d’un run existant : `python scripts/seam_diagnostic.py Output/runs/<id> <panorama>`.
- Seuils `POLICY` non calibrés.

Après une segmentation des 13 panoramas `ACCEPTED`, lancer une nouvelle expérience SfM masquée : `configs/salon-masked.json` (`"semantic_run": "salon-sam3-001"`) injecte les `geometry_mask` vérifiés dans `prepare`. La partition provisoire précédente n’est pas réutilisée (`split.json` : `not_proposed` jusqu’à l’algorithme AUTO-05).

```sh
.venv-sfm/bin/python -m theta_pipeline diagnostic --config configs/salon-masked.json --run-id salon-masked-001
```

## Reproduire l’audit

Utiliser Python avec Pillow ≥ 10.1 et NumPy, puis depuis la racine du projet :

```sh
python3 scripts/audit_capture.py --input Input --output Output/RD/audit
```

Dans l’environnement local utilisé pour cette étude :

```sh
'/Users/stani/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' scripts/audit_capture.py
```

Le script lit les images et écrit seulement le dossier d’audit. Il produit un manifeste JSON, une planche des panoramas et des projections perspectives destinées à l’inspection visuelle. Ces projections ne sont pas des caméras calibrées ni une reconstruction 3D. Aucun envoi des photos vers un service externe n’est nécessaire pour cet audit.

Environnement de l’audit exécuté : Pillow 12.3.0 et NumPy 2.3.5. Les 13 empreintes des originaux ont été revérifiées après la rédaction ; elles sont inchangées. La partition proposée dans le plan d’expériences couvre les 13 fichiers sans chevauchement.
