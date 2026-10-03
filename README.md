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
- La partition panorama est produite comme `split.json` provisoire; la navigation, le sol et les masques sémantiques restent à revoir.
- Entraînement Gaussian Splatting, densification MVS et lecteur final : non exécutés.
- Les originaux restent dans `Input/`. Le dossier `outpout` mentionné initialement n’était pas présent ; les livrables sont dans `Output/RD/`.

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
