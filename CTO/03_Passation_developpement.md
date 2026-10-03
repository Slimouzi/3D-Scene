# Passation développeurs — 1er octobre 2026

Développement arrêté à la demande du commanditaire. Ce document décrit l’état réel du code et l’ordre de reprise. La directive `CTO/02_Instructions_developpement.md` reste la référence produit ; ses tickets ne doivent pas être considérés comme terminés sur la seule présence de code.

## État livré

- Package `theta_pipeline/` : CLI `audit`, `prepare`, `sfm`, `report`, `diagnostic`.
- `storage.py` : identités d’essai, empreintes des sources/configuration/code/dépendances, verrou exclusif, écritures JSON atomiques, états d’étapes, vérification du cache et pic mémoire du processus.
- `geometry.py` : projections ERP/perspective, conventions COLMAP, attribution des caractéristiques aux vues du rig, rotations des six faces du cube. Les six faces d’entraînement ne sont pas encore générées par une commande.
- `stages.py` : audit, préparation des douze vues SfM par panorama, masques séparés, extraction, appariement exhaustif avec contraintes de rig, mapping, conversion des poses et rapport diagnostic.
- `configs/salon.json` : diagnostic sur les treize originaux ; ERP 4096×2048, faces 1024×1024, quatre threads CPU, limite de mapping 600 secondes. Cette limite ne plafonne pas l’ensemble de la commande.
- `requirements/sfm.lock.txt` : versions exactes des quatre paquets installés dans `.venv-sfm` ; environnement qualifié pour les imports et les tests, pas encore pour une reconstruction complète. Il ne s’agit pas d’un verrou multi-plateforme avec empreintes des distributions.
- Douze tests passent : axes, convention sphérique amont, couverture du cube, centres communs, référence du rig, masque de couture, attribution à pleine résolution, cache, changements d’entrées, erreurs et verrouillage.

Environnement réellement utilisé : CPython 3.12.8, macOS 15.3 arm64, PyCOLMAP 4.2.1 sans CUDA, NumPy 2.3.5, Pillow 12.3.0, OpenCV headless 4.13.0.92. L’exécutable système COLMAP 3.11.1 n’est pas utilisé par cette CLI.

## Dernière exécution et incident

`Output/runs/salon-cpu-001/` contient uniquement un audit réussi et un début de préparation. Le processus est terminé avec le code 139, sur un SIGSEGV natif pendant le calcul du masque d’attribution à résolution 1024. Journal : `Output/logs/salon-cpu-001.log`.

Le défaut a été reproduit avec un appel isolé à `ownership`. Les multiplications matricielles volumineuses ont été remplacées par `numpy.einsum(..., optimize=False)` ; le test de régression à 1024 passe. La pile native suggère le chemin de calcul Accelerate/NumPy, sans preuve définitive de la cause interne. **Le diagnostic complet n’a pas été relancé après cette correction.**

Le manifeste de cet ancien essai conserve `prepare: running` : un crash natif ne traverse pas le gestionnaire d’exceptions Python. Ne pas prendre cet état pour un processus actif ou une préparation réussie. Conserver l’essai et son journal comme preuve de l’incident.

Aucune pose, aucun nuage SfM validé, aucun Gaussian PLY et aucun lecteur Web n’ont été produits. Les tests ne constituent pas une validation des appels de reconstruction de bout en bout.

## Reprise immédiate

1. Lire le code et exécuter les tests. Ajouter un superviseur de processus qui enregistre un échec après signal natif ou interruption, sans publier de résultat réussi. Tester cette situation dans un sous-processus ; ne pas provoquer un crash du lanceur des tests.
2. Conserver `salon-cpu-001`. Utiliser un nouvel identifiant d’essai : toute modification du code, de la configuration ou des entrées invalide volontairement le cache actuel.
3. Lancer `audit`, puis `prepare`, et inspecter `prepare/masks_preview.jpg`, `views.json` et `rig.json`. Vérifier les 156 vues attendues et les douze orientations. Les masques sémantiques ne sont pas renseignés : les masques par défaut ne valent pas exclusion de l’opérateur, des miroirs ou des objets mobiles.
4. Fournir si nécessaire les masques ERP à la taille des originaux dans la configuration, puis créer un nouvel essai. Blanc/non nul = valide, noir = exclu ; clés par identifiant de panorama : `geometry` et `rgb`. Vérifier visuellement la projection et les marges d’exclusion autour des contours.
5. Exécuter `sfm` avec journal. Vérifier réellement les API PyCOLMAP, l’intégrité des artefacts, les appariements entre centres distincts et le maintien des intrinsèques/extrinsèques fixes. Corriger les erreurs d’intégration avant toute conclusion sur les captures.
6. Produire et examiner `poses.json`, `quality.json`, `report.md`, ainsi qu’une visualisation des centres et du nuage sparse. Un bon taux de recalage ne valide pas seul le parcours. L’état `requires_spatial_review` est intentionnel : couverture, reflets et zone navigable doivent encore être examinés.

Commandes depuis la racine du projet, avec l’environnement déjà présent :

```sh
.venv-sfm/bin/python -m unittest discover -s tests -v
.venv-sfm/bin/python -m theta_pipeline audit --config configs/salon.json --run-id salon-cpu-002
.venv-sfm/bin/python -m theta_pipeline prepare --config configs/salon.json --run-id salon-cpu-002
.venv-sfm/bin/python -m theta_pipeline sfm --config configs/salon.json --run-id salon-cpu-002
.venv-sfm/bin/python -m theta_pipeline report --config configs/salon.json --run-id salon-cpu-002
```

Adapter l’identifiant si l’essai existe déjà ou si le code change. La commande `diagnostic` enchaîne ces étapes, mais commencer par les commandes séparées facilite la qualification.

## Points techniques à traiter avant de déclarer le socle robuste

- Ajouter la supervision des sorties natives et une récupération explicite des étapes interrompues ; ne jamais supprimer automatiquement une base existante.
- Vérifier l’intégrité de la base entre extraction et appariement : le cache d’extraction ne conserve actuellement qu’un résumé immuable, alors que la base est mutable pendant l’appariement.
- Tester les masques fournis par l’utilisateur, leur érosion conservatrice et leur projection sur données synthétiques. Le test actuel de couture ne couvre pas toute cette chaîne.
- Ajouter un petit test d’intégration des étapes et des conversions de poses ; les tests actuels portent surtout sur les conventions mathématiques et le stockage.
- Confirmer le diagnostic sur le jeu complet avant de qualifier les dépendances comme combinaison opérationnelle.
- Les pics mémoire sont ceux du processus depuis son démarrage, pas des maxima indépendants par étape. Les durées sont mesurées par étape.
- Mettre à jour le README, qui décrit encore principalement l’étude R&D.

## Suite après diagnostic exploitable

Suivre DEV-05 à DEV-10 de la directive CTO : partition spatiale par panorama, reconstruction sur entraînement seulement, localisation des vues réservées sur géométrie figée, six faces de rendu depuis les originaux, adaptateur neutre, témoin gsplat à poses fixes, reprise complète, export Gaussian PLY puis lecteur Three.js/Spark.

`build-dataset`, `train`, `evaluate` et `export` ne sont pas implémentés et ne sont pas exposés par la CLI actuelle. L’environnement CUDA, le verrou d’entraînement et le lecteur Web restent à développer. Choisir le calcul après profilage ; aucune ressource payante n’a été provisionnée.

Premier livrable demandé aux développeurs : **un diagnostic des poses réellement exécuté, avec visualisation, limites de couverture et décision argumentée sur la suite**, avant d’engager l’entraînement Gaussian Splatting.
