# Décision CTO — validation automatisée sans intervention par scène

Étude du 2 octobre 2026. Proposition à implémenter ; aucun modèle de segmentation ou de profondeur n’a été exécuté pour cette étude. Référence locale : `Output/runs/salon-cpu-008/spatial_review.md`.

## Décision

Supprimer l’annotation et l’approbation humaines du traitement courant. Les remplacer par une chaîne de contrôles reproductibles qui peut accepter, limiter, réessayer ou refuser automatiquement. Ne pas introduire d’état « attendre validation humaine ».

Cette exigence automatise la décision ; elle ne garantit pas que toute scène sera reconstruite correctement. Les zones non démontrées restent inconnues. Les échecs entraînent un repli panorama ou une demande de capture calculée. Les modèles de vision ne constituent pas à eux seuls une vérité terrain.

Le compte rendu actuel est trop catégorique sur la recapture : des masques absents sont d’abord une étape de traitement manquante. Les 88 observations de R0010009 et les 175 de R0010010 justifient des contrôles renforcés, pas leur suppression automatique. Un graphe de correspondances connecté n’est pas un graphe de déplacement libre.

## Technologies retenues et alternatives

| Besoin | Choix initial | Rôle et limite |
|---|---|---|
| Segmentation sans clic | SAM 3, branche image, avec vocabulaire prédéfini | Produit des candidats miroirs, vitrages, personnes, mobilier et sol. Sa segmentation par concepts évite les boîtes dessinées à la main ; le résultat doit être contrôlé. |
| Alternative de segmentation | Grounding DINO + SAM 2 via Grounded-SAM-2 | Solution de remplacement ou comparaison sur régions incertaines ; ne pas installer deux chaînes complètes d’emblée. |
| Cas verre difficile | GEM comme candidat spécialisé | À évaluer seulement si le témoin manque les vitrages ; disponibilité exacte des poids, droits et dépendances à qualifier avant intégration. |
| Poses | Rig COLMAP actuel | Refaire l’extraction après masquage ; vérifier stabilité et support spatial, en plus des résidus. |
| Géométrie dense de contrôle | DA3-Base avec poses connues | Hypothèse de profondeur à recouper entre centres ; ne pas confondre sa confiance avec une probabilité calibrée. |
| Décision | Règles Python versionnées | Produit des permissions distinctes pour entraînement, déplacement et publication. Aucun LLM ne signe une acceptation. |

SAM 3 fournit une segmentation guidée par concepts textuels. SAM 3.1 apporte notamment des évolutions vidéo ; notre lot de photographies n’impose pas d’utiliser un suivi vidéo. [Dépôt officiel SAM 3](https://github.com/facebookresearch/sam3), [note SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

Grounded-SAM-2 assemble détection textuelle et segmentation. C’est une alternative d’intégration, pas une preuve indépendante d’exactitude. [Dépôt officiel](https://github.com/IDEA-Research/Grounded-SAM-2).

Les travaux spécialisés sur le verre confirment l’intérêt de traiter ce matériau séparément. Les résultats publiés sur SAM historique ne doivent pas être extrapolés quantitativement à SAM 3. [GEM, dépôt auteur](https://github.com/isjinghao/GEM), [étude SAM et verre](https://arxiv.org/abs/2305.00278).

DA3 accepte des poses connues et expose profondeur et confiance ; DA3-Base est annoncé Apache 2.0, contrairement aux grands modèles et variantes Nested annoncés non commerciaux. Retenir Base pour la première intégration. [Modèles et licences](https://github.com/ByteDance-Seed/Depth-Anything-3), [API officielle](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md).

SAM 3 utilise une licence SAM spécifique : archiver le texte avec le poids choisi, sans le déclarer Apache par analogie avec SAM 2. [Licence officielle](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Figer commits, poids, SHA-256 et dépendances ; cette étude ne vaut pas qualification des performances ou compatibilités locales. Prévoir des environnements ML isolés et profiler sur GPU disponible, sans achat ni location automatique.

## 1. Générer les masques à partir des originaux

Créer `auto-mask`, indépendant de la reconstruction : lecture des ERP originaux, projections perspectives couvrant toute la sphère avec recouvrement, dont zénith/nadir. Employer des orientations supplémentaires pour les frontières de faces. Un seul agrandissement d’une image réduite ne récupère pas les détails : rééchantillonner les régions incertaines depuis l’original avec un champ de vue plus étroit.

Prompts courts versionnés : `mirror`, `window`, `glass door`, `glass partition`, `person`, `animal`, `tripod`, `floor`, `wall`, `sofa`, `chair`, `table`. Une classe meuble n’implique pas un objet mobile ; la mobilité exige des contradictions entre acquisitions. Segmenter la surface du miroir/vitrage, pas seulement ce qu’elle reflète ou laisse voir. Les cadres opaques peuvent rester utiles à la géométrie si séparables avec confiance.

Reprojeter les probabilités et masques dans les coordonnées ERP originales. Traiter périodicité horizontale, pôles, recouvrement et résolution angulaire. Mesurer l’accord entre deux projections différentes d’une même région ; elles ne comptent pas comme deux centres géométriques indépendants.

Sorties séparées :

- `geometry_valid` : exclut candidats réfléchissants/transparents trompeurs, personnes et incertitudes critiques ; ajoute les marges de rééchantillonnage.
- `rgb_valid` : exclut personnes, mouvements et corruption. Les miroirs restent évalués comme région d’apparence à risque ; si le moteur échoue sur eux, restreindre le rendu ou basculer au panorama plutôt que masquer le défaut dans le score global.
- `semantic_unknown` : zones sans couverture d’inférence, désaccords ou faible confiance. Aucun masque blanc de secours en cas d’erreur du modèle.

L’absence de détection ne prouve pas l’absence de miroir. Conserver les scores bruts, les versions et l’accord des projections. Combiner conservativement les candidats pour la géométrie ; mesurer la perte de support SfM pour détecter un sur-masquage.

## 2. Valider les poses automatiquement

Relancer SfM dans un nouvel essai avec les masques. Publier par panorama : observations 3D distinctes, nombre de centres contributeurs, répartition angulaire des pistes, résidus médian/p95, angles de triangulation et rôle dans le graphe.

Ajouter des tests de stabilité : plusieurs sous-échantillonnages déterministes de pistes et réoptimisations. Aligner les reconstructions par Sim(3) sur les centres communs, par composante, avant de comparer rotations et translations normalisées. Compléter si utile par la covariance BA en fixant la jauge ; une covariance indisponible est un contrôle inconnu, pas une incertitude nulle. [API officielle PyCOLMAP](https://colmap.github.io/pycolmap/pycolmap.html).

Sur R0010009 et R0010010, tester si le maintien ou l’exclusion de la station change fortement les poses voisines ou coupe le graphe. Une station faible mais stable peut rester ; une station instable est exclue du parcours ou déclenche une capture supplémentaire. Ne pas multiplier sans borne les variantes jusqu’à obtenir un score favorable.

## 3. Évaluer surface et espace navigable

Inférer DA3 sur vues perspectives avec poses SfM fixées. Aligner les profondeurs au repère arbitraire ; documenter profondeur axiale versus distance radiale. Vérifier profondeur reprojetée et visibilité entre plusieurs centres distincts, avec gestion des occultations. Les désaccords et surfaces masquées ne produisent pas de volume libre.

Construire une représentation à trois états : occupé, libre observé, inconnu. Ne jamais assimiler absence de points sparse ou faible opacité Gaussian à espace libre. Combiner classes sol/mur avec plans robustes et normales pour estimer une verticale ; si ambiguë, rester en repère de reconstruction sans prétendre identifier une hauteur physique.

Les chemins candidats relient des stations dont la visibilité et la géométrie dense se recoupent. Tester tout le volume balayé par la caméra virtuelle, avec marge liée à l’incertitude, pas uniquement les extrémités. Choisir une résolution de contrôle cohérente avec les voxels et l’incertitude. Les mesures restent relatives sans référence métrique fiable ; aucune garantie de passage humain n’est requise ni déduite pour cette visite virtuelle.

En cas d’inconnu sur un segment : interdire l’interpolation 3D de ce segment et utiliser une téléportation/fondu entre panoramas. Une navigation par hotspots reste entièrement automatisable même quand aucun déplacement continu n’est validé.

## 4. Décisions explicites, sans opérateur

`auto-gates` produit des résultats PASS/FAIL/UNKNOWN par contrôle, puis des permissions distinctes :

| Permission | Condition |
|---|---|
| `panorama_delivery` | Originaux exploitables ; liens entre stations présentés sans affirmer un passage libre. |
| `research_training` | Masques et poses suffisants selon la politique expérimentale ; partition figée et localisation réservée réussie. Ne nécessite pas déjà un sol entièrement validé. |
| `guided_3d_navigation` | Chaque segment autorisé passe le contrôle d’espace libre et, après entraînement, le contrôle du rendu. |
| `free_3d_navigation` | Autorisée uniquement dans le volume explicitement qualifié, jamais dans toute la pièce par extrapolation. |
| `product_delivery` | Politique préalablement calibrée, tests réservés, export, parcours et performances Web réussis. |

Les issues sont `accept`, `accept_restricted`, `retry_processing`, `request_recapture`, `reject`. UNKNOWN interdit seulement les permissions dépendantes du contrôle. Réessais bornés : une passe de segmentation plus fine et une variante SfM motivée au départ, avec plafond temps/mémoire. Une panne logicielle produit `retry_processing`/`reject`, pas une fausse demande de nouvelles photos.

Les seuils doivent être calibrés par logiciel sur jeux annotés existants et scènes synthétiques, avec séparation calibration/test. Mesurer omissions de régions à risque, faux accords, taux d’abstention et taux de rejet. Sans données de calibration représentatives, autoriser le mode expérimental et le repli panorama ; ne pas prétendre avoir mesuré le taux d’erreur produit. Aucun nouveau dessin manuel par scène n’est demandé.

Les indicateurs historiques 12/13 et 1,5 px restent des alertes de diagnostic. Ne pas transformer 88 observations ou une confiance modèle de 0,9 en règles universelles. Chaque règle porte unité, résolution, version, provenance et domaine de validité.

## 5. Partition et évaluation sans fuite

L’algorithme propose une partition spatiale déterministe sur le graphe, vérifie la couverture de l’entraînement et enregistre ses raisons. Geler automatiquement quand les contrôles préalables passent ; sinon conserver `provisional` et une cause calculée. Le gel précède l’évaluation des rendus.

Reconstruire ensuite uniquement avec les panoramas train ; localiser validation/test sur les points figés. Les masques réservés sont produits indépendamment, sans propagation de pixels/profondeurs test vers train. DA3 conjoint train+test ne doit pas alimenter la géométrie d’entraînement. Séparer explicitement diagnostic tout-lot et benchmark strict.

Une localisation réservée qui échoue entraîne l’échec de la qualification ; ne pas déplacer discrètement le panorama dans train pour améliorer le score. Les variantes et checkpoints utilisent validation ; le test final ne devient pas une boucle de réglage.

Après un entraînement court, automatiser rendus réservés, PSNR/SSIM/LPIPS par régions, fraction réellement évaluée, stabilité temporelle, trous/floaters candidats et concordance export. Aucun score sans référence ne prouve seul le photoréalisme. Le test de rendu appartient à la qualification finale ; il ne doit pas être exigé avant que le premier entraînement expérimental puisse démarrer.

## 6. Contrats et tickets développeurs

Commandes proposées, non encore implémentées : `auto-mask`, `validate-geometry`, `build-navigation`, `auto-gates`, `freeze-split`.

| Ticket | Livrable | Recette automatisée |
|---|---|---|
| AUTO-01 | Schéma des décisions, règles et permissions ; `gate_results.json` | UNKNOWN, erreur modèle, contrôle manquant : aucune acceptation implicite. |
| AUTO-02 | SAM 3 et projection ERP ; `semantic_masks.json` + cartes | Couture, pôles, masques vides, zones sans inférence et empreintes des poids. |
| AUTO-03 | SfM masqué et stabilité ; `geometry_validation.json` | Composantes indépendantes, station faible stable/instable, absence de faux voisins intra-centre. |
| AUTO-04 | Profondeur contrôlée et volume ; `navigation_graph.json` | Scène synthétique avec mur entre deux centres : segment refusé malgré extrémités valides ; inconnu ≠ libre. |
| AUTO-05 | Partition déterministe et gel conditionnel | Aucun panorama partagé, aucune donnée réservée dans les points train, échec de localisation explicite. |
| AUTO-06 | Qualification des rendus/export et repli Web | Un défaut ou contrôle inconnu désactive le segment concerné ; retour panorama disponible. |

Chaque artefact conserve les empreintes de ses entrées, ses raisons et les zones affectées. Le moteur de règles doit être indépendant des modèles afin de tester ses décisions avec des fixtures. Une demande de recapture produit `recapture_plan.json` : panorama de référence, direction angulaire annotée, type de manque et prises suggérées. Sans échelle, ne pas inventer une distance en mètres ; privilégier les zones accessibles déjà observées pour guider la capture.

Ordre immédiat : AUTO-01 et AUTO-02, puis nouvel SfM et AUTO-03. AUTO-04 conditionne le déplacement continu ; il ne bloque pas la visite par hotspots ni, si les autres conditions passent, un entraînement expérimental. Aucune nouvelle intervention manuelle n’est ajoutée au pipeline.
