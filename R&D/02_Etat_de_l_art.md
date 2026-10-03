# État de l’art et choix technologiques

**Veille arrêtée au 1er octobre 2026.** Sources primaires : articles, pages des auteurs, dépôts officiels et documentations. Disponibilité documentaire vérifiée ; installations et performances sur nos captures non testées. Les appréciations d’adéquation ci-dessous sont nos orientations R&D.

## 1. Ce que signifie reconstruire à partir de photos 360

Un panorama donne toutes les directions depuis une position. La profondeur multi-vues vient des déplacements entre positions et des correspondances entre observations. La rotation seule, ou le découpage d’une sphère en faces, ne fournit pas une base de triangulation supplémentaire.

Le Gaussian Splatting représente l’apparence par des primitives 3D orientées, avec couleur, opacité et extension spatiale. Un nuage de points XYZ/RGB ou un maillage `.ply` n’est donc pas automatiquement un fichier de splats. La sortie recherchée est une représentation adaptée au rendu de nouvelles vues ; elle n’implique ni maillage fermé, ni exactitude métrique, ni surfaces invisibles correctement reconstruites.

Avec la Theta Z1, le panorama assemblé approxime une caméra à centre unique alors que deux objectifs distincts ont formé les images. Les objets proches et les raccords peuvent contredire cette approximation. Seam360GS traite explicitement l’écart entre centres optiques et les distorsions dans la reconstruction. Cette observation justifie de distinguer un pipeline pour JPEG assemblés d’un pipeline pour données fisheye brutes. [Seam360GS, ICCV 2025](https://arxiv.org/abs/2508.20080).

## 2. Socle de production envisagé

| Composant | État vérifié | Choix R&D |
|---|---|---|
| COLMAP / PyCOLMAP | Documentation officielle des rigs panoramiques ; version 4.2.1 publiée le 29 septembre 2026. Le modèle sphérique apparaît dans les notes 4.1.0 du 26 juin. | Version 4.2.1 candidate à figer et qualifier dans un environnement isolé. Comparer rig perspective et caméra sphérique native. |
| gsplat | Bibliothèque CUDA, exemple d’entraînement depuis COLMAP, licence Apache-2.0 affichée. Le README sépare le tag 1.5.3 des évolutions 1.6 sur `main`, annoncées comme non encore sur PyPI. | Témoin sur tag stable qualifié ; expériences récentes dans une branche avec commit exact. Ne pas confondre documentation `main` et paquet installé. |
| PPISP | Méthode CVPR 2026 de compensation des variations photométriques, avec contrôleur pour de nouvelles vues. | Expérience ciblée sur nos variations d’exposition ; examiner aussi la stabilité temporelle et l’export. |
| Spark / Three.js | Lecteur de splats, formats multiples et fonctions de niveaux de détail documentés. | Choix initial pour une visite intégrée à une application Web personnalisée. |
| PlayCanvas / SuperSplat | Édition, conversion et diffusion de formats SOG, dont variantes progressives. | Outil de QA et alternative au lecteur Spark si le temps de développement devient prioritaire. |

Sources : [COLMAP, versions](https://github.com/colmap/colmap/releases), [COLMAP, rigs](https://colmap.github.io/rigs.html), [gsplat](https://github.com/nerfstudio-project/gsplat), [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/), [Spark](https://github.com/sparkjsdev/spark), [SuperSplat, formats](https://developer.playcanvas.com/user-manual/supersplat/editor/import-export/).

**Point d’attention sur COLMAP :** les auteurs décrivent le traitement sphérique natif comme généralement plus rapide, mais moins précis que la conversion perspective. Il doit être comparé, sans supposer qu’une projection native gagne systématiquement. La documentation courante porte aussi une version de développement ; utiliser les scripts correspondant au tag installé. [Notes 4.1.0 et versions suivantes](https://github.com/colmap/colmap/releases), [script au tag 4.2.1](https://github.com/colmap/colmap/blob/4.2.1/python/examples/panorama_sfm.py).

## 3. Méthodes panoramiques spécialisées

| Méthode et date | Apport documenté / disponibilité | Adéquation aux 13 JPEG |
|---|---|---|
| **360-GS**, prépublication 2024 | Utilise la structure de la pièce pour guider le splatting panoramique, en particulier sur les zones planes et peu texturées. | Inspiration pour une régularisation douce des murs/sols ; ne pas imposer une pièce orthogonale sans vérification. [Article](https://arxiv.org/abs/2402.00763). |
| **ODGS**, NeurIPS 2024 ; **OmniGS**, WACV 2025 | Adaptent le rendu Gaussian aux observations omnidirectionnelles. Dépôts officiels accessibles. | Références pour mesurer l’effet d’un entraînement panoramique. OmniGS documente un problème dans son chemin perspective : contrôler particulièrement l’export vers notre lecteur. Une adaptation du rasterizer ne résout pas les mauvaises poses. [ODGS](https://github.com/esw0116/ODGS), [OmniGS](https://github.com/liquorleaf/OmniGS). |
| **Splatter-360**, CVPR 2025 | Reconstruction généralisable depuis panoramas à large base, avec correspondances sphériques ; modèles annoncés disponibles. | Challenger avec poses exploitables ; ne pas transposer directement les résultats HM3D/Replica aux JPEG Theta réels. [Article](https://arxiv.org/abs/2412.06250), [projet](https://3d-aigc.github.io/Splatter-360/). |
| **PanSplat**, CVPR 2025 | Deux panoramas d’entrée et représentation gaussienne hiérarchique pour synthèse 4K ; code de test publié. | Intéressant pour un aperçu rapide à partir d’une paire. Assembler plusieurs prédictions par paires en une scène unique demande une étape supplémentaire. [Dépôt](https://github.com/chengzhag/PanSplat). |
| **PanoSplatt3R**, ICCV 2025 | Reconstruction depuis panoramas sans poses, tirant parti du préentraînement perspective ; code et poids référencés. | Challenger prioritaire si le recalage classique échoue. Inférence sur poids existants ; aucun réentraînement de fondation prévu. Le dépôt indique ≥ 48 Go pour sa procédure d’entraînement, pas comme exigence universelle d’inférence. [Dépôt](https://github.com/zhichu99/PanoSplatt3R). |
| **Seam360GS**, ICCV 2025 | Optimisation conjointe de la calibration et des gaussiennes pour les panoramas imparfaitement assemblés. | Très pertinent pour les raccords Theta. Article consulté ; dépôt officiel exploitable non identifié durant cette veille. Risque d’intégration supérieur au socle. [Article](https://arxiv.org/abs/2508.20080). |
| **PFGS360**, mars 2026, CVPR 2026 selon le dépôt | Optimise poses et gaussiennes depuis vidéos panoramiques, avec a priori de profondeur cohérents ; code, rasterizer et dépendance UniK3D documentés. | Essai conditionnel : nos 13 photos espacées ne sont pas une vidéo dense. Audit des licences des dépendances nécessaire pour la trajectoire produit. [Article](https://arxiv.org/abs/2603.23324), [implémentation](https://github.com/zcq15/PFGS360). |

**Choix :** ne pas lancer toutes ces méthodes. Établir le témoin, identifier son défaut dominant, puis sélectionner un challenger capable de le traiter. Aucun score numérique publié sur des jeux différents n’est utilisé pour créer un classement commun artificiel.

## 4. Géométrie apprise : accélérateur, avec confiance mesurée

**Depth Anything 3, novembre 2025**, prédit profondeur et poses, avec possibilité de conditionner la profondeur sur les caméras connues. L’API fournit aussi une tête Gaussian pour les variantes Giant/Nested. Pour la branche perspective, utiliser d’abord un modèle compatible avec le futur usage produit, puis aligner et filtrer ses profondeurs par les observations multi-vues. Les petites variantes peuvent contribuer à la géométrie sans produire directement des splats. [Dépôt et cartes des modèles](https://github.com/ByteDance-Seed/Depth-Anything-3), [API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md).

**UniK3D, CVPR 2025**, accepte notamment une caméra sphérique et peut prédire des points 3D et rayons depuis une image. Il constitue un candidat technique pour des profondeurs ERP, mais sa licence annoncée est CC BY-NC 4.0. Ne pas en faire une dépendance obligatoire du produit. [Dépôt officiel](https://github.com/lpiccinelli-eth/UniK3D).

Notre proposition : les profondeurs apprises constituent un **a priori pondéré**, rejeté sur les reflets, vitres, silhouettes incertaines et zones incohérentes entre vues. Ne pas fusionner aveuglément treize nuages monoculaires avec treize échelles différentes. Préserver une variante sans a priori pour mesurer si le réseau corrige effectivement la reconstruction.

**SHARP, décembre 2025**, permet une synthèse locale rapide à partir d’une seule image et une prédiction sur MPS. Cela n’établit pas la cohérence d’une pièce navigable construite en fusionnant des faces indépendantes. En outre, la licence des poids limite l’usage à la recherche non commerciale et exclut explicitement le développement produit. Il est écarté de la branche produit proposée. [Dépôt Apple](https://github.com/apple-aiml-research/ml-sharp), [licence du modèle](https://github.com/apple-aiml-research/ml-sharp/blob/main/LICENSE_MODEL).

## 5. Frontière 2026 et deuxième génération de capture

**FullCircle, 23 mars 2026**, est la piste la plus directement pertinente pour une future acquisition 360 brute : reconstruction à partir des deux fisheye, masquage de l’opérateur et calibration COLMAP. Son implémentation est disponible, mais attend des données différentes des JPEG ERP présents. Notre inférence est qu’éviter l’assemblage préalable pourrait réduire certaines incohérences géométriques ; cela devra être testé avec une calibration propre à la Theta Z1. Un JPEG assemblé ne permet pas de récupérer les observations brutes perdues. [Article](https://arxiv.org/abs/2603.22572), [dépôt](https://github.com/theialab/fullcircle).

**3DGUT, CVPR 2025**, prend en charge des caméras déformantes dans une formulation de rasterisation. **3DGRT** utilise le lancer de rayons et possède d’autres exigences matérielles. Le dépôt 3DGRUT les distingue et renvoie vers gsplat pour un moteur modulaire. Ne pas supposer qu’un export PLY de ces variantes reproduira exactement leur rendu dans un lecteur 3DGS classique : vérifier le noyau des primitives, l’opacité et les paramètres d’apparence, ou prévoir une conversion/distillation mesurée. [Implémentations NVIDIA](https://github.com/nv-tlabs/3dgrut).

**PanoPlane, 13 mai 2026**, complète des régions non observées par génération panoramique guidée par des plans. Son intérêt est la réduction des trous ; sa limite pour nous est la fidélité des zones inventées. Article consulté, reproductibilité locale non établie. [Article](https://arxiv.org/abs/2605.14135).

**Spherical-GOF, 9 mars 2026**, vise la cohérence géométrique panoramique ; sa page annonce une publication du code. **PanoGS-SLAM, 15 septembre 2026**, vise le suivi et la cartographie continus ; son résumé annonce un code à venir. Ces deux travaux restent en veille, le second répondant davantage à une future capture vidéo qu’au traitement hors ligne des 13 photos. [Spherical-GOF](https://arxiv.org/abs/2603.08503), [PanoGS-SLAM](https://arxiv.org/abs/2609.17387).

## 6. Licences, maintenance et export : décisions d’ingénierie

| Élément | Constat dans les sources consultées | Conséquence |
|---|---|---|
| gsplat | Apache-2.0 affichée | Bon candidat de socle ; inventorier aussi les dépendances et les commits. |
| DA3 | Code Apache-2.0 ; Base/Small/Metric/Mono indiqués Apache-2.0 ; Large/Giant/Nested indiqués CC BY-NC 4.0 | L’export direct GS des Giant ne doit pas devenir implicitement le chemin produit. |
| UniK3D | CC BY-NC 4.0 | Usage commercial non présumé autorisé, y compris lorsqu’il arrive indirectement via un autre pipeline. |
| SHARP | Conditions distinctes pour code et poids ; modèle limité à la recherche non commerciale | Exclure la dépendance produit sans droits supplémentaires adaptés. |
| FullCircle | Fichier LICENSE racine Apache-2.0 | Vérifier séparément sous-modules, réseaux de masquage et ressources associés. |
| PanSplat / PanoSplatt3R | Code et modèles référencés ; dépendances de recherche | La présence d’un dépôt public ne suffit pas à qualifier toute la chaîne de licences. |
| SPZ | Bibliothèque MIT ; convention d’axes et conversion documentées | Bon format candidat, sous réserve de compatibilité exacte avec la version du lecteur. |

Sources de contrôle : [gsplat](https://github.com/nerfstudio-project/gsplat), [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3#model-cards), [UniK3D](https://github.com/lpiccinelli-eth/UniK3D#license), [SHARP](https://github.com/apple-aiml-research/ml-sharp/blob/main/LICENSE_MODEL), [FullCircle](https://github.com/theialab/fullcircle/blob/main/LICENSE), [SPZ](https://github.com/nianticlabs/spz).

Ces constats servent au choix des dépendances ; la qualification juridique complète est une étape du passage produit. Un usage interne nommé « R&D » n’autorise pas automatiquement tous les poids non commerciaux.

## 7. Arbitrage final

**Lot actuel :** rig COLMAP, optimisation 3DGS par scène, comparaison de profondeur avec confiance et de compensation d’apparence ; export standard vérifié dans le navigateur. **Capture future :** conserver RAW/fisheye et comparer une branche FullCircle/3DGUT à qualité finale et effort opérateur égaux. **Veille :** complétion générative, géométrie sphérique avancée et SLAM panoramique.

Le facteur limitant attendu est la combinaison couverture–poses–apparence, avant le nombre de gaussiennes. Cette hypothèse sera validée ou infirmée par le protocole expérimental associé.
