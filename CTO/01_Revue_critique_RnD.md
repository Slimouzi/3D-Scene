# Revue CTO des orientations R&D

**1er octobre 2026 — décision d’ingénierie, avant développement.**

Documents examinés : les trois notes du dossier `R&D/`, identiques aux exemplaires de `Output/RD/`, le manifeste des captures et le script d’audit existant. La revue a été complétée par une lecture du code amont et des documentations citées ci-dessous. Aucun benchmark de reconstruction n’a été exécuté pour cette revue.

## Décision

**Je retiens le Gaussian Splatting pour une visite Web, mais je conditionne le choix définitif du moteur à un premier résultat de bout en bout.** Le besoin immédiat est de vérifier que nos captures permettent une navigation convaincante et que le rendu exporté reste fidèle. Les développements de profondeur apprise, de compensation avancée et de nouvelle acquisition constituent des extensions ciblées.

La R&D a bien identifié les risques de capture et les familles de méthodes. Son dossier laisse cependant trop de choix ouverts pour une équipe de développement et sous-estime plusieurs raccordements entre outils. Les quatre à six semaines restent une enveloppe indicative de qualification ; elles ne doivent pas devenir un délai avant la première preuve.

## 1. Les treize captures suffisent-elles ?

**Pas démontré.** Des changements de position visibles et des JPEG lisibles ne garantissent ni un recalage cohérent, ni les observations nécessaires aux déplacements entre canapés, autour de la table et près des baies. Les faces dérivées ne multiplient pas les centres optiques disponibles.

**Décision :** diagnostic des poses et de la couverture en premier, avec un parcours souhaité explicitement dessiné. Le résultat attendu est une carte des positions, des zones observées et des zones à reprendre. Le seuil de 12/13 poses est un indicateur ; une treizième vue qui est la seule à couvrir un passage peut être déterminante. À l’inverse, écarter une vue redondante défectueuse ne condamne pas toute la scène.

**Conséquence produit :** le MVP propose une navigation dans une zone validée. Une promenade libre dans toute la pièce ne devient une promesse qu’après contrôle des vues intermédiaires. Une visite guidée ou des panoramas de repli constituent un périmètre de livraison possible, à nommer clairement.

## 2. COLMAP → gsplat est-il déjà un pipeline prêt à assembler ?

**Non ; c’est un socle à adapter.** La lecture du code apporte les corrections suivantes :

| Constat vérifié dans les sources | Décision CTO |
|---|---|
| Le mode panoramique COLMAP 4.2.1 emploie 12 vues à 90°, avec inclinaisons −35°, 0°, +35°. Leur taille dépend de celle du panorama. | Distinguer les images servant aux poses des images servant au rendu ; expliciter résolution et couverture. |
| Le traitement produit des masques de répartition des points caractéristiques ; ce ne sont pas les masques des personnes, supports ou reflets. | Composer les deux avant extraction des caractéristiques. |
| Le rig a pour référence une caméra virtuelle ; la pose de cette référence n’est pas directement l’orientation du panorama. | Tester la conversion rig → panorama → faces d’entraînement. |
| Le workflow supprime une base de correspondances déjà présente dans sa destination. | Chaque reconstruction utilise un répertoire neuf ; la reprise relève de notre orchestrateur. |

Source : [code panoramique COLMAP au tag 4.2.1](https://github.com/colmap/colmap/blob/4.2.1/python/pycolmap/panorama.py).

**Notre analyse géométrique :** avec les vues ci-dessus, les directions exactement au zénith et au nadir ne sont pas couvertes. Pour entraîner une représentation de toute la sphère, produire six faces de cube, plafond et sol compris, à partir des poses estimées. Les masques de validité signaleront les zones inutilisables. Ce choix évite aussi de reprendre tous les recouvrements des images SfM dans la fonction de coût.

L’exemple gsplat 1.5.3 sélectionne ses images réservées selon leur index. Son lecteur utilise `SceneManager` et des masques associés à la rectification, sans fournir notre contrat par panorama. Ses dépendances désignent un autre dépôt `pycolmap`, distinct des bindings COLMAP modernes. **Décision :** environnements séparés et petit adaptateur de données explicite ; ne pas installer tous les fichiers de dépendances amont dans le même environnement. [Lecteur de données](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/datasets/colmap.py), [dépendances de l’exemple](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/requirements.txt).

## 3. Les scripts d’exemple respectent-ils déjà notre contrat d’entraînement ?

**Pas entièrement.** Dans l’exemple inspecté, masquer met à zéro des couleurs rendues avant le calcul des pertes ; ce n’est pas une réduction explicite sur les seuls pixels valides. La pose peut être optimisée par image. La sauvegarde montrée n’enregistre pas tout l’état nécessaire à une reprise exacte. Certaines apparences apprises sont simplifiées à l’export PLY. [Entraîneur gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py).

**Décision :** utiliser les primitives d’entraînement existantes, avec adaptations limitées et testées des entrées, masques, sauvegardes et exports. Geler les poses et conserver des harmoniques sphériques standard pour le premier témoin. Réécrire un moteur de rendu serait prématuré ; exécuter l’exemple sans vérifier ses hypothèses serait insuffisant.

## 4. Faut-il imposer NVIDIA/cloud dès maintenant ?

**NVIDIA est cohérent pour gsplat ; le cloud reste un mode d’exécution à choisir.** La note déduit trop vite l’infrastructure du moteur envisagé. Brush propose un entraînement natif macOS, une CLI et des entrées COLMAP/Nerfstudio. Cela justifie un essai court sur le Mac existant, sans prouver sa qualité ni sa rapidité sur nos données. [Documentation officielle Brush](https://github.com/ArthurBrussee/brush).

**Décision :** conserver gsplat comme référence de développement pour ses possibilités d’instrumentation. Faire un essai Brush borné à une demi-journée d’intégration après obtention des poses, comme comparaison d’effort et solution locale éventuelle. Si une machine NVIDIA est disponible, exécuter le témoin instrumenté ; sinon la première vérification visuelle locale peut avancer. La branche CUDA vise initialement 24 Go de VRAM. Aucun achat ni contrat cloud ne découle de cette note.

Comparer le temps humain, le temps machine, la mémoire, le résultat exporté et la reproductibilité. Une démo locale sans le même protocole d’évaluation ne devient pas automatiquement un benchmark équivalent.

## 5. Profondeur apprise et PPISP : indispensables au MVP ?

**Non.** Les écarts d’exposition des captures constituent une raison de tester une correction ; ils ne démontrent pas que PPISP sera nécessaire. De même, une profondeur plausible peut améliorer les murs ou introduire une géométrie erronée derrière une vitre.

**Décision :** établir le témoin, classer ses défauts, puis activer une seule amélioration à la fois. Profondeur si la géométrie est faible malgré de bonnes poses ; compensation photométrique si les variations d’apparence dominent. Tout composant devra améliorer le résultat final dans le navigateur et rester compatible avec l’usage envisagé. Le traitement RAW/fisheye, FullCircle et la complétion générative sortent du premier lot de développement.

## 6. Les métriques proposées suffisent-elles à décider ?

**Le protocole est bien orienté ; ses seuils ne sont pas encore étalonnés.** Deux panoramas de test ne suffisent pas à soutenir un gain généralisable de 10 % de LPIPS. Les graines supplémentaires mesurent la variabilité de l’optimisation, pas la diversité des logements. Une cadence moyenne peut cacher des à-coups ; un fichier compressé peut consommer beaucoup de mémoire après chargement.

**Décision :** figer une partition par panorama après l’audit géométrique, régler les variantes sur la validation et réserver le test à la comparaison finale. Rapporter chaque vue et chaque zone, les défauts sur un trajet imposé, les temps de frame et la mémoire. Les 60 fps desktop, 30 fps mobile et moins de 5 s de chargement restent des cibles mesurées sur des appareils nommés, et non des propriétés promises par le choix de bibliothèque.

Le protocole strict reconstruit les points d’entraînement sans les panoramas réservés. Un recalage utilisant toutes les prises peut servir au diagnostic, avec un identifiant d’essai distinct. Si les vues réservées ne se localisent pas correctement, le benchmark échoue ; aucune image d’entraînement ne vient les remplacer silencieusement.

## 7. SPZ et niveaux de détail désignent-ils un seul livrable ?

**Non.** La documentation actuelle de Spark 2.0 distingue les fichiers de splats du format RAD préparé pour les niveaux de détail et le chargement paginé ; elle indique que son ancien SPZ étendu pour les niveaux de détail est déprécié. [Documentation Spark](https://sparkjs.dev/docs/lod-getting-started/).

**Décision :** PLY master pour l’interopérabilité ; un seul lecteur Web, Three.js + Spark ; SPZ possible pour un modèle compact simple ; RAD pour la branche paginée si les mesures l’exigent. Le convertisseur et le lecteur doivent être verrouillés ensemble. Reporter PlayCanvas à une alternative motivée, afin de ne pas maintenir deux intégrations dès le départ.

## Arbitrage de livraison

Le premier jalon est un **diagnostic de reconstructibilité sous trois jours ouvrés visés**. Le second est une **première chaîne jusqu’au navigateur sous cinq à dix jours ouvrés visés**, selon disponibilité du calcul et qualité des poses. Un diagnostic d’échec étayé est un résultat utile ; une scène jolie depuis ses seules positions d’entraînement ne valide pas la visite.

Ces délais sont des objectifs de pilotage, pas des performances observées. Les instructions opérationnelles sont dans [la directive développeurs](</Users/stani/code/3D Scene/CTO/02_Instructions_developpement.md>). Elles tranchent les options du dossier R&D pour le premier lot, sans modifier les constats historiques de ce dossier.
