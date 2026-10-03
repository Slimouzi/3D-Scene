# Directive CTO — premier pipeline Theta Z1 vers visite Web

**Version 1 — 1er octobre 2026. Statut : instructions à implémenter.**  
Fondement : [revue critique CTO](</Users/stani/code/3D Scene/CTO/01_Revue_critique_RnD.md>). Cette directive tranche le périmètre initial ; le dossier R&D reste la référence de veille. Les tickets ci-dessous ne sont pas encore exécutés.

## 1. Livrable attendu

À partir des treize panoramas du salon, produire un traitement relançable qui restitue : les poses vérifiées, un fichier Gaussian PLY, une visite dans le navigateur et un rapport de qualité. En cas de couverture insuffisante, produire un diagnostic indiquant quelles prises complémentaires sont nécessaires.

Le MVP concerne une scène statique et une zone de navigation contrôlée. La géométrie est en unités de reconstruction tant qu’aucune mesure externe ne fixe l’échelle. La visite doit montrer clairement son état de chargement, proposer un retour au point de départ et offrir un panorama de repli en cas d’échec du rendu 3D.

La première livraison est un dossier local consultable, avec une procédure de lancement. Comptes utilisateurs, téléversement public, facturation, traitement de plusieurs clients, mesures métriques, édition de mobilier, maillage détaillé et capture RAW ne font pas partie de ce lot.

## 2. Choix fixés pour commencer

| Élément | Instruction |
|---|---|
| Orchestration | CLI Python et fichiers d’artefacts versionnés. Un processus par étape suffit. |
| Poses | Qualifier COLMAP/PyCOLMAP **4.2.1** dans un environnement isolé. Rig perspective, intrinsèques et extrinsèques relatives fixes. Le COLMAP 3.11.1 installé sur le Mac ne constitue pas cet environnement. |
| Appariements | Exhaustifs entre panoramas pour ce petit lot, avec rejet des paires d’un même centre. Mapper incrémental au départ. |
| Résolution SfM | Copie de travail ERP 4096 × 2048, donnant des vues à 1024 × 1024 dans le plan officiel à 90°. Conserver l’original intact. |
| Images d’entraînement | Six faces de cube à 90°, 1024 × 1024, produites directement depuis les originaux et les poses panoramiques validées. Inclure zénith et nadir. |
| Entraînement de référence | **gsplat 1.5.3** comme point de départ à qualifier, avec adaptateur de données et pertes masquées explicites. CUDA dans un environnement distinct. |
| Réglage du témoin | Poses fixes ; initialisation SfM ; couleurs/harmoniques sphériques standard ; stratégie de densification par défaut. Profondeur apprise, correction photométrique apprise et optimisation de poses désactivées. |
| Comparaison locale | Essai Brush natif, limité à une demi-journée d’intégration, après les poses. Servir de vérification visuelle locale ou de référence d’effort ; déclarer toute différence de protocole. |
| Web | TypeScript, Three.js et Spark. Figer la version exacte après essai d’import/export. Un seul lecteur à maintenir. |
| Formats | PLY master ; SPZ simple si utile ; RAD avec niveaux de détail uniquement lorsque les mesures de chargement ou de rendu le justifient. |
| Calcul | CPU pour les étapes compatibles ; machine NVIDIA 24 Go comme capacité initiale pour le témoin CUDA. Choisir local/cloud selon disponibilité et profilage, sans achat anticipé. |

Les versions sont des candidats d’intégration issus de la revue, pas un ensemble déjà testé. Le ticket DEV-01 doit verrouiller une combinaison fonctionnelle Python/PyTorch/CUDA/compilateur/NumPy, avec une image ou un environnement reproductible. Toute substitution de version est accompagnée du problème constaté et du test qui la valide. Ne pas travailler sur une branche distante mouvante sans conserver son commit.

## 3. Organisation des travaux

Un responsable vision/3D porte poses, adaptation du jeu de données et entraînement. Un responsable Web porte lecteur et mesure du rendu. Le responsable technique porte environnements, intégration et recette ; ces rôles peuvent être tenus par deux personnes. Ils désignent un propriétaire nominatif par ticket au lancement.

La branche Web peut avancer avec une petite scène de test synthétique clairement étiquetée. La validation du salon attend son véritable export. L’équipe commence par DEV-01 à DEV-04 ; DEV-05 et le lecteur avancent dès que leurs contrats sont définis.

## 4. Contrats entre étapes

Utiliser un dossier neuf `Output/runs/<run_id>/`. Tous les chemins enregistrés dans les artefacts sont relatifs à ce dossier ou à une racine d’entrée déclarée ; aucun chemin spécifique au Mac du développeur dans les formats d’échange.

| Artefact | Contenu obligatoire |
|---|---|
| `run.json` | Version du schéma, identifiant d’essai, type `diagnostic`/`benchmark`/`delivery`, empreintes des entrées/configuration, versions, état de chaque étape, durée, erreurs et mémoire mesurée. |
| `capture.json` | Pour chaque original : identifiant stable, chemin, SHA-256, dimensions, métadonnées utiles. S’appuyer sur l’audit existant. |
| `split.json` | Identifiants des panoramas entraînement/validation/test, protocole de poses, date de gel et justification spatiale. |
| `views.json` | Pour chaque vue : panorama parent, rôle SfM ou entraînement, chemin image/masques, dimensions, intrinsèques, transformation face-vers-panorama ou son inverse explicitement nommé. |
| `poses.json` | Pose monde-vers-panorama, statut de recalage, composante, qualité ; convention d’axes et unités déclarées. |
| `dataset.json` + `points.npz` | Vues et partitions explicitement développées ; caméras perspective prêtes pour le moteur ; points initiaux provenant exclusivement des entrées autorisées. |
| `quality.json` | Statut de chaque contrôle, métriques par vue, couverture et zones invalides ; valeur absente avec sa cause lorsque non mesurable. |
| `scene.json` | Fichiers exportés et empreintes, repère, échelle, pose de départ, parcours autorisé, profil de qualité et panorama de repli. |

Conventions internes : matrices homogènes 4 × 4, vecteurs colonnes, noms `T_destination_from_source`, caméra x droite/y bas/z avant. Stocker les poses en float64 pendant les conversions ; passer en float32 là où le moteur le nécessite. La transformation entre repère de reconstruction et repère Web est unique, explicite et appliquée de façon cohérente aux caméras et à la scène. Ne pas supposer que le monde COLMAP a déjà son axe vertical aligné.

Écrire les résultats temporaires avant de marquer une étape terminée. Une relance réutilise seulement les artefacts dont les empreintes sont compatibles ; elle crée un nouvel essai pour une modification de configuration. Une étape échouée retourne un code d’erreur et un diagnostic, sans créer un fichier présenté comme une scène réussie.

La CLI à implémenter expose les actions `audit`, `prepare`, `sfm`, `build-dataset`, `train`, `evaluate`, `export` et `report`. Chaque action accepte une configuration et un identifiant d’essai. Ces noms définissent une interface à développer ; ils ne désignent pas des commandes déjà disponibles.

## 5. Backlog priorisé

Les charges sont des estimations en jours d’ingénierie, à réviser après le premier diagnostic. Les dépendances expriment un ordre technique, sans exiger une réunion d’approbation à chaque étape.

| Ticket | Priorité / rôle | Travail et dépendance | Livrable et acceptation | Charge |
|---|---|---|---|---|
| **DEV-01** | P0 / responsable technique | Isoler SfM, entraînement et Web. Vérifier imports, versions et capacité de calcul. | Verrous de dépendances et test court d’exécution sur chaque environnement ; collision entre paquets `pycolmap` évitée. | 0,5–1 j |
| **DEV-02** | P0 / vision | Réutiliser l’audit, créer les identifiants et préparer les masques. | Les 13 originaux sont traçables et inchangés ; masques visualisables avec leur image ; rapport des exclusions. | 0,5–1 j |
| **DEV-03** | P0 / vision | Adapter les projections SfM et composer les masques avant extraction. Dépend de 01–02. | Rig documenté, vues à dimensions attendues, contrôles de rayons/axes ; aucun centre artificiellement déplacé. | 1 j |
| **DEV-04** | P0 / vision | Recalage du lot complet pour diagnostic, contrôle des poses et du parcours. Dépend de 03. | Graphe, centres, résidus, couverture par zone et décision `usable` ou `insufficient_capture`, avec causes. | 1–2 j |
| **DEV-05** | P0 / vision + QA | Geler la partition spatiale ; reconstruire le sous-ensemble d’entraînement et localiser les vues réservées. Dépend de 04. | `split.json`, poses de test figées, points sans contribution interdite ; échecs de localisation explicités. | 1 j |
| **DEV-06** | P0 / vision | Générer les six faces de rendu et l’adaptateur neutre. Dépend de 03 et 05. | `dataset.json`, points et masques importables sans lecteur COLMAP historique ; tests de conventions et d’absence de fuite réussis. | 1 j |
| **DEV-07** | P0 / vision | Intégrer le témoin gsplat et une véritable reprise. Dépend de 01 et 06. | Test court, checkpoint restaurable, premier PLY et rendus réservés. Mémoire et durée enregistrées. | 1–2 j |
| **DEV-08** | P1 / vision | Comparaison locale Brush bornée. Dépend de 04 et d’un export de caméras adapté. | Note effort/mémoire/export ; classement qualitatif seulement si partitions ou réglages diffèrent. | 0,5 j max |
| **DEV-09** | P0 / Web | Lecteur, chargement, point de départ, parcours borné, repli panorama. Contrat `scene.json` requis ; intégration salon après 07. | Démo locale fonctionnelle sur le vrai PLY, sans trou ou inversion introduit par la conversion des axes. | 1–2 j |
| **DEV-10** | P0 / QA + Web | Comparer rendu de référence et export ; instrumenter le parcours et le chargement. Dépend de 07 et 09. | Rapport de qualité, vidéo du parcours, temps de frame, appareil/navigateur et limites connues. | 1 j |
| **DEV-11** | P1 / vision + Web | Élagage/compression ; RAD si nécessaire. Dépend de 10. | Mesure avant/après sur mêmes caméras et réseau ; master conservé ; qualité et mémoire suivies. | 1 j |
| **DEV-12** | P1 conditionnel / vision | Une amélioration ciblée après classement des défauts : géométrie ou photométrie. Dépend de 10. | Comparaison contre le témoin sur validation, puis test final ; coût supplémentaire documenté. | 1–2 j par variante |

Le total des charges représente environ **8 à 12 jours d’ingénierie pour le socle P0**, répartissables entre les rôles. Le premier diagnostic est visé à J3 ; la première chaîne navigable à J5–J10 selon les moyens disponibles. Les comparaisons et optimisations suivent ce premier résultat. Les quatre à six semaines de la R&D concernent la qualification élargie, sous réserve de nouvelles scènes disponibles.

## 6. Points d’implémentation qui conditionnent la validité

**Masques.** Séparer validité RGB, exclusion pour la géométrie et attribution des caractéristiques aux vues SfM. Dans le témoin, l’opérateur et les éléments mobiles peuvent être exclus du rendu supervisé ; les surfaces réfléchissantes sont d’abord exclues de la géométrie trompeuse. Pour la perte RGB, normaliser sur les pixels valides. Pour SSIM, exclure les fenêtres dont le support intersecte une région invalide, ou employer une formulation masquée démontrée. Noircir simplement les images n’est pas ce contrat. Une vue entièrement masquée est ignorée avec une cause, sans NaN.

**Partition.** Chaque face hérite de son panorama parent. Le jeu provisoire de la R&D est un point de départ spatial à vérifier, sans ajustement selon les scores. La sélection du checkpoint et des variantes se fait sur la validation. Les pixels test servent à la localisation sur géométrie figée et à la mesure finale, jamais à la supervision ni à la correction d’apparence. Ne pas employer le découpage automatique de l’exemple amont.

**Chargement des données.** Ne pas réappliquer un facteur de sous-échantillonnage implicite aux faces déjà dimensionnées. Échantillonner les panoramas équitablement, puis leurs faces valides ; fixer et tracer la politique de pondération. Une éventuelle correction d’angle solide sera identique entre variantes. Les intrinsèques suivent exactement chaque redimensionnement. Calculer la normalisation de scène sur les seules données d’entraînement, puis appliquer la même transformation aux caméras réservées.

**Optimisation.** Démarrer par un test de quelques centaines d’itérations pour détecter erreurs et dépassements mémoire. Examiner ensuite un premier palier court ; prolonger vers 7 000 puis 30 000 uniquement si nécessaire. Enregistrer temps écoulé et observations traitées : un nombre d’itérations ne suffit pas à comparer des lots ou résolutions différents. Geler les poses du témoin. Toute future optimisation de pose s’applique au panorama entier.

**Reprise.** Sauvegarder gaussiennes, optimiseurs, calendriers, état de densification, compteur, configuration et états aléatoires/échantillonnage. Reprendre avec le même jeu de données et vérifier une trajectoire de perte compatible, dans une tolérance déclarée pour l’éventuel non-déterminisme GPU. Une sauvegarde utilisable pour afficher la scène seulement n’est pas une reprise d’entraînement.

**Export.** Vérifier le schéma Gaussian du PLY, l’encodage des échelles/opacités, l’ordre des quaternions, les harmoniques et l’espace couleur. Le chemin initial doit conserver l’apparence standard jusque dans le lecteur. Toute variante qui exige une simplification à l’export est évaluée après cette simplification.

**Budget.** Fixer un plafond par exécution, conserver les résultats intermédiaires et arrêter proprement au plafond. Faire un bilan après les dix premières GPU-heures cumulées ; l’enveloppe de 40–80 h proposée en R&D ne constitue pas un besoin démontré. Mesurer aussi le temps opérateur. En cas de manque de mémoire, commencer par la résolution et le nombre de primitives avant de changer de machine.

## 7. Recette et règles de poursuite

| Jalon | Preuves nécessaires | Si le résultat est insuffisant |
|---|---|---|
| **J1 — poses exploitables** | Une composante couvre les zones prévues ; centres et orientations plausibles ; angles de triangulation et pistes contrôlés. 12/13 et résidu médian ≤ 1,5 pixel équivalent à 1024 sont des indicateurs d’alerte. | Vérifier axes, masques et correspondances ; faire au plus une variante SfM motivée. Produire ensuite une demande de recapture localisée si la couverture reste insuffisante. |
| **J2 — premier rendu crédible** | PLY réel, vues réservées rendues, trajet de 30–60 s avec déplacement entre stations ; zones fragiles identifiées. | Classer chaque défaut : acquisition, poses, photométrie, moteur ou export. Choisir la correction correspondante. |
| **J3 — visite Web utilisable** | Contrôles et repli fonctionnels ; repère correct ; absence d’artefact bloquant dans le parcours validé ; performances mesurées. | Réduire budget de rendu/résolution ou préparer les niveaux de détail ; signaler un parcours plus limité si nécessaire. |
| **J4 — candidat industrialisable** | Exécution reproductible, coût mesuré, droits des composants recensés et résultats sur au moins trois intérieurs supplémentaires. | Conserver le statut de prototype salon ; aucune généralisation déduite du seul lot initial. |

Un artefact est bloquant s’il empêche de reconnaître une zone, masque durablement le champ de vision, fait traverser un obstacle du parcours ou interrompt la navigation. Archiver les exemples et leur position ; la seule appréciation « joli » ne suffit pas.

Cibles conservées pour J3 : 60 fps médianes desktop à 1920 × 1080 pixels effectifs, temps de frame p95 ≤ 33 ms ; 30 fps mobile, p95 ≤ 50 ms sur cinq minutes. Le Mac M3 existant est le premier poste desktop à documenter. Choisir un téléphone réellement disponible et inscrire modèle, OS, navigateur et résolution dans le rapport avant les essais.

Chargement : cache froid, 50 Mbit/s, RTT 50 ms ; première interaction 3D en moins de 5 s visée, premier contenu ≤ 15 Mo. Un panorama affiché en attendant ne compte pas comme première interaction 3D. Mesurer poids complet, mémoire après décodage et temps de chargement séparément. Ces cibles ne sont pas garanties à ce stade.

PSNR/SSIM/LPIPS sont fournis par vue avec masque et fraction évaluée. Comparer séparément erreur de reconstruction, erreur d’export et erreur de compression. Aucun gain relatif unique de LPIPS ne déclenche à lui seul une décision produit.

## 8. Tests utiles et définition de terminé

Les tests prioritaires ciblent les risques de validité : projection sur points/rayons synthétiques connus, conversion du rig, couverture des six directions, centres communs, séparation par panorama, influence nulle des pixels exclus, reprise d’un entraînement interrompu et concordance des caméras entre moteur et navigateur.

Un ticket de pipeline est terminé lorsque son artefact et son journal sont produits par une commande documentée, que les erreurs attendues sont explicites et que ses contrôles pertinents réussissent. Le lot est terminé lorsque l’équipe peut repartir des entrées et recréer la démo, avec versions, coûts, captures de contrôle et limitations documentés.

Déposer avec chaque résultat une courte note indiquant : **ce qui est observé, ce qui reste une hypothèse, le défaut dominant et la prochaine action**. Commencer maintenant par les environnements, la préparation et le diagnostic des poses ; le premier résultat attendu du développement est J1.
