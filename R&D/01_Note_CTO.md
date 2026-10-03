# Photos 360 → visite Web en Gaussian Splatting

**Note d’orientation R&D — 1er octobre 2026**  
**Cas étudié :** salon capturé avec une Ricoh Theta Z1.  
**Priorité confirmée :** visite immersive photoréaliste sur le Web. Infrastructure à définir après l’étude.

## Décision recommandée

Construire un **pipeline multi-vues modulaire**, avec contrôle explicite de la géométrie, puis optimisation de Gaussian Splatting par scène. Pour les JPEG disponibles, retenir comme socle **COLMAP avec rig de vues perspectives → gsplat → export PLY de référence → compression et niveaux de détail → lecteur Web**. Ajouter la profondeur apprise et la compensation photométrique seulement lorsque les expériences démontrent leur apport.

Cette recommandation est un choix d’architecture pour notre cas, **pas un classement mesuré sur ce salon**. L’approche la plus aboutie associe une reconstruction vérifiable, les améliorations récentes utiles et un rendu réellement déployable. Le résultat d’un article sur un autre jeu de données ne suffit pas à choisir le moteur.

COLMAP propose un traitement des panoramas par caméras virtuelles solidaires ; gsplat fournit le moteur d’optimisation CUDA et des exemples de reconstruction à partir de COLMAP. Ces composants permettent de remplacer un étage sans réécrire la chaîne entière. [COLMAP, rigs](https://colmap.github.io/rigs.html), [gsplat, reconstruction COLMAP](https://docs.gsplat.studio/main/examples/colmap.html).

## Ce que les captures permettent d’affirmer

L’audit local a trouvé **13 JPEG dans `Input/`**, de `R0010004.jpeg` à `R0010016.jpeg`, tous lisibles, distincts par SHA-256 et en **6720 × 3360**, soit 22,58 mégapixels. Leur poids total est de **58,47 Mo**. Les EXIF identifient la Theta Z1 ; les prises s’étendent de 17:42:16 à 17:45:25, le 1er octobre 2026, décalage +02:00 déclaré par l’appareil.

L’inspection visuelle montre des changements de position dans le salon et la salle à manger. C’est une base crédible pour un premier prototype. Elle ne prouve pas encore que toutes les prises pourront être recalées ni que la couverture suffira pour une navigation libre. Les principaux risques visibles sont les baies vitrées lumineuses, le miroir et autres reflets, les grandes surfaces unies, les objets fins et les occultations autour du mobilier.

Les paramètres varient entre ISO 500 et 1000, avec des temps de pose de 1/60 à 1/40 s, à f/2,1. La compensation des écarts d’apparence mérite donc une expérience dédiée. **Treize panoramas restent au maximum treize centres de capture**, même si l’on en extrait 156 vues virtuelles.

## Les avancées récentes qui changent les choix

| Avancée | Conséquence pour le projet |
|---|---|
| **COLMAP 4.1–4.2, 2026** | Le modèle sphérique natif offre une variante à comparer au rig perspective. Les auteurs signalent un compromis vitesse/précision ; conserver le rig comme référence initiale. [Versions](https://github.com/colmap/colmap/releases). |
| **PPISP, CVPR 2026** | Candidat pertinent pour dissocier les variations d’exposition/couleur de la structure 3D. Tester son gain et le comportement de son apparence après export Web. [Projet NVIDIA](https://research.nvidia.com/labs/sil/projects/ppisp/). |
| **PanoSplatt3R, ICCV 2025 ; PFGS360, CVPR 2026 selon son dépôt** | Challengers pour les panoramas sans poses et l’initialisation. Leur transfert aux 13 JPEG espacés doit être mesuré ; le second cible des vidéos. [PanoSplatt3R](https://github.com/zhichu99/PanoSplatt3R), [PFGS360](https://github.com/zcq15/PFGS360). |
| **FullCircle, mars 2026** | Orientation forte pour une future capture en deux fisheye bruts, avec calibration et gestion de l’opérateur. Le dépôt attend des images fisheye absentes du lot actuel. [Article](https://arxiv.org/abs/2603.22572), [code](https://github.com/theialab/fullcircle). |
| **SOG/SPZ et lecteurs avec niveaux de détail** | La diffusion Web fait partie du pipeline : compression, chargement progressif et budget de gaussiennes visibles. [PlayCanvas](https://developer.playcanvas.com/user-manual/supersplat/streaming/), [Spark](https://sparkjs.dev/docs/lod-getting-started/). |

Les modèles génératifs de complétion peuvent inventer des zones non photographiées. Ils restent une piste secondaire pour des effets visuels explicitement assumés. La priorité est de restituer le salon réellement capturé.

## Programme proposé

Prévoir **4 à 6 semaines**, avec un ingénieur vision/3D principal et un appui Web à temps partiel. C’est une estimation de planification, sous réserve de la qualité des poses et des éventuelles nouvelles captures.

1. **Semaine 1 — établir la reconstructibilité.** Masques, rig panoramique, graphe de correspondances, poses, diagnostic des zones insuffisamment observées. Premier jalon : une reconstruction connectée et visuellement cohérente.
2. **Semaine 2 — produire le témoin.** Premier PLY Gaussian Splatting et parcours Web, évaluation sur des positions photographiques exclues de l’entraînement.
3. **Semaines 3–4 — comparer les améliorations.** Profondeur avec confiance, traitement photométrique, puis un challenger panoramique si une limitation précise le justifie.
4. **Semaines 5–6 — qualifier la diffusion.** Compression, niveaux de détail, parcours autorisé, tests desktop/mobile, reproductibilité sur au moins trois intérieurs supplémentaires.

**Infrastructure conseillée pour les essais :** location temporaire d’une machine Linux NVIDIA avec 24 Go de VRAM comme point de départ, 48 Go uniquement si les mesures mémoire ou un challenger le justifient. Le MacBook Air M3 / 16 Go actuel convient à l’audit, à l’orchestration et au lecteur Web ; il ne fournit pas CUDA. Aucun achat matériel ni lancement cloud n’a été effectué.

## Critères de décision CTO

Les seuils suivants sont des **cibles initiales à tester**, pas des performances obtenues : parcours sans artefact bloquant ; 60 images/s médianes à 1080p sur le poste desktop de référence et 30 sur le mobile retenu ; première vue interactive en moins de 5 s sur une connexion contrôlée à 50 Mbit/s ; première charge visée ≤ 15 Mo. Le choix du moteur doit aussi tenir compte du temps opérateur et du taux de reconstructions réussies.

La qualité doit être mesurée sur des **panoramas entiers tenus à l’écart**, jamais en répartissant les faces d’un même panorama entre entraînement et test. La scène exportée dans le navigateur doit être évaluée elle aussi : un bon rendu dans le moteur d’entraînement ne garantit pas une bonne visite Web.

**Décision à prendre après le premier jalon :** poursuivre avec les 13 prises, compléter les zones faibles, ou privilégier une visite guidée avec déplacements limités. Le premier investissement doit valider la capture et les poses avant d’augmenter la complexité du modèle.

## Livrables de cette phase

L’audit des fichiers et des images est réalisé. L’état de l’art, l’architecture et le protocole sont documentés. **Aucun modèle 3D n’a encore été entraîné ; aucun `.ply` reconstruit n’est présenté comme résultat.**

- [Étude technologique détaillée](</Users/stani/code/3D Scene/Output/RD/02_Etat_de_l_art.md>)
- [Pipeline, expériences et critères de validation](</Users/stani/code/3D Scene/Output/RD/03_Pipeline_et_validation.md>)
- [Planche des 13 captures](</Users/stani/code/3D Scene/Output/RD/audit/panoramas_contact.jpg>)
- [Manifeste vérifiable des fichiers](</Users/stani/code/3D Scene/Output/RD/audit/capture_manifest.json>)
