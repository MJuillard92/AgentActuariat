# data_submission

Source : `AF8796-TD3_v1.0.pdf` — pages 4–7

## Chunk 0 (2237 chars)

1. LES CONTRATS
Allianz commercialise des contrats temporaire décès garantissant le paiement d’un capital en
cas de décès. Les contrats concernés sont ceux gérés par le système GCP, qui sont les suivants :
AGFCanopee, AGFEssentiel, AGFEssentielS, AssDCAGF, ChorusAV, ChorusAvGT,
ChorusCl, ChorusGC, ChorusGT, HERMES, NM_PFA_TP, Stabila, TVA, Variato et Variato5.
Une présentation plus détaillée de ces contrats est effectuée dans le rapport de certification de
2003, n°AF8796-TD1.
Les contrats sont de nature relativement diverse, tant au niveau du mode de diffusion que des
procédures d’acceptation ou des règles de tarification.
On notera qu’il existe une sélection médicale à l’entrée, ce qui conduit Allianz à ne retenir pour
la construction que les polices de plus de 2 ans d’ancienneté. Dans le présent rapport les
contrôles sont effectués sur l’ensemble de la population sous risque.
Allianz a communiqué une série d’informations sur la production et le stock de contrats en
portefeuille pour une période d’observation s’étendant du 01/01/2007 au 31/12/2011.
Au vu de ces éléments, il est possible d’apporter un certain nombre de commentaires sur la
population assurée :
− L’effectif sous risque au 31/12/2011 est d’environ 250 000 contrats (37 % de femmes
et 63 % d’hommes).
− L’âge moyen du portefeuille sur la période d’observation est d’environ 44 ans.
− Au global, l’ensemble des critères retenus (âge, nature des contrats, exclusions, etc.)
définit une cible plutôt large.
2. LES DONNEES TRANSMISES
2.1. DONNEES INITIALES
Le fichier de données avant traitement, est issu du fichier au format CSV « TP2012_FP.txt »
(305 881 observations).
Identification des décès : la date de sortie de l’exposition est calculée à partir des champs
DECES et SORTIE en appliquant la règle suivante : si le champ DECES contient une date
strictement inférieure au 31/2/9999 alors on observe une sortie par décès et la date du décès est
fournie par la valeur du champ DECES. Si l’individu n’est pas sorti par décès on regarde si une
date est indiquée dans le champ SORTIE, celle-ci est alors utilisée comme date de sortie
(censurée).
Le fichier de données initiales a fait l’objet des traitements suivants :
– calcul de l’âge en date de sortie ;

## Chunk 1 (1516 chars)

– distinction2 des sorties par décès des autres sorties ;
– ventilation des décès par année de survenance ;
– application d’un ensemble de retraitements permettant de supprimer les données
aberrantes :
o suppression des lignes pour lesquelles l’âge à la souscription est négatif ;
o suppression des lignes pour lesquelles l’âge de sortie est négatif ;
o suppression des lignes pour lesquelles l’âge à la souscription est supérieur
à 110 ans ;
o suppression des lignes pour lesquelles l’âge de sortie est supérieur à l’âge
à la souscription.
o suppression des lignes en dehors de la plage d’observation [2007-2011].
La séquence des traitements présentée ci-dessus conduit à une base de 253 067 lignes.
Les enregistrements de la table ainsi construite sont des polices et un regroupement selon la clé
composée de l'identifiant (numéro client) et du sexe est effectué afin de disposer d'une table par
individu. Ce traitement ne modifie pas la volumétrie de la base qui reste de 253 067 lignes.
Ces traitements sont synthétisés ci-après :
Tableau 1 – Construction de la table de données
Sur cette base on effectue quelques statistiques descriptives présentées ci-après.
2.2. STATISTIQUES DESCRIPTIVES DE LA BASE RETRAITEE
Les caractéristiques de la base retraitée sont présentées ci-après :
Tableau 2 – Statistiques de base
De manière plus précise on a :
2 Une sortie est considérée comme un décès dès lors que la variable Statut prend la modalité « sinistré » et que ce
décès intervient durant la période d’observation.

## Chunk 2 (2157 chars)

Tableau 3 – statistiques descriptives sur l’effectif sous risque
2007 2008 2009 2010 2011 Exposition
Total 163 053 158 418 155 235 154 276 149 430 780 411
Age moyen 44,5 44,5 44,2 43,8 43,2 44,0
On notera une bonne stabilité de l’exposition et une légère baisse de l’âge moyen, avec une
moyenne à 44 ans. Les données utilisées pour les travaux de certification de décembre 2008 (cf.
AF8796-TD2) conduisaient à des expositions pour les années 2007 et 2008 respectivement
égales à 158 890 années et 156 023 années, ce qui est cohérent avec les valeurs ci-dessus3.
L’analyse des décès conduit à :
Tableau 4 – statistiques descriptives sur les décès
Nombre de
2007 2008 2009 2010 2011
décès
Total 472 450 463 393 415 2 193
Taux 0,289% 0,284% 0,298% 0,255% 0,278% 0,281%
Age moyen 55,0 54,6 54,3 55,5 54,7 54,8
Le taux de décès comme l’âge moyen au décès sont stables sur la période d’observation. Dans
le rapport de certification précédent, les effectifs de décès en 2007 et 2008 était de 448 et 422
pour des âges moyens de 54,5 et 54,7 ans, cohérents avec les valeurs de la présente étude.
Graphiquement, l’allure générale de l’exposition et des décès est la suivante :
Figure 5 – Exposition au risque et nombre de décès
! "! #! A! %! &!! &"!
3 La convention de calcul pour les arrondis d’âges dans le calcul de l’exposition a changé depuis, ce qui explique
une part significative de l’écart.
!!!'&
!!!!&
!!!'
!
EF*
2.0N0/.-,H
!"#$%&D&$E)#*+),-.
3.44*/
5*44*/
! "! #! A! %! &!! &"!
!%
!A
!#
!"
!
EF*
/=<;:9*:9*874.6
/012%)#*+),-.
3.44*/
5*44*/

Enfin, la cohérence globale des données est confortée par l’examen du positionnement de la
mortalité masculine par rapport à la mortalité féminine. Pour cela on utilise le modèle de Cox,
qui fournit les résultats suivants :
Tableau 6 – positionnement de la mortalité masculine par rapport à la mortalité féminine
On trouve qu’à tous les âges, les taux de décès masculins sont environ le double des taux
féminins, ce qui est classiquement observé en pratique. On peut également noter que
l’hypothèse de proportionnalité est ici acceptée largement, la p-valeur du test sur les résidus de
Schönfeld étant égale à 0,96.
