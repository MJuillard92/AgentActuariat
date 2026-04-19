# obs_vs_modeled

Source : `AF8796-TD3_v1.0.pdf` — pages 8–10

## Chunk 0 (2197 chars)

4. COMMENTAIRES
Dans le cadre des opérations de certification, une fois les données validées, des contrôles sont
effectués sur les 3 points suivants :
– les choix retenus en termes de construction ;
– la comparaison des décès observés avec les décès reconstitués par la table
d’expérience ;
– le positionnement par rapport à la table d’expérience précédente.
Il est précisé que les commentaires effectués sur les méthodes de construction sont fournis ici
à titre informatif et n’impactent pas le caractère « certifiable » ou pas de la table. De ce point
de vue, seule est prise en compte la cohérence de la table proposée avec les observations.
4.1. DECES OBSERVES ET DECES MODELISES
Afin de mesurer la prudence de la table d’expérience, l’approche retenue a consisté à comparer
le nombre de décès observés et le nombre de décès prédit par la table d’expérience. Cette
comparaison est effectuée sur la période d’observation.
Les contrôles sont effectués sur l’ensemble de la période d’observation et prennent en compte
les contrats de moins de 2 ans d’ancienneté, exclus de la construction pour des raisons de
prudence du fait de la sélection médicale.
Pour déterminer le nombre de décès estimé par la table d’expérience, il a été nécessaire de
reconstituer les expositions au risque par âge et par sexe. Les expositions au risque ont été
déterminées sur une base journalière en observant, pour chaque assuré, le temps passé entre
deux âges. Le calcul de la contribution de l’assuré d’âge x à l’exposition de l’année la période
31/12/N-01/01/N+1 est effectué comme suit4 :
!"#
{([!]+!A )
%CD(F*+,"#-./+01O3"4#"
}
F*+5+.63-./+01O3"4#
74#30".63"4# ([!]) = ,
%CD
!"#
{
A%C'(F*+,C-."/(0F1
([!]+!O )
34567
}
80F/-(+9/(0F ([!]+O ) = ,
345
où :
− !"#A%&D(F#*+,-%.&=/0N!O0NO!+N/ 3,-#4,%FF,&5#!FF6*# désigne l’âge en jours à la fin de
l’observation de cet assuré pour l’année d’observation N,
− !"#A#%&D(%)#*+,D-./=!0NO3O43!0 A,D#5,-)),/6#!))&*# désigne l’âge en jours au début de
l’observation de cet assuré pour l’année d’observation N,
− [!] désigne la partie entière de x, soit l’âge en années entières.
4 Le choix de la règle d’arrondi avec 365 ou 365,25 impacte d’environ 0,5 % les résultats obtenus.

## Chunk 1 (1665 chars)

Par convention, la date de fin d’observation d’un assuré décédé au cours de l’année N a été
fixée au 01/01/N+1. Au global, l’utilisation de la table d’expérience conduit à majorer de 14,5%
le nombre de décès sur la période d’observations.
La table d’expérience apparait globalement très prudentes au regard du nombre de décès
observé, ce dernier étant inférieur à la borne inférieure de l’intervalle de confiance à 95 %. La
comparaison par tranches d’âges représentant près de 10 % de l’exposition au risque vient
confirmer cette observation :
Tableau 7 – Comparaison des décès observés et des décès modélisés (par classe d’âges)
Rapport
Décès Décès
Age Exposition Proportion Ecart différence/décès IC Min 95% IC Max 95%
observés prédits
prédits
0-28 82 327 10,5% 43 40 3 8,1% 27 52
29-33 70 618 9,0% 42 40 2 4,0% 28 53
34-37 75 873 9,7% 57 59 - 2 -3,8% 44 74
38-41 90 554 11,6% 59 101 - 42 -41,5% 81 121
42-44 73 613 9,4% 112 117 - 5 -3,9% 95 138
45-47 75 998 9,7% 137 153 - 16 -10,4% 129 177
48-50 73 978 9,5% 181 209 - 28 -13,5% 181 238
51-54 88 280 11,3% 353 360 - 7 -2,0% 323 397
55-59 83 278 10,7% 482 525 - 43 -8,3% 481 570
60-120 65 891 8,4% 727 907 - 180 -19,8% 848 965
Total 780 411 100,0% 2 193 2 512 - 319 -12,7% 2 414 2 610
Le fait que le nombre de décès observé est inférieur à la borne inférieure de l’intervalle de
confiance à 95 % pour les décès théoriques met en évidence une prudence significative.
L’analyse par âge conforte ce constat :
Figure 8 – Comparaison des décès observés et des décès modélisés (par âges)
)C#$##%%%
)"#$##%%%
)##$##%%%
(#$##%%%
D#$##%%%
C#$##%%%
"#$##%%%
!
,+*)('&('&%A#"!
-.&
*+I-.%/0.123+. *+I-.%1.456+. 78!9:; 78<%9:;

## Chunk 2 (934 chars)

Cette analyse est complétée, pour chaque année de la période considérée, par une comparaison
entre les décès prédits en début d’année et les décès effectivement observés. Plus précisément,
on considère l’effectif présent au début de chaque année et, en supposant qu'il reste exposé toute
l’année, on calcule une prédiction du nombre de décès de l’exercice. Cette prédiction est
rapprochée du nombre de décès effectivement observé pendant l’année parmi les assurés entrés
dans le risque avant le début de l’année.
La prédiction annuelle du nombre de décès conduit globalement à un nombre de décès prédit
rapporté au nombre de décès observé toujours supérieur à 1,5 quelle que soit l’année de 2007 à
2011. Ce rapport est même supérieur à 2 en 2010 et 2011. La répartition âge par âge est la
suivante :
Figure 9 – Comparaison des décès prédits et des décès observés
!)*
!)"
1.23
&)*
&)"
!"&&
&""
!"&"
%"
!""(
$"
.//0- !""%
#"
+,-
!""' !"
