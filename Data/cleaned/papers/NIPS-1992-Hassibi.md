# SecondOrderDerivativesForNetworkPruning: OptimalBrainSurgeonBabakHassibi* andDavidG. StorkRicohCaliforniaResearchCenter2882SandHillRoad, Suite115MenloPark, CA94025-7022stork@crc.ricoh.comand 
* DepartmentofElectricalEngineeringStanfordUniversityStanford, CA94305 

# AbstractWeinvestigatetheuseofinformationfrom *all* secondorderderivativesoftheerrorfunctiontoperfonnnetworkpruning (i.e., removingunimportantweightsfroma trainednetwork) inordertoimprovegeneralization, simplifynetworks, reducehardwareorstoragerequirements, increasethespeedoffurthertraining, andinsomecasesenableruleextraction. Ourmethod, OptimalBrainSurgeon (OBS), isSignificantlybetterthanmagnitude-basedmethodsandOptimalBrainDamage [LeCun, DenkerandSol1a, 1990], 
whichoftenremovethewrongweights. OBSpermitsthepruningofmoreweightsthanothermethods (forthesameerroronthetrainingset), andthusyieldsbettergeneralizationontestdata. CrucialtoOBSisa recursionrelationforcalculatingtheinverseHessianmatrixH-Ifromtrainingdataandstructuralinformationofthenet. OBSpermitsa 90%, a76%, anda 62% reductioninweightsoverbackpropagationwithweighLdecayonthreebenchmarkMONK'sproblems [ThrunetaI., 1991]. OfOBS, OptimalBrainDamage, andmagnitude-basedmethods, onlyOBSdeletesthecorrectweightsfroma trainedXORnetworkineverycase. Finally, whereasSejnowskiandRosenberg [1987Jused18,000weightsintheirNETtalknetwork, weusedOBStoprunea networktojust1560weights, yieldingbettergeneralization. 

## 1IntroductionA centralprobleminmachinelearningandpatternrecognitionistominimizethesystemcomplexity (descriptionlength, VC-dimension, etc.) consistentwiththetrainingdata. Inneuralnetworksthisregularizationproblemisoftencastasminimizingthenumberofconnectionweights. Withoutsuchweighteliminationoverfiltingproblemsandthuspoorgeneralizationwillresult. Conversely, iftherearetoofewweights, thenetworkmightnotbeabletolearnthetrainingdata. Ifwebeginwitha trainednetworkhavingtoomanyweights, thequestionsthenbecome: Whichweightsshouldbeeliminated? Howshouldtheremainingweightsbeadjustedforbestperformance? Howcansuchnetworkpruningbedoneina computationallyefficientway? 

Magnitudebasedmethods [Hertz, KroghandPalmer, 1991] eliminateweightsthathavethesmallestmagnitude. Thissimpleandnaivelyplausibleideaunfortunatelyoftenleadstotheeliminationofthewrongweights - smallweightscanbenecessaryforlowerror. OptimalBrainDamage [LeCun, DenkerandSolla, 1990] usesthecriterionofminimalincreaseintrainingerrorforweightelimination. Forcomputationalsimplicity, OBDassumesthattheHessianmatrixisdiagonal: infact. however, Hessiansforeveryproblemwehaveconsideredarestrongly *non-diagonal,* andthisleadsOBDtoeliminatethewrongweights. Thesuperiorityofthemethoddescribedhere - OptimalBrainSurgeon - liesingreatpantothefactthatitmakesnorestrictiveassumptionsabouttheformofthenetwork'sHessian, andtherebyeliminatesthecorrectweights. Moreover, unlikeothermethods, OBSdoesnotdemand (typicallyslow) retrainingafterthepruningofa weight. 

## 2OptimalBrainSurgeon

$$\deltaE=\left(\frac{\partialE}{\partial\mathbf{w}}\right)^{T}\cdot\delta\mathbf{w}+\frac{1}{2}\,\delta\mathbf{w}^{T}\cdot\mathbf{H}\cdot\delta\mathbf{w}+O(\|\delta\mathbf{w}\|^{3})$$

Inderivingourmethodwebegin, asdoLeCun, DenkerandSolla [1990], byconsideringa networktrainedtoa localminimuminerror. ThefunctionalTaylorseriesoftheerrorwithrespecttoweights (orparameters, seebelow) is: 

$$|\mathbf{\theta}\rangle$$
$$({\boldsymbol{3}})$$
$\left(4\right)$. 
(1) 
whereH = ;]2E/ aw2istheHessianmatrix (containingallsecondorderderivatives) andthesuperscriptT denotesvectortranspose. Fora networktrainedtoa localminimuminerror, thefirst (linear) termvanishes: wealsoignorethethirdandallhigherorderterms. Ourgoalisthentosetoneoftheweightstozero (whichwecallwq) tominimizetheincreaseinerrorgivenbyEq. l. EliminatingWqisexpressedas: 

$$\delta\mathbf{w}_{q}+\mathbf{w}_{q}=0\qquad\text{ormoregenerally}\quad\mathbf{e}_{q}^{T}\cdot\delta\mathbf{w}+\mathbf{w}_{q}=0\tag{2}$$ where $\mathbf{e}_{q}$ istheunitvectorinweightspacecorrespondingto (scalar) weight $\mathbf{w}_{q}$. Ourgoalisthentosolve:
$$Min_{q}\{Min_{\delta\mathbf{w}}\{\frac{1}{2}\,\delta\mathbf{w}^{\mathrm{T}}\cdot\mathbf{H}\cdot\delta\mathbf{w}\}\quad\text{suchthat}\quad\epsilon_{q}^{T}\cdot\delta\mathbf{w}+\mathbf{w}_{q}=0\}$$ TosolveEq. 3weforma LagrangianfromEqs. 1and2: $$L=\frac{1}{2}\,\delta\mathbf{w}^{\mathrm{T}}\cdot\mathbf{H}\cdot\delta\mathbf{w}+\lambda(\epsilon_{q}^{T}\cdot\delta\mathbf{w}+\mathbf{w}_{q})$$

whereA. isa Lagrangeundeterminedmultiplier. Wetakefunctionalderivatives, employtheconstraintsofEq. 2, andusematrixinversiontofindthattheoptimalweightchangeandresultingchangeinerrorare: 

$$\delta\mathrm{w}=-\frac{w_{q}}{\left[\mathrm{H}^{-1}\right]_{qq}}\mathrm{H}^{-1}\cdot\mathrm{e}_{q}\qquad\text{and}\qquadL_{q}=\frac{1}{2}\frac{w_{q}^{2}}{\left[\mathrm{H}^{-1}\right]_{qq}}\tag{5}$$

NotethatneitherH norH·Ineedbediagonal (asisassumedbyLeCunetal.): moreover, ourmethodrecalculatesthemagnitudeof *all* theweightsinthenetwork, bytheleftsideofEq. 5. WecallLqthe "saliency" ofweightq - theincreaseinerrorthatresultswhentheweightiseliminated - adefinitionmoregeneralthanLeCunetal. 's, andwhichincludestheirsinthespecialcaseofdiagonalH. Thuswehavethefollowingalgorithm: 

#### OptimalBrainSurgeonProcedure1. Traina "reasonablylarge" networktominimumerror. 

2. ComputeH·I . 

3. Findtheq thatgivesthesmallestsaliencyLq = Wq2/(2[H·I]qq). IfthiscandidateerrorincreaseismuchsmallerthanE, thentheqthweightshouldbedeleted, andweproceedtostep4; otherwisegotostep5. (Otherstoppingcriteriacanbeusedtoo.) 
4. Usetheq fromstep3 toupdate *all* weights (Eq. 5). Gotostep2. 5. NomoreweightscanbedeletedwithoutlargeincreaseinE. (Atthispointitmaybedesirabletoretrainthenetwork.) 
Figure1 illustratesthebasicidea. Therelativemagnitudesoftheerrorafterpruning (beforeretraining. ifany) dependupontheparticularproblem, buttosecondorderobey: E(mag) ~ E(OBD) ~ E(OBS). whichisthekeytothesuperiorityofOBS. InthisexampleOBSandOBDleadtotheeliminationofthesameweight (weight1). Inmanycases, however. OBSwilleliminate *different* weightsthanthoseeliminatedbyOBD (cf. Sect. 6). WecallourmethodOptimalBrain *Surgeon* becauseinadditiontodeletingweights, itcalculatesand *changes* thestrengthsofotherweightswithouttheneedforgradientdescentorother 

![2_image_0.png](2_image_0.png)

Figure1: Errorasa functionoftwoweightsina network. The (local) minimumoccursatweightw·, foundbygradientdescentorotherlearningmethod. Inthisillustration, amagnitudebasedpruningtechnique (mag) thenremovesthesmallestweight, weight2; OptimalBrainDamagebeforeretraining (OBD) removesweightI. Incontrast, ourOptimalBrainSurgeonmethod (OBS) notonlyremovesweightI, butalsoautomaticallyadjuststhevalueofweight2 tominimizetheerror, withoutretraining. Theerrorsurfacehereisgeneralinthatithasdifferentcurvatures (secondderivatives) alongdifferentdirections, aminimumata non-specialweightvalue, anda non-diagonalHessian (i.e., principalaxesare *not* paralleltotheweightaxes). Wehavefound (tooursurprise) thateveryproblemwehaveinvestigatedhasstronglynon-diagonalHessians 
- therebyexplainingtheimprovmentofourmethodoverthatofLeCunetal. 

## 3ComputingTheInverseHessianThedifficultyappearstobestep2 intheOBSprocedure, sinceinvertinga matrixofthousandsormillionsoftermsseemscomputationallyintractable. Inwhatfollowsweshallgivea generalderivationoftheinverseHessianfora *fullytrained* neuralnetwork. Itmakesnodifferencewhetheritwastrainedbybackpropagation, competitivelearning, theBoltzmannalgorithm, oranyothermethod, solongasderivativescanbetaken (seebelow). WeshallshowthattheHessiancanbereducedtothesamplecovariancematrixassociatedwithcertaingradientvectors. Furthennore, thegradientvectorsnecessaryforOBSarenormallyavailableatsmallcomputationalcost; thecovarianceformoftheHessianyieldsa recursiveformulaforcomputingtheinverse. 

Considera generalnon-linearneuralnetworkthatmapsaninputvectorinofdimensionnjintoanoutputvector0 ofdimensionno' accordingtothefollowing: 

$$\mathbf{0}=\mathbf{F}(\mathbf{w},\mathbf{in})$$

0= F(w,in) (6) 
wherew isann dimensionalvectorrepresentingtheneuralnetwork'sweightsorotherparameters. Weshallrefertow asa weightvectorbelowforsimplicityanddefiniteness, butitmustbestressedthatw couldrepresent *any* continuousparameters, suchasthosedescribingneuraltransferfunction, weightsharing, andsoon. Themeansquareerrorcorrespondingtothetrainingsetisdermedas: 

$E=\frac{1}{2\text{P}}\sum_{k=1}^{\text{P}}(\mathfrak{t}^{[k]}-\mathfrak{o}^{[k]})^{T}(\mathfrak{t}^{[k]}-\mathfrak{o}^{[k]})$
$\left(\boldsymbol{\Theta}\right)$. 

$$\left(T\right)$$
whereP isthenumberoftrainingpatterns, andtlk] andolk] arethedesiredresponseandnetworkresponseforthekthtrainingpattern. Thefirstderivativewithrespecttow is: 

$\partialE$$\partialw$$=-\frac{1}{\rho}\sum\limits_{k=1}^{\rho}\frac{\partialF(w,in^{[k]})}{\partialw}(t^{[k]}-o^{[k]})$
$$({\mathfrak{s}})$$
$$({\mathfrak{g}})$$
andthesecondderivativeorHessianis: 

$$\mathbf{H}\equiv{\frac{\partial^{2}E}{\partial\mathbf{w}^{2}}}={\frac{1}{P}}\sum_{k=1}^{P}{\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}}\cdot{\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}}^{T}-{\frac{\partial^{2}\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}^{2}}}\cdot(\mathbf{t}^{[k]}-\mathbf{0}^{[k]})]$$

Nextweconsidera networkfullytrainedtoa localminimuminerroratw*. UnderthisconditionthenetworkresponseO[k] willbeclosetothedesiredresponset[k], andhenceweneglectthetenninvolving (t[k]- ork]). Evenlateinpruning, whenthiserrorisnotsmallfora singlepattern, thisapproximationcanbejustified (seenextSection). Thissimplificationyields: 

H =! fdF(w,in[k]). dF(w,in[k) T 
$\mathbf{H}=\frac{1}{\mathbf{P}}\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}\cdot\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}$
$$(10)$$
Ifoutnetworkhasjusta singleoutput, wemaydefinethen-dimensionaldatavectorXrk] ofderivativesas: 

$\mathbf{x}^{[k]}\equiv\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}$ $\mathbf{H}=\frac{1}{\mathbf{P}}\sum\mathbf{x}^{[k]}\cdot\mathbf{x}^{[k]}$
ThusEq. 10canbewrittenas: H =! fX[k). X[k]TP *k=1* 
$$(11)$$
$$(12)^{\frac{1}{2}}$$
$$(13)$$
$$(14)$$
ThusEq. 10canbewrittenas:
Ifinsteadournetworkhas *mUltiple* outputunits, thenX willbeann xnomatrixofthefonn: 

$$\mathbf{X}^{[k]}=\frac{\partial\mathbf{F}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}}=(\frac{\partial\mathbf{F}_{1}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}},...,\frac{\partial\mathbf{F}_{n_{\mathbf{a}}}(\mathbf{w},\mathbf{in}^{[k]})}{\partial\mathbf{w}})=(\mathbf{X}_{1}^{[k]},...,\mathbf{X}_{n_{\mathbf{a}}}^{[k]})$$ where $\mathbf{F}_{i}$ isthe $i^{\text{th}}$ componentof $\mathbf{F}_{i}$. HenceinthismultipleoutputunitcaseEq. 10generalizesto:
$\mathbf{H}=\frac{1}{\mathbf{P}}\sum\limits_{\mathbf{x}}^{n_{o}}\mathbf{x}^{[\mathbf{k}]}.\mathbf{x}^{[\mathbf{k}]}$ $\mathbf{P}\sum\limits_{\mathbf{k}=1}^{n_{o}}\mathbf{x}^{[\mathbf{k}]}.\mathbf{x}^{[\mathbf{k}]}$ $\mathbf{P}\sum\limits_{\mathbf{k}=1}^{n_{o}}\mathbf{x}^{[\mathbf{k}]}.\mathbf{x}^{[\mathbf{k}]}$ 
Equations12and14showthatH isthesamplecovariancematrixassociatedwiththegradientvariableX. 

Equation12alsoshowsthatforthesingleoutputcasewecancalculatethefullHessianbysequentiallyaddinginsuccessive "component" Hessiansas: 

$${\rmH}_{\rmm+1}={\rmH}_{\rmm}+\frac{1}{\rmP}\,{\rmX}^{\{{\rmm+1}\}}\,.\,{\rmX}^{\{{\rmm+1}\}{\rmT}}\quad\mbox{with}\quad{\rmH}_{0}=\alpha{\rmI}\,\,\,\,{\rmand}\,\,\,\,{\rmH}_{\rmP}={\rmH}\tag{15}$$ ButOptimalBrainSurgeonrequiresthe_inverse_ofH (Eq. 5). Thisinversecanbecalculatedusinga 
standardmatrixinversionfonnula [Kailath, 1980]: 

(A + 8 . C . 0)-1 = A-I - A-I. 8 . (C-I + D. A-I. 8)-1 . D . A-I (16) appliedtoeachtennintheanalogoussequenceinEq. 16: H-1 . X[m+1) . X[m+1)T . H-IH-I - H-I - mm withHOi = a-IIandHpl = H-Im+1 - mp + x[m+I)T . H-I . X[m+lIm (17) anda (l0·8S aS 10-4) asmallconstantneededtomakeHO•Imeaningful, andtowhichourmethodisinsensitive [Hassibi, StorkandWolff, 1993b]. Actually, Eq. 17leadstothecalculationoftheinverseof 
(H + ciI), andthiscorrespondstotheintroductionofa penaltytermallliwll2inEq. 4. Thishasthebenefitofpenalizinglargecandidatejumpsinweightspace, andthushelpingtoinsurethattheneglectingofhigherorderLennsinEq. 1isvalid. 

Equation17permitsthecalculationofH·Iusinga *single* sequentialpassthroughthetrainingdata1 Sm SP. ItisalsostraightforwardtogeneralizeEq. 18tothemultipleoutputcaseofEq. 15: inthiscaseEq. 15willhaverecursionsonboththeindicesm andI giving: 

$$\begin{array}{l}\mbox{H}_{\rmm\;}_{l+1}=\mbox{H}_{\rmm\;}_{l}+\frac{1}{\mbox{P}}\mbox{X}_{l+1}^{[\rmm]}\cdot\mbox{X}_{l+1}^{[\rmm]T}\\ \mbox{H}_{\rmm+11}=\mbox{H}_{\rmm\;}_{\rmn_{0}}+\frac{1}{\mbox{P}}\mbox{X}_{1}^{[\rmm+1]}\cdot\mbox{X}_{1}^{[\rmm+1]T}\end{array}\tag{18}$$

TosequentiallycalculateU-Iforthemultipleoutputcase, weuseEq. 16, asbefore. 

# 4The (T - 0) ~ 0ApproximationTheapproximationusedforEq. 10canbejustifiedoncomputationalandfunctionalgrounds, evenlateinpruningwhenthetrainingerrorisnotnegligible. Fromthecomputationalview, wenote [rrstthatnonnallyH isdegenerate - especiallybeforesignificantpruninghasbeendone - anditsinversenotwelldefined. 

TheapproximationguaranteesthattherearenosingularitiesinthecalculationofH-1• ItalsokeepsthecomputationalcomplexityofcalculatingH-1thesameasthatforcalculatingH - O(pn2). InStatisticstheapproximationisthebasisofFisher'smethodofscoringanditsgoalistoreplacethetrueHessianwithitsexpectedvalueandguaranteethatH ispositivedefinite (therebyavoidingstabilityproblemsthatcanplagueGauss-Newtonmethods) [SeberandWild, 1989]. Equallyimportantarethefunctionaljustificationsoftheapproximation. Considera highcapactiynetworktrainedtosmalltrainingerror. Wecanconsiderthenetworkstructureasinvolvingbothsignalandnoise. 

Asweprune, wehopetoeliminatethoseweightsthatleadto "overfilting," i.e., learningthenoise. Ifourpruningmethoddid *not* employthe (t - 0) ~ 0approximation, everypruningstep (Eqs. 9and5) wouldinjectthenoisebackintothesystem, bypenalizingfornoisetenns. Adifferentwaytothinkoftheapproximationisthefollowing. AftersomepruningbyOBSwehavereacheda newweightvectorthatisa localminimumoftheerror (cf. Fig. 1). Evenifthiserrorisnotnegligible, wewanttostayascloseto *that* valueoftheerroraswecan. Thusweimaginea new, effectiveteachingsignal **t*,** thatwouldkeepthenetworknearthisnewerrorminimum. Itisthen (t* - 0) thatweineffectsettozerowhenusingEq. 10insteadofEq. 9. 

# 5AbsAndBackPropagationUsingthestandardtennino)ogyfrombackpropagation [Rumelhart, HintonandWilliams, 1986JandthesingleoutputnetworkofFig. 2, itisstraightforwardtoshowfromEq. 11thatthederivativevectorsare: 

$\mathbf{x}^{[k]}=\begin{pmatrix}\mathbf{x}^{[k]}\\ \mathbf{x}^{[k]}\end{pmatrix}$
where (20) referstoderivativeswithrespecttohidden-to-outputweightsVjand [X~.t)]T = (f' (net[.t)f (net\.t)v\.t)o~!L .... f' (net[.t)f (net\.t)v~.t)o~~) .... , 
$\mathbf{f}\left(\text{net}^{[k]}\right)\mathbf{f}\left(\text{net}^{[k]}_{n_{j}}\right)\mathbf{v}^{[k]}_{n_{j}}\mathbf{o}^{[k]}_{1},...,\mathbf{f}\left(\text{net}^{[k]}\right)\mathbf{f}\left(\text{net}^{[k]}_{n_{j}}\right)\mathbf{v}^{[k]}_{n_{j}}\mathbf{o}^{[k]}_{n_{i}}\right)$
$$(19)$$
$$(200)^{\frac{1}{2}}$$
$$(21)$$
referstoderivativeswithrespecttoinput-to-hiddenweightsuji' andwherelexicographicalorderinghasbeenused. Theneuronnonlinearityisf(·). 

![4_image_0.png](4_image_0.png)

Figure2: Backpropagationnetwithlliinputsandnjhiddenunits. Theinput-to-hiddenweightsareUjiandhidden-to-outputweightsVj. Thederivative ("data") vectorsareXvandXu (Eqs. 20and21). 

#### 6SimulationResultsWeappliedOBS, OptimalBrainDamage, anda magnitudebasedpruningmethodtothe2-2-1networkwithbiasunitofFig. 3, trainedonallpatternsoftheXORproblem. Thenetworkwasfirsttrainedtoa localminimum, whichhadzeroerror, andthenthethreemethodswereusedtopruneoneweight. Asshown,themethodsdeleteddifferentweights. WethentrainedtheoriginalXORnetworkfromdifferentinitialconditions, therebyleadingtoa differentlocalminima. WhereasthereweresomecasesinwhichOBDormagnitudemethodsdeletedthecorrectweight, onlyOBSdeletedthecorrectweightin *every* case. 

Moreover, OBSchangedthevaluesoftheremainingweights (Eq.5) toachieveperfectperfonnancewithoutanyretrainingbythebackpropagationalgorithm. Figure4 showstheHessianofthetrainedbutunprunedXORnetwork. 

![5_image_0.png](5_image_0.png)

![5_image_1.png](5_image_1.png)

Figure3: AnineweightXORnetworktrainedtoa localminimum. Thethicknessofthelinesindicatestheweightmagnitudes, andinhibitoryweightsareshowndashed. Subsequentpruningusinga magnitudebasedmethod (Mag) woulddeleteweightv3; usingOptimalBrainDamage (OBD) woulddeleteU22. Evenwithretraining, thenetworkprunedbythosemethodscannotlearntheXORproblem. Incontrast, OptimalBrainSurgeon (OBS) deletesU23andfurthennorechangedallotherweights (cf. Eq. 5) toachievezeroerrorontheXORproblem. 

Figure4: TheHessianofthetrainedbutunprunedXORnetwork, calculatedbymeansofEq. 12. Whiterepresentslargevaluesandblacksmallmagnitudes. TherowsandcolumnsarelabeledbytheweightsshowninFig. 3. Asistobeexpected, thehidden-to-outputweightshavesignificantHessiancomponents. NoteespeciallythattheHessianisfarfrombeingdiagonal. TheHessiansforallproblemswehaveinvestigated, includingtheMONK'sproblems (below), arefarfrombeingdiagonal. 

Figure5 showstwo-dimensional "slices" ofthenine-dimensionalerrorsurfaceintheneighborhoodofa localminimumatw· fortheXORnetwork. ThecutscomparetheweighteliminationofMagnitudemethods (left) andOBD (right) withtheeliminationandweightadjustmentgivenbyOBS. 

![5_image_2.png](5_image_2.png)

magnitudebasedpruningmethodwoulddeleteweightV3whereasOBSdeletesU23. (Right) TheXORerrorsurfaceasa functionofweightsU22andU23. OptimalBrainDamagewoulddeleteU22whereasOBSdeletesU23. Forthisminimum, onlydeletingU23willallowtheprunednetworktosolvetheXORproblem. 

AfterallnetworkweightsareupdatedbyEq. 5thesystemisatzeroerror (notshown). ItisespeciallynoteworthythatinneithercaseofpruningbymagnitudemethodsnorOptimalBrainDamagewillfurtherretrainingbygradientdescentreducethetrainingerrortozero. Inshort, magnitudemethodsandOptimalBrainDamagedeletethewrongweights, andtheirmistakecannotbeovercomebyfurthernetworktraining. 

OnlyOptimalBrainSurgeondeletesthecorrectweight. 

WealsoappliedOBStolargerproblems, threeMONK'sproblems, andcomparedourresultstothoseofThrunetal. [1991], whosebackpropagationnetworkoutperformedallotherapproaches (networkandrulebased) onthesebenchmarkproblemsinanextensivemachinelearningcompetition. 

| | | | Accuracy | |
|--------|------|----------|------------|-----------|
| | | training | testing | # weights |
| MONKl | BPWD | 100 | 100 | 58 |
| | aBS | 100 | 100 | 14 |
| MONK2 | BPWD | 100 | 100 | 39 |
| | aBS | 100 | 100 | 15 |
| MONK3 | BPWD | 93.4 | 97.2 | 39 |
| | aBS | 93.4 | 97.2 | 4 |

Table1: Theaccuracyandnumberofweightsfoundbybackpropagationwithweightdecay (BPWD) foundbyThrunetal. [1991], andbyOBSonthreeMONK'sproblems. 

TableI showsthatforthesameperfonnance, OBS (withoutretraining) requiredonly24%, 38% and10% 
oftheweightsofthebackpropagationnetwork, whichwasalreadyregularizedwithweightdecay (Fig. 6). 

TheerrorincreaseL (Eq. 5) accompanyingpruningbyOBSnegligiblyaffectedaccuracy. 

![6_image_0.png](6_image_0.png)

Figure6: OptimalnetworksfoundbyThrunusingbackpropagationwithweightdecay 
(Left) andbyOBS (Right) onMONKI, whichisbasedonlogicalrules. Solid (dashed) 
linesdenoteexcitatory (inhibitory) connections; biasunitsareatleft. 

ThedramaticreductioninweightsachievedbyOBSyieldsa networkthatissimpleenoughthatthelogicalrulesthatgeneratedthedatacanberecoveredfromtheprunednetwork, forinstancebythemethodsofTowellandShavlik [1992]. HenceOBSmayhelptoaddressa criticismoftenleviedatneuralnetworks: 
thefactthattheymaybeunintelligible. WeappliedOBStoa three-layerNETtalknetwork. WhileSejnowskiandRosenberg [1987] used18,000weights, webeganwithjust5546weights, whichafterbackpropagationtraininghada testerrorof5259. AfterpruningthisnetwithOBSto2438weights, andthenretrainingandpruningagain, weachieveda netwithonly1560weightsandtesterrorofonly4701 - asignificantimprovementovertheoriginal, morecomplexnetwork [Hassibi, StorkandWolff, 1993a]. ThusOBScanbeappliedtoreal-worldpatternrecognitionproblemssuchasspeechrecognitionandopticalcharacterrecognition, whichtypicallyhaveseveralthousandparameters. 

## 7AnalysisAndConclusionsWhyisOptimalBrainSurgeonsosuccessfulatreducingexcessdegreesoffreedom? Conversely, giventhisnewstandardinweightelimination, wecanask: Whyaremagnitudebasedmethodssopoor? 

ConsideragainFig. 1. Startingfromthelocalminimumatw·, amagnitudebasedmethoddeletesthewrongweight, weight2, andthroughretraining, weight1 will *increase.* Thefinal "solution" isweight1 4large, weight2 = O. Thisispreciselythe *opposite* ofthesolutionfoundbyOBS: weight1 = 0, weight2 4large. AlthoughtheactualdifferenceinerrorshowninFig. 1maybesmall, inlargenetworks, differencesfrommanyincorrectweighteliminationdecisionscanadduptoa significantincreaseinerror. 

Butmostimportantly, itissimplywishfulthinkingtobelievethataftertheeliminationofmanyincorrectweightsbymagnitudemethodsthenetcan "sortitallout" throughfurthertrainingandreacha globaloptimum, especiallyifthenetworkhasalreadybeenprunedsignificantly (cf. XORdiscussion, above). WehavealsoseenhowtheapproximationemployedbyOptimalBrainDamage - thatthediagonalsoftheHessianaredominant - doesnotholdfortheproblemswehaveinvestigated. Therearetypicallymanyoff-diagonaltermsthatarecomparabletotheirdiagonalcounterparts. ThisexplainswhyOBDoftendeletesthewrongweight, whileOBSdeletesthecorrectone. Wenotetoothatourmethodisquitegeneral, andsubsumespreviousmethodsforweightelimination. Inourterminology, magnitudebasedmethodsassumeisotropicHessian (HexI); OBDassumesdiagonalH: FARM [KungandHu, 1991] assumeslinearf(net) andonlyupdatesthehidden-to-outputweights. Wehaveshownthatnoneofthoseassumptionsarevalidnorsufficientforoptimalweightelimination. Weshouldalsopointoutthatourmethodisevenmoregeneralthanpresentedhere [Hassibi, StorkandWotff, 1993bl. Forinstance, ratherthanpruninga weight (parameter) bysettingittozero, onecaninsteadreducea degreeoffreedombyprojectingontoan *arbitrary* plane, e.g., Wq = aconstant, thoughsuchnetworkstypicallyhavea largedescriptionlength [Rissanen, 1978]. Thepruningconstraintw q = 0discussedthroughoutthispapermakesretraining (ifdesired) particularlysimple. *Several* weightscanbedeletedsimultaneously; biasweightscanbeexemptfrompruning, andsoforth. AslightgeneralizationofOBSemployscross-entropyortheKullback-Leiblererrormeasure, leadingtoFisherInfonnationmatrixratherthantheHessian (Hassibi, StorkandWolff, 1993b). WenotetoothatOBSdoesnotbyitselfgivea criterionforwhentostoppruning, andthusOBScanbeutilizedwitha widevarietyofsuchcriteria. Moreover, gradualmethodssuchasweightdecayduringlearningcanbeusedinconjunctionwithOBS. 

## AcknowledgementsThefirstauthorwassupportedinpartbygrantsAFOSR91-0060andDAAL03-91-C-OOlOtoT. Kailath, whointumprovidedconstantencouragementDeepthanksgotoGregWolff (Ricoh) forassistancewithsimulationsandanalysis, andJeromeFriedman (Stanford) forpointerstorelevantstatisticsliterature. 

# ReferencesHassibi, B. Stork, D. G. andWolff, G. (1993a). OptimalBrainSurgeonandgeneralnetworkpruning 
(submittedtoICNN, SanFrancisco) 
Hassibi, B. Stork, D. G. andWolff, G. (1993b). OptimalBrainSurgeon, InformationTheoryandnetworkcapacitycontrol (inpreparation) 
Hertz, J., Krogh, A. andPalmer, R. G. (1991). *IntroductiontotheTheoryofNeuralComputation* Addison-Wesley. 

Kailath, T. (1980). *LinearSystems* Prentice-Hall. Kung, S. Y. andHu, Y. H. (1991). AFrobeniusapproximationreductionmethod (FARM) fordetenniningtheoptimalnumberofhiddenunits, *ProceedingsoftheIJCNN-9I* Seattle, Washington. 

LeCun, Y., Denker, J. S. andSoUa, S. A. (1990). OptimalBrainDamage, inProceedingsoftheNeuralInformationProcessingSystems-2, D. S. Touretzky (ed.) 598-605, Morgan-Kaufmann. 

Rissanen, J. (1978). Modellingbyshortestdatadescription, *Aulomatica* 14,465-471. Rumelhart, D. E., Hinton, G. E., andWilliams, R. J. (1986). LearningInternalrepresentationsbyerrorpropagation, Chapter8 (318-362) in *ParallelDistributedProcessingI* D. E. RumelhartandJ. L. McClelland (eds.) MITPress. 

Seber, G. A. F. andWild, C. J. (1989). *NonlinearRegression* 35-36Wiley. Sejnowski, T. J., andRosenberg, C. R. (1987). ParallelnetworksthatlearntopronounceEnglishtext, ComplexSyslemsI, 145-168. 

Thrun, S. B. and23co-authors (1991). TheMONK'sProblems - Aperfonnancecomparisonofdifferentlearningalgorithms, CMU-CS-91-197Carnegie-MellonU. DepartmentofComputerScienceTechReport. 

Towell, G. andShavlik, J. W. (1992). Interpretationofartificialneuralnetworks: Mappingknowledgebasedneuralnetworksintorules, in *ProceedingsoftheNeuralIn/ormationProcessingSystems-4,* ]. 

E. Moody, D. S. TouretzkyandR. P. Lippmann (eds.) 977-984, Morgan-Kaufmann.