# WeightedLow-RankApproximationsNathanSrebronati@mit.eduTommiJaakkolatommi@ai.mit.eduDept. ofElectricalEngineeringandComputerScience, MassachusettsInstituteofTechnology, Cambridge, MA

## AbstractWestudythecommonproblemofapproximatinga targetmatrixwitha matrixoflowerrank. Weprovidea simpleandefficient (EM) algorithmforsolving *weighted* low-rankapproximationproblems, which, unliketheirunweightedversion, donotadmita closedformsolutioningeneral. Weanalyze, inaddition, thenatureoflocallyoptimalsolutionsthatariseinthiscontext, demonstratetheutilityofaccommodatingtheweightsinreconstructingtheunderlyinglow-rankrepresentation, andextendtheformulationtonon-
Gaussiannoisemodelssuchaslogisticmodels. Finally, weapplythemethodsdevelopedtoa collaborativefilteringtask.

## 1. IntroductionFactormodelsarenaturalintheanalysisofmanykindsoftabulateddata. Thisincludesuserpreferencesovera listofitems, microarray (geneexpression) measurements, andcollectionsofimages. Consider, forexample, adatasetofuserpreferencesformoviesorjokes. Thepremisebehinda factormodelisthatthereisonlya smallnumberof *factors* influencingthepreferences, andthata user'spreferencevectorisdeterminedbyhoweachfactorappliestothatuser. Ina linearfactormodel, eachfactorisa preferencevector, anda user'spreferencescorrespondtoa linearcombinationofthesefactorvectors, withuser-specificcoefficients. Thus, forn usersandd items, thepreferencesaccordingtoa k-factormodelaregivenbytheproductofann × k *coefficientmatrix* (eachrowrepresentingtheextenttowhicheachfactorisused) anda k × dfactormatrixwhoserowsarethefactors. Thepreferencematriceswhichadmitsucha factorizationarematricesofrankatmostk. Thus, trainingsucha linearfactormodelamountstoapproximatingtheempiricalpreferenceswitha low-rankmatrix.

Low-rankmatrixapproximationwithrespecttotheFrobeniusnorm-minimizingthesumsquareddifferencestothetargetmatrix-canbeeasilysolvedwithSingularValueDecomposition (SVD). Formanyapplications, however, thedeviationbetweentheobservedmatrixandthelow-rankapproximationshouldbemeasuredrelativetoa weighted (orother) norm. Whiletheextensiontotheweighted-normcaseisconceptuallystraightforward, anddatesbacktoearlyworkonfactoranalysis (Young, 1940), standardalgorithms (suchasSVD) forsolvingtheunweightedcasedonotcarryovertotheweightedcase. Weightednormscanariseina numberofsituations. Zero/oneweights, forexample, arisewhensomeoftheentriesinthematrixarenotobserved. Moregenerally, wemayintroduceweightsinresponsetosomeexternalestimateofthenoisevarianceassociatedwitheachmeasurement. Thisisthecase, forexample, ingeneexpressionanalysis, wheretheerrormodelformicroarraymeasurementsprovidesentry-specificnoiseestimates. Settingtheweightsinverselyproportionaltotheassumednoisevariancecanleadtoa betterreconstructionoftheunderlyingstructure. Inotherapplications, entriesinthetargetmatrixmayrepresentaggregatesofmanysamples. Thestandardunweightedlow-rankapproximation (e.g., forseparatingstyleandcontent (Tenenbaum & Freeman, 2000)) wouldinthiscontextassumethatthenumberofsamplesisuniformacrosstheentries. Non-uniformweightsareneededtoappropriatelycaptureanydifferencesinthesamplesizes. Despiteitsusefulness, theweightedextensionhasattractedrelativelylittleattention. Shpak (1990) andLuetal. (1997) studiedweighted-normlow-rankapproximationsforthedesignoftwo-dimensionaldigitalfilterswheretheweightsarisefromconstraintsofvaryingimportance. Shpakdevelopedgradient-basedoptimizationmethodswhileLuetal. suggestedalternatingoptimizationmethods. Inbothcases, rank-kapproximationsaregreedilycombinedfromk rank-oneapproximations. Unlikefortheunweightedcase, sucha greedyprocedureissub-optimal. Wesuggestoptimizationmethodsthataresignificantlymoreefficientandsimplertoimplement (Section2). Wealsoconsiderothermeasuresofdeviation, beyondweightedFrobeniusnorms. Suchmeasuresarise, forexample, whenthenoisemodelassociatedwithmatrixelementsisknownbutnotisGaussian. Forexample, forbinarydata, alogisticmodelwithanunderlyinglow-rankrepresentationmightbemoreappropriate. InSection3 weshowhowweighted-normapproximationproblemsariseassubroutinesforsolvingsucha low-rankproblem. Finally, inSection4, weillustratetheuseofthesemethodsbyapplyingthemtoa collaborativefilteringproblem.

## 2. WeightedLow-RankApproximationsGivena targetmatrixA ∈ <n×d, acorrespondingnonnegativeweightmatrixW ∈ <n×d
+ , anda desired (integer) rankk, wewouldliketofinda matrixX ∈ <n×dofrank (atmost) k, thatminimizestheweightedFrobeniusdistanceJ(X) = Pi,aWi,a (Xi,a − Ai,a)
2. Inthissection, weanalyzethisoptimizationproblemandconsideroptimizationmethodsforit.

#### 2.1. AMatrix-FactorizationViewItwillbeusefultoconsiderthedecompositionX =
UV0 whereU ∈ <n×kandV ∈ <d×k. Sinceanyrankkmatrixcanbedecomposedinsucha way, andanypairofsuchmatricesyieldsa rank-kmatrix, wecanthinkoftheproblemasanunconstrainedminimizationproblemoverpairsofmatrices (*U, V* ) withtheminimizationobjective

$$J(U,V)=\sum_{i,a}W_{i,a}\left(A_{i,a}-(UV^{\prime})_{i,a}\right)^{2}$$ $$=\sum_{i,a}W_{i,a}\left(A_{i,a}-\sum_{\alpha}U_{i,\alpha}V_{a,\alpha}\right)^{2}.$$

Thisdecompositionisnotunique. ForanyinvertibleR ∈ <k×k, thepair (*UR, VR*−1) providesa factorizationequivalentto (U, V ), i.e. J(*U, V* ) = J(*UR, VR*−1),
resultingina k2-dimensionalmanifoldofequivalentsolutions1. Inparticular, any (non-degenerate) solution
(*U, V* ) canbeorthogonalizedtoa (non-unique) equivalentorthogonalsolutionU¯ = UR, V¯ = VR−1suchthatV¯ 0V¯ = IandU¯0U¯ isa diagonalmatrix.2Wefirstrevisitthewell-studiedcasewhereallweightsareequaltoone. Itisa standardresultthatthelowrankmatrixminimizingtheunweightedsum-squareddistancetoA isgivenbytheleadingcomponentsofthesingularvaluedecompositionofA. Itwillbeinstructivetoconsiderthiscasecarefullyandunderstandwhytheunweightedlow-rankapproximationhassucha cleanandeasilycomputableform. Wewillthenbeabletomoveontotheweightedcase, andunderstandhow, andwhy, thesituationbecomeslessfavorable. Intheunweightedcase, thepartialderivativesoftheobjectiveJ withrespectto *U, V* are ∂J
∂U 
= 2(UV0 −
A)V ,
∂J
∂V = 2(VU0 − A0)U. Solving 
∂J
∂U = 0forU
yieldsU = AV (V0V )
−1; focusingonanorthogonalsolution, whereV
0V = IandU
0U = Λisdiagonal, yieldsU = AV . Substitutingbackinto ∂J
∂V = 0, wehave0 = VU0U − A0U = VΛ − A0AV . ThecolumnsofV aremappedbyA0Atomultiplesofthemselves, i.e. theyareeigenvectorsofA0A. Thus, thegradient
∂J
∂(U,V )
vanishesatanorthogonal (*U, V* ) ifandonlyifthecolumnsofV areeigenvectorsofA0AandthecolumnsofU arecorrespondingeigenvectorsofAA0, scaledbythesquarerootoftheireigenvalues. Moregenerally, thegradientvanishesatany (U, V ) ifandonlyifthecolumnsofU arespannedbyeigenvectorsofAA0andthecolumnsofV arecorrespondinglyspannedbyeigenvectorsofA0A. IntermsofthesingularvaluedecompositionA = U0SV0 0, thegradientvanishesat (U, V ) ifandonlyifthereexistmatricesQ0UQV = I ∈ <k×k(ormoregenerally, azero/onediagonalmatrixratherthanI) suchthatU = U0SQU , V = V0QV . Thisprovidesa completecharacterizationofthecriticalpointsofJ. Wenowturntoidentifyingtheglobalminimumandunderstandingthenatureoftheremainingcriticalpoints. Theglobalminimumcanbeidentifiedbyinvestigatingthevalueoftheobjectivefunctionatthecriticalpoints. Letσ1 *≥ · · · ≥* σmbetheeigenvaluesofA0A. Forcritical (*U, V* ) thatarespannedbyeigenvectorscorrespondingtoeigenvalues {σq|q ∈ Q}, theerrorofJ(*U, V* ) isgivenbythesumoftheeigenvaluesnotinQ (Pq6∈Qσq), andsotheglobalminimumisattainedwhentheeigenvectorscorrespondingtothehighesteigenvaluesaretaken. Aslongastherearenorepeatedeigenvalues, all (*U, V* ) globalminimacorrespondtothesamelow-rankmatrixX = UV0, andbelongtothesameequivalenceclass. 31Anequivalenceclassofsolutionsactuallyconsistsofa collectionofsuchmanifolds, asymptoticallytangenttooneanother.

2Weslightlyabusethestandardlinear-algebranotionof
"orthogonal" sincewecannotalwayshavebothU¯0U¯ = IandV¯ 0V¯ = I.

3Iftherearerepeatedeigenvalues, theglobalminimacorrespondtoa polytopeoflow-rankapproximationsinX space; in *U, V* space, theyforma collectionofhigherdimensionalasymptoticallytangentmanifolds.
Inordertounderstandthebehavioroftheobjectivefunction, itisimportanttostudytheremainingcriticalpoints. Fora criticalpoint (*U, V* ) spannedbyeigenvectorscorrespondingtoeigenvaluesasabove (assumingnorepeatedeigenvalues), theHessianhasexactlyPq∈Qq−k2negativeeigenvalues: wecanreplaceanyeigencomponentwitheigenvalueσ withanalternateeigencomponentnotalreadyin (*U, V* ) witheigenvalueσ 0 > σ, decreasingtheobjectivefunction. Thechangecanbedonegradually, replacingthecomponentwitha convexcombinationoftheoriginalandimprovedcomponents. Thisresultsina linebetweenthetwocriticalpointswhichisa monotonicimprovementpath. SincetherearePq∈Qq −k2suchpairsofeigencomponents, thereareatleastthismanydirectionsofimprovement. Otherthanthesedirectionsofimprovement, andthek 2directionsalongtheequivalencemanifoldcorrespondingtothek 2zeroeigenvaluesoftheHessian, allothereigenvaluesoftheHessianarepositive
(orzero, inverydegenerateA). Hence, intheunweightedcase, allcriticalpointsthatarenotglobalminimaaresaddlepoints. Thisisanimportantobservation: DespiteJ(*U, V* ) notbeinga convexfunction, allofitslocalminimaareglobal. Wenowmoveontotheweightedcase, andtrytotakethesamepath. Unfortunately, whenweightsareintroduced, thecriticalpointstructurechangessignificantly. Thepartialderivativesbecome (with ⊗ denotingelement-wisemultiplication):

$$\begin{array}{l}{{\frac{\partialJ}{\partialU}=2(W\otimes(UV^{\prime}-A))V}}\\ {{\frac{\partialJ}{\partialV}=2(W\otimes(VU^{\prime}-A^{\prime}))U}}\end{array}$$

Theequation ∂J
∂U = 0isstilla linearsysteminU, andfora fixedV , itcanbesolved, recoveringU
∗
V = argminUJ(U, V ) (sinceJ(*U, V* ) isconvexinU). However, thesolutioncannotbewrittenusinga singlepseudo-inverseV (V0V )
−1. Instead, aseparatepseudo-inverseisrequiredforeachrow (U
∗
V)iofU
∗
V:

$$\begin{array}{c}{{(U_{V}^{*})_{i}=(V^{\prime}\underline{{{W_{i}V}}})^{-1}V^{\prime}\underline{{{W_{i}A_{i}}}}}}\\ {{=\mathrm{pinv}(\sqrt{\underline{{{W_{i}V}}}})(\sqrt{\underline{{{W_{i}}}}}A_{i})}}\end{array}$$

whereWi ∈ <k×kisa diagonalmatrixwiththeweightsfromthei throwofW onthediagonal, andAiisthei throwofthetargetmatrix4. Inordertoproceedasintheunweightedcase, wewouldhavelikedtochooseV suchthatV
0WiV = I (orisatleastdiagonal). Thiscancertainlybedonefora singlei, butinordertoproceedweneedtodiagonalizeallV
0WiV *concurrently*.

WhenW isofrankone, suchconcurrentdiagonalizationispossible, allowingforthesamestructureasintheunweightedcase, andinparticularaneigenvectorbasedsolution (Irani & Anandan, 2000). However, forhigher-rankW, wecannotachievethisconcurrentlyforallrows. Thecriticalpointsoftheweightedlow-rankapproximationproblem, therefore, lacktheeigenvectorstructureoftheunweightedcase. Anotherimplicationofthisisthattheincrementalstructureofunweightedlow-rankapproximationsislost: anoptimalrank-kfactorizationcannotnecessarilybeextendedtoanoptimalrank-(k + 1) factorization. Lackingananalyticsolution, wereverttonumericaloptimizationmethodstominimizeJ(U, V ). ButinsteadofoptimizingJ(*U, V* ) bynumericallysearchingover (*U, V* ) pairs, wecantakeadvantageofthefactthatfora fixedV , wecancalculateU
∗
V, andthereforealsotheprojectedobjectiveJ
∗(V ) = minUJ(*U, V* ) =
J(U
∗
V
, V ). TheparameterspaceofJ
∗(V ) isofcoursemuchsmallerthanthatofJ(U, V ), makingoptimizationofJ
∗(V ) moretractable. ThisisespeciallytrueinmanytypicalapplicationswherethethedimensionsofA arehighlyskewed, withonedimensionseveralordersofmagnitudelargerthantheother (e.g. ingeneexpressionanalysisoneoftendealswiththousandsofgenes, butonlya fewdozenexperiments). RecoveringU
∗
Vusing (1) requiresn inversionsofk × kmatrices. Thedominatingfactorisactuallythematrixmultiplications: EachcalculationofV
0WiVrequiresO(dk2) operations, fora totalofO(ndk2) operations. Althoughmoreinvolvedthantheunweightedcase, thisisstillsignificantlylessthantheprohibitiveO(n3k3) requiredforeachiterationsuggestedbyLuetal. (1997), orforHessianmethodson (*U, V* ) (Shpak, 1990), andisonlya factorofk largerthantheO(ndk) requiredjusttocomputethepredictionUV0.

AfterrecoveringU
∗
V, wecaneasilycomputenotonlythevalueoftheprojectedobjective, butalsoitsgradient. Since 
∂J(V,U)
∂UU=U∗V
= 0, wehave

$$\frac{\partialJ^{*}(V)}{\partialV}=\left.\frac{\partialJ(V,U)}{\partialV}\right|_{U=U_{V}^{*}}=2(W\otimes(VU_{V}^{*\,\prime}-A^{\prime}))U_{V}^{*}.$$
$$\left(1\right)$$

ThecomputationrequiresonlyO(ndk) operations, andistherefore "free" afterU
∗
Vhasbeenrecovered.

Equippedwiththeabovecalculations, wecanusestandardgradient-descenttechniquestooptimizeJ
∗(V ).

Unfortunately, though, unlikeintheunweightedcase, J(U, V ), andJ
∗(V ), mighthavelocalminimathatarenotglobal. Figure1 showstheemergenceofa non-globallocalminimumofJ
∗(V ) fora rank-oneapproximationofA =11.11 −1
. ThematrixV isa twodimensionalvector. ButsinceJ
∗(V ) isinvariantunder4Hereandthroughoutthepaper, rowsofmatrices, suchasAiand (U
∗V )i, aretreatedinequationsas *column* vectors.
invertiblescalings, Vcanbespecifiedasanangleθ ona semi-circle. WeplotthevalueofJ
∗([cosθ,sinθ]) foreachθ, andforvaryingweightmatricesoftheformW =1+α1 11+α
. Atthefrontoftheplot, theweightmatrixisuniformandindeedthereisonlya singlelocalminimum, butatthebackoftheplot, wheretheweightmatrixemphasizesthediagonal, anon-globallocalminimumemerges.

![3_image_0.png](3_image_0.png)

Figure1. Emergenceoflocalminimawhentheweightsbecomenon-uniform.

Despitetheabundanceoflocalminima, wefoundgradientdescentmethodsonJ
∗(V ), andinparticularconjugategradientdescent, equippedwitha long-rangeline-searchforchoosingthestepsize, veryeffectiveinavoidinglocalminimaandquicklyconvergingtotheglobalminimum.

### 2.2. AMissing-ValuesViewAndAnEmProcedureInthissectionwepresentanalternativeoptimizationprocedure, whichismuchsimplertoimplement. Thisprocedureisbasedonviewingtheweightedlowrankapproximationproblemasa maximum-likelihoodproblemwithmissingvalues. Considerfirstsystemswithonlyzero/oneweights, whereonlysomeoftheelementsofthetargetmatrixA areobserved (thosewithweightone) whileothersaremissing (thosewithweightzero). Referringtoa probabilisticmodelparameterizedbya low-rankmatrixX, whereA = X + ZandZ iswhiteGaussiannoise, theweightedcostofX isequivalenttothelog-likelihoodoftheobservedvariables.

ThissuggestsanExpectation-Maximizationprocedure. IneachEMupdatewewouldliketofinda newparametermatrixmaximizingtheexpectedloglikelihoodofa filled-inA, wheremissingvaluesarefilledinaccordingtothedistributionimposedbythecurrentestimateofX. Thismaximum-likelihoodparametermatrixisthe (unweighted) low-rankapproximationofthemeanfilled-inA, whichisA withmissingvaluesfilledinfromX. Tosummarize: intheExpectationstepvaluesfromthecurrentestimateofX arefilledinforthemissingvaluesinA, andintheMaximizationstepX isreestimatedasa low-rankapproximationofthefilled-inA. Inordertoextendthisapproachtoa generalweightmatrix, considera probabilisticsystemwithseveraltargetmatrices, A(1), A(2)*, . . . , A*(N), butwitha singlelow-rankparametermatrixX, whereA(r) = X + Z(r) andtherandommatricesZ(r) areindependentwhiteGaussiannoisewithfixedvariance. Whenalltargetmatricesarefullyobserved, themaximumlikelihoodsettingforX isthelow-rankapproximationofthetheiraverage. Now, ifsomeoftheentriesofsomeofthetargetmatricesarenotobserved, wecanusea similarEMprocedure, whereintheexpectationstepvaluesfromthecurrentestimateofX arefilledinforallmissingentriesinthetargetmatrices, andinthemaximizationstepX isupdatedtobea low-rankapproximationofthemeanofthefilled-intargetmatrices. Toseehowtousetheaboveproceduretosolveweightedlow-rankapproximationproblems, considersystemswithweightslimitedtoWia =
wiaN withintegerwia ∈ {0, 1*, . . . , N*}. Sucha low-rankapproximationproblemcanbetransformedtoa missingvalueproblemoftheformaboveby "observing" thevalueAiainwiaofthetargetmatrices (foreachentry *i, a*),
andleavingtheentryasmissingintherestofthetargetmatrices. TheEMupdatethenbecomes:

$$X^{(t+1)}=\mathrm{LRA}_{k}\left(W\otimesA+({\bf1}-W)\otimesX^{(t)}\right)\quad(2)$$

whereLRAk(X) istheunweightedrank-kapproximationofX, ascanbecomputedfromtheSVD. NotethatthisprocedureisindependentofN. Foranyweightmatrix (scaledtoweightsbetweenzeroandone) theprocedureinequation (2) canthusbeseenasanexpectation-maximizationprocedure. Thisprovidesfora verysimple, tweaking-freemethodforfindingweightedlow-rankapproximations. AlthoughwefoundthisEM-inspiredmethodeffectiveinmanycases, insomeothercasestheprocedureconvergestoa localminimumwhichisnotglobal. Sincethemethodiscompletelydeterministic, initializationofX playsa crucialroleinpromotingconvergencetoa global, oratleastdeeplocal, minimum, aswellasthespeedwithwhichconvergenceisattained. TwoobviousinitializationmethodsaretoinitializeX
toA, andtoinitializeX tozero. InitializingX toA worksreasonablywelliftheweightsareboundedawayfromzero, orifthetargetvaluesinA haverelativelysmallvariance. However, whentheweightsarezero, orveryclosetozero, thetargetvaluesbecomemeaningless, andcanthrowoffthesearch. InitializingX tozeroavoidsthisproblem, astargetvalueswithzeroweightsarecompletelyignored (astheyshouldbe), andworkswellaslongastheweightsarefairlydense. However, whentheweightsaresparse, itoftenconvergestolocalminimawhichconsistentlyunderpredictthemagnitudeofthetargetvalues. Asanalternativetotheseinitializationmethods, wefoundthefollowingprocedureveryeffective: weinitializeX tozero, butinsteadofseekinga rank-kapproximationrightaway, westartwitha fullrankmatrix, andgraduallyreducetherankofourapproximations. Thatis, thefirstd − kiterationstaketheform:

$$X^{(t+1)}=\mathrm{LRA}_{d-t}\left(W\otimesA+({\bf1}-W)\otimesX^{(t)}\right),\ \ (3)$$

resultinginX(t) ofrank (d−t+1). Afterreachingrankk, werevertbacktotheiterationsofequation (2) untilconvergence. NotethatwithiterationsoftheformX(t+1) = W ⊗A*+ (1*−W)⊗X(t), withoutrankreductions, wewouldhaveX
(t)
ia = (1 − (1 − Wia)
t))Aia →
(1 − e
−tWia )Aia, whichconvergesexponentiallyfasttoA forpositiveweights. Ofcourse, becauseoftherankreduction, thisdoesnothold, buteventhefewhigh-rankiterationssetvalueswithweightsawayfromzeroclosetotheirtargetvalues, aslongastheydonotsignificantlycontradictothervalues.

### 2.3. ReconstructionExperimentsSincetheunweightedorsimplelow-rankapproximationproblempermitsa closed-formsolution, onemightbetemptedtousesucha solutioneveninthepresenceofnon-uniformweights (i.e., ignoretheweights). Wedemonstrateherethatthisprocedureresultsina substantiallossofreconstructionaccuracyascomparedtotheEMalgorithmdesignedfortheweightedproblem. Tothisend, wegenerated1000 × 30lowrankmatricescombinedwithGaussiannoisemodelstoyieldtheobserved (target) matrices. Foreachmatrixentry, thenoisevarianceσ 2iawaschosenuniformlyinsomenoiselevelrangecharacterizedbya *noisespreadratio* maxσ 2/ minσ 2. Theplantedmatrixwassubsequentlyreconstructedusingbotha weightedlow-rankapproximationwithweightsWia = 1/σ2ia, andanunweightedlow-rankapproximation (usingSVD). Thequalityofreconstructionwasassessedbyanunweightedsquareddistancefromthe "planted" matrix.

![4_image_0.png](4_image_0.png)

Figure2. Reconstructionofa 1000×30rank-threematrix. Left: (a) weightedandunweightedreconstructionwitha noisespreadof100 ; right: (b) reductioninreconstructionerrorforvariousnoisespreads.

Figure2(a) showsthequalityofreconstructionattainedbythetwoapproachesasa functionofthesignal (weightedvarianceofplantedlow-rankmatrix) tonoise (averagenoisevariance) ratio, fora noisespreadratioof100 (correspondingtoweightsintherange0.01-1). Thereconstructionerrorattainedbytheweightedapproachisgenerallyovertwentytimessmallerthantheerroroftheunweightedsolution. Figure2(b) showsthisimprovementinthereconstructionerror, intermsoftheerrorratiobetweentheweightedandunweightedsolutions, forthedatainFigure2(a),
aswellasforsmallernoisespreadratiosoftenandtwo. Evenwhenthenoisevariances (andhencetheweights) arewithina factoroftwo, westillseea consistenttenpercentimprovementinreconstruction. Theweightedlow-rankapproximationsinthisexperimentwerecomputedusingtheEMalgorithmofSection2.2. Fora widenoisespread, whenthelowrankmatrixbecomesvirtuallyundetectable (asignalto-noiseratiowellbelowone, andreconstructionerrorsinexcessofthevarianceofthesignal), EMoftenconvergestoa non-globalminimum. Thisresultsinweightedlow-rankapproximationswitherrorsfarhigherthancouldotherwisebeexpected, ascanbeseeninbothfigures. Insuchsituations, conjugategradientdescentmethodsprovedfarsuperiorinfindingtheglobalminimum.

## 3. Low-RankLogisticRegressionIncertainsituationswemightliketocapturea binarydatamatrixy *∈ {−*1, +1}
n×dwitha low-rankmodel.

Anaturalchoiceinthiscaseisa logisticmodelparameterizedbya low-rankmatrixX ∈ <n×d, suchthatPr (Yia = +1|Xia) = g(Xia) independentlyforeachi, a, whereg isthelogisticfunctiong(x) = 11+e−x . Onethenseeksa low-rankmatrixX maximizingthelikelihoodPr (Y = y|X). Suchlow-ranklogisticmodelsweresuggestedbyCollinsetal. (2002) andbyGordon
(2003) andrecentlystudiedbyScheinetal. (2003). Usinga weightedlow-rankapproximation, wecanfita low-rankmatrixX minimizinga quadraticlossfromthetarget. Inordertofita non-quadraticlosssuchasa logisticloss, Loss(Xia; yia) = logg(yiaXia), weusea quadraticapproximationtotheloss. Considerthesecond-orderTaylorexpansionoflogg(yx) about ˜x:

$$\logg(yx)\approx$$
 Let $\bar{x}$ bea positiveinteger, $\approx\logg(y\bar{x})+yg(-y\bar{x})(x-\bar{x})-\frac{g(y\bar{x})g(-y\bar{x})}{2}\left(x-\bar{x}\right)^2$ $\approx-\frac{g(y\bar{x})g(-y\bar{x})}{2}\left(x-\left(\bar{x}+\frac{y}{g(y\bar{x})}\right)\right)^2+\logg(y\bar{x})+\frac{g(-y\bar{x})}{2g(y\bar{x})}$. 
.
Thelog-likelihoodofa low-rankparametermatrixX canthenbeapproximatedas:

$$\log\operatorname*{Pr}\left(y|X\right)\approx1$$
$$\log\Pr\left(y|X\right)\approx$$ $$-\sum_{ia}\frac{g(y_{ia}\tilde{X}_{ia})g(-y_{ia}\tilde{X}_{ia})}{2}\left(X_{ia}-\left(\tilde{X}_{ia}+\frac{y_{ia}}{g(y_{ia}\tilde{X}_{ia})}\right)\right)^{2}$$ $$+\mbox{Const}\quad\mbox{(4)}$$

Maximizing (4) isa weightedlow-rankapproximationproblem. Notethatforeachentry (*i, a*), weusea second-orderexpansionabouta *different* pointX˜ia. TheclosertheoriginX˜iaistoXia, thebettertheapproximation. Thissuggestsaniterativeapproach, whereineachiterationwefinda parametermatrixX usinganapproximationofthelog-likelihoodabouttheparametermatrixfoundinthepreviousiteration. FortheTaylorexpansion, theimprovementoftheapproximationisnotalwaysmonotonic. Thismightcausethemethodoutlinedabovenottoconverge. Inordertoprovidefora morerobustmethod, weusethefollowingvariationalboundonthelogistic (Jaakkola & Jordan, 2000):

$\begin{array}{c}\logg(yx)\geq\logg(y\bar{x})+\frac{yx-y\bar{x}}{2}-\frac{\tanh(\bar{x}/2)}{4\bar{x}}\left(x^{2}-\bar{x}^{2}\right)\\ =-\frac{1}{4}\frac{\tanh(\bar{x}/2)}{\bar{x}}\left(x-\frac{y\bar{x}}{\tanh(\bar{x}/2)}\right)+\text{Const},\end{array}$ yieldingthecorrespondingboundonthelikelihood:
$$\log\Pr\left(y|X\right)\geq$$ $$-\frac{1}{4}\sum_{ia}\frac{\tanh(\bar{X}_{ia}/2)}{\bar{X}_{ia}}\left(X_{ia}-\frac{y_{ia}\bar{X}_{ia}}{\tanh(\bar{X}_{ia}/2)}\right)+\mbox{Const}\tag{5}$$
withequalityifandonlyifX = X˜. ThisboundsuggestsaniterativeupdateoftheparametermatrixX(t) byseekinga low-rankapproximationX(t+1) forthefollowingtargetandweightmatrices:

A (t+1) ia = yia/W(t+1) iaW (t+1) ia = tanh(X (t) ia /2)/X(t) ia
$$(6)$$
Fortunately, wedonotneedtoconfrontthesevereproblemsassociatedwithnestingiterativeoptimizationmethods. Inordertoincreasethelikelihoodofourlogisticmodel, wedonotneedtofinda lowrankmatrixminimizingtheobjectivespecifiedby (6),
justoneimprovingit. Anylow-rankmatrixX(t+1) witha lowerobjectivevaluethanX(t)(withrespecttoA(t+1) andW(t+1)) isguaranteedtohavea higherlikelihood: Alowerobjectivecorrespondstoa higherupperboundin (5), andsincetheboundistightforX(t), thelog-likelihoodofX(t+1) mustbehigherthanthelog-likelihoodofX(t). Moreover, ifthelikelihoodofX(t)isnotalreadymaximal, thereareguaranteedtobematriceswithlowerobjectivevalues. Therefore, wecanmixweightedlow-rankapproximationiterationsandlogisticboundupdateiterations, whilestillensuringconvergence.

Inmanyapplicationswemayalsowanttoassociateexternalweightswitheachentryinthematrix (e.g. toaccommodatemissingvalues), ormoregenerally, weights (counts) ofpositiveandnegativeobservationsineachentry (e.g. tocapturethelikelihoodwithrespecttoanempiricaldistribution). Thiscaneasilybedonebymultiplyingtheweightsin (6) bytheexternalweights, ortakinga weightedcombinationcorrespondingtoy = +1andy = −1. NotethatthetargetandweightmatricescorrespondingtotheTaylorapproximationandthosecorrespondingtothevariationalboundaredifferent: ThevariationaltargetisalwaysclosertothecurrentvalueofX, andtheweightsaremoresubtle. Thisensurestheguaranteedconvergence (asdiscussedabove), butthepricewepayisa muchlowerconvergencerate. Althoughwehaveobservedmanyinstancesinwhicha 'Taylor' iterationincreases, ratherthendecreases, theobjective, overallconvergencewasattainedmuchfasterusing 'Taylor', ratherthan 'variational' iterations.

## 4. ACollaborativeFilteringExampleToillustratetheuseofweighted, andgeneralized, lowrankapproximations, weappliedourmethodstoa collaborativefilteringproblem. Thetaskofcollaborativefilteringis, givensomeentriesofa userpreferencesmatrix, topredicttheremainingentries. Wedothisbyapproximatingthoseobservedvaluesbya low-rankmatrix (usingweightedlow-rankapproximationwithzero/oneweights). Unobservedvaluesarepredictedaccordingtothelearnedlow-rankmatrix. Usinglow-rankapproximationforcollaborativefilteringhasbeensuggestedinthepast. Goldbergetal. (2001) usea low-rankapproximationofa fullyobservedsubsetofcolumnsofthematrix, thusavoidingtheneedtointroduceweights. BillsusandPazzani (1998) usea singularvaluedecompositionofa sparsebinaryobservationmatrix. BothGoldbergandBillsususethelow-rankapproximationonlyasa preprocessingstep, andthenuseclustering (Goldberg) andneuralnetworks (Billsus) tolearnthepreferences. Azaretal. (2001) provedasymptoticconsistencyofa methodinwhichunobservedentriesarereplacedbyzeros, andobservedentriesarescaledinverselyproportionallytotheprobabilityofthembeingobserved. Noguaranteesareprovidedforfinitedatasets, andtothebestofourknowledgethistechniquehasnotbeenexperimentallytested.

Weanalyzeda subsetoftheJesterdata5(Goldbergetal., 2001). Thedatasetcontainsonehundredjokes, withuserratings (boundedcontinuousvaluesenteredbyclickinganon-screen "funniness" bar) forsomeofthejokes. Allusersrateda coresetoftenjokes, andmostusersratedanextendedcoresetofa totaloftwentyjokes. Eachuseralsorateda variablenumberofadditionaljokes. Weselectedatrandomonethousanduserswhoratedtheextendedcoresetandatleasttwoadditionaljokes. Foreachuser, weselectedatrandomtwonon-corejokesandheldouttheirratings. Wefitlow-rankmatricesusingthefollowingtechniques: svdUnobservedvalueswerereplacedwithzeros, andtheunweightedlow-rankapproximationtotheresultingmatrixwassought.

subsetAnunweightedlow-rankapproximationforthecoresubsetofjokeswassought (similarlytoGoldberg'sinitialstep). Thematrixwasextendedtotheremainingjokesbyprojectingeachjokecolumnontothecolumnsubspaceofthismatrix.

rescalingFollowingAzaretal. (2001), theratingsforeachjokewerescaledinverselyproportionaltothefrequencywithwhichthejokewasrated (between0.197and0.77). Anunweightedlowrankapproximationfortheresultingmatrixwassought.

wlraA weightofonewasassignedtoeachobservedjoke, anda weightofzerotoeachunobservedjoke, anda weightedlow-rankapproximationwassoughtusinggradientdescenttechniques.

Foreachlow-rankmatrix, thetesterrorontheheldoutjokes (Figure3) andthetrainingerrorweremeasuredintermsoftheaveragesquareddifferencetothetruerating, scaledbythepossiblerangeofratings. Normalizedmeanabsoluteerror (NMAE) wasalsomeasured, producingverysimilarresults, withnoqualitativedif-

![6_image_0.png](6_image_0.png)

Figure3. PredictionerrorsonJesterjokes: testerror (mainfigure) andtrainingerror (insert).

ferences. Beyondtheconsistentreductionintrainingerror (whichisguaranteedbytheoptimizationobjective), weobservethatwlraachievesa bettertesterrorthananyoftheothermethods. Notsurprisingly, italsoover-fitsmuchmorequickly, asitbecomespossibletoapproximatetheobservedvaluesbetterattheexpenseofextremevaluesintheotherentries.

![6_image_1.png](6_image_1.png)

Figure4. Training (dottedlines) andtestperformanceonJesterjokes.

Asdiscussedintheintroduction, minimizingthesquarederrortotheabsoluteratingsisnotnecessarilythecorrectobjective. Takingtheviewthateachjokehasa 'probabilityofbeingfunny' foreachuser, weproceededtotrytofita low-ranklogisticregressionmodel. Wefirsttransformedtherawobservedvaluesinto 'funniness' probabilitiesbyfittinga mixturemodelwithtwoequal-varianceGaussiancomponentstoeachuser'sratings, andusingtheresultingcomponent-posteriorprobabilities. Thisprocedure5ThedatasetwaskindlyprovidedbyKenGoldberg.
alsoensuresscaleandtransformationinvariabilityfora user'sratings, andplacesmoreemphasisonuserswitha bimodalratingdistributionthanonusersforwhichallratingsareclusteredtogether. Weproceededtofita low-ranklogisticmodel (q.v. Section3) usingtheobservedposteriorprobabilitiesasempiricalprobabilities. Sincetheresultinglow-rankmodelnolongerpredictstheabsoluteratingofjokes, wemeasuredsuccessbyanalyzingtherelativerankingofjokesbyeachuser. Specifically, foreachuserweheldoutonenoncorejokewhichwasratedamongthetopquarterbytheuser, andonenon-corejokewhichwasratedinthebottomquarter. Wethenmeasuredthefrequencywithwhichtherelativerankingsofthepredictionsonthesetwojokeswasconsistentwiththetruerelativeranking. Usingthismeasure, wecomparedthelogisticlow-rankmodeltothesum-squarederrormethodsdiscussedabove, appliedtoboththeabsoluteratings (asabove) andtheprobabilities. Figure4 showsthetrainingandtestperformanceofthelogisticmethod, thewlramethodappliedtotheratings, thewlramethodappliedtotheprobabilities, andthesvdmethodappliedtotheratings (allothermethodstestedperformworsethanthoseshown). Althoughtheresultsindicatethatthewlramethodperformsbetterthanthelogisticmethod, itisinterestingtonotethatforsmallranks, k = 2, 3, thetrainingperformanceofthelogisticmodelisbetter-inthesecasesthelogisticviewallowsustobettercapturetherankingsthana sumsquared-errorview (Scheinetal. (2003) investigatesthetrainingerrorofotherdatasets, andarrivesatsimilarconclusions). Apossiblemodificationtothelogisticmodelthatmightmakeitmoresuitableforsuchtasksistheintroductionoflabelnoise.

## 5. ConclusionWehaveprovidedsimpleandefficientalgorithmsforsolvingweightedlow-rankapproximationproblems. TheEMalgorithmisextremelysimpletoimplement, andworkswellinsomecases. Inmorecomplexcases, conjugategradientdescentonJ
∗(V ) providesefficientconvergence, usuallytotheglobalminimum. Weightedlow-rankapproximationproblemsareimportantintheirownrightandappearassubroutinesinsolvinga classofmoregenerallow-rankproblems. Onesuchproblem, fittinga low-ranklogisticmodel, wasdevelopedinthispaper. Similarapproachescanbeusedforotherconvexlossfunctionswitha boundedHessian. Anotherclassofproblemsthatwecansolveusingweightedlow-rankapproximationasa subroutineislow-rankapproximationwithrespecttoa mixtureof-Gaussiansnoisemodel. Thisapplicationwillbe

## ReferencesAzar, Y., Fiat, A., Karlin, A. R., McSherry, F., & Saia, J. (2001). Spectralanalysisofdata. ProceedingsoftheThirtyThirdACMSymposiumonTheoryofComputing.

Billsus, D., & Pazzani, M. J. (1998). Learningcollaborativeinformationfilters. *Proceedingsof15th* InternationalConferenceonMachineLearning.

Collins, M., Dasgupta, S., & Schapire, R. (2002). Ageneralizationofprincipalcomponentanalysistotheexponentialfamily. AdvancesinNeuralInformationProcessingSystems14.

Goldberg, K., Roeder, T., Gupta, D., & Perkins, C.

(2001). Eigentaste: Aconstanttimecollaborativefilteringalgorithm. *InformationRetrieval*, 4, 133- 151.

Gordon, G. (2003). Generalized2linear2models. AdvancesinNeuralInformationProcessingSystems15.

Irani, M., & Anandan, P. (2000). Factorizationwithuncertainty. ProceedingsoftheSixthEuropeanConferenceonComputerVision.

Jaakkola, T., & Jordan, M. (2000). Bayesianparameterestimationviavariationalmethods. StatisticsandComputing, 10, 25-37.

Lu, W.-S., Pei, S.-C., & Wang, P.-H. (1997). Weightedlow-rankapproximationofgeneralcomplexmatricesanditsapplicationinthedesignof2-Ddigitalfilters. IEEETransactionsonCircuitsandSystems-I, 44, 650-655.

Schein, A. I., Saul, L. K., & Ungar, L. H. (2003).

Ageneralizedlinearmodelforprincipalcomponentanalysisofbinarydata. ProceedingsoftheNinthInternationalWorkshoponArtificialIntelligenceandStatistics.

Shpak, D. (1990). Aweighted-least-squaresmatrixdecompositionmethodwithapplicationtothedesignoftwo-dimensionaldigitalfilters. IEEEThirtyThirdMidwestSymposiumonCircuitsandSystems.

Tenenbaum, J. B., & Freeman, W. T. (2000). Separatingstyleandcontentwithbilinearmodels. *Neural* Computation, 12, 1247-1283.

Young, G. (1940). Maximumlikelihoodestimationandfactoranalysis. *Psychometrika*, 6, 49-53.