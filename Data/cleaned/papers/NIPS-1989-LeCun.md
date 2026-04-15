# OptimalBrainDamageYannLeCun, JohnS. DenkerandSaraA. Sol1aAT&TBellLaboratories, Holmdel, N. J. 07733 

# AbstractWehaveusedinformation-theoreticideastoderivea classofpracticalandnearlyoptimalschemesforadaptingthesizeofa neuralnetwork. Byremovingunimportantweightsfroma network, severalimprovementscanbeexpected: bettergeneralization, fewertrainingexamplesrequired, andimprovedspeedoflearningand/orclassification. Thebasicideaistousesecond-derivativeinformationtomakea tradeoffbetweennetworkcomplexityandtrainingseterror. Experimentsconfirmtheusefulnessofthemethodsona real-worldapplication. 

# 1IntroductionMostsuccessfulapplicationsofneuralnetworklearningtoreal-worldproblemshavebeenachievedusinghighlystructurednetworksofratherlargesize [forexample (Waibel, 1989; LeCunetal., 1990a)]. Asapplicationsbecomemorecomplex, thenetworkswillpresumablybecomeevenlargerandmorestructured. Designtoolsandtechniquesforcomparingdifferentarchitecturesandminimizingthenetworksizewillbeneeded. Moreimportantly, asthenumberofparametersinthesystemsincreases, overfittingproblemsmayarise, withdevastatingeffectsonthegeneralizationperformance. Weintroducea newtechniquecalledOptimalBrainDamage 
(OBD) forreducingthesizeofa learningnetworkbyselectivelydeletingweights. 

WeshowthatOBDcanbeusedbothasanautomaticnetworkminimizationprocedureandasaninteractivetooltosuggestbetterarchitectures. 

ThebasicideaofOBDisthatitispossibletotakea perfectlyreasonablenetwork, deletehalf (ormore) oftheweightsandwindupwitha networkthatworksjustaswell, orbetter. Itcanbeappliedinsituationswherea complicatedproblemmustbesolved, andthesystemmustmakeoptimaluseofa limitedamountoftrainingdata. Itisknownfromtheory (Denkeretal., 1987; BaumandHaussler, 1989; Sollaetal., 1990) andexperience (LeCun, 1989) that, fora fixedamountoftrainingdata, networkswithtoomanyweightsdonotgeneralizewell. Ontheotherhand. networkswithtoofewweightswillnothaveenoughpowertorepresentthedataaccurately. Thebestgeneralizationisobtainedbytradingoffthetrainingerrorandthenetworkcomplexity. 

Onetechniquetoreachthistradeoffistominimizea costfunctioncomposedoftwoterms: theordinarytrainingerror, plussomemeasureofthenetworkcomplexity. 

Severalsuchschemeshavebeenproposedinthestatisticalinferenceliterature [see (Akaike, 1986; Rissanen, 1989; Vapnik, 1989) andreferencestherein] aswellasintheNNliterature (Rumelhart, 1988; Chauvin, 1989; HansonandPratt, 1989; MozerandSmolensky, 1989). Variouscomplexitymeasureshavebeenproposed, includingVapnik-Chervonenkisdimensionality (VapnikandChervonenkis, 1971) anddescriptionlength (Rissanen, 1989). Atime-honored (albeitinexact) measureofcomplexityissimplythenumberofnon-zerofreeparameters, whichisthemeasurewechoosetouseinthispaper [butsee (Denker, LeCunandSolla, 1990)]. Freeparametersareusedratherthanconnections, sinceinconstrainednetworks, severalconnectionscanbecontrolledbya singleparameter. Inmostcasesinthestatisticalinferenceliterature, thereissome *apriori* orheuristicinformationthatdictatestheorderinwhichparametersshouldbedeleted; forexample, ina familyofpolynomials, asmoothnessheuristicmayrequirehigh-ordertermstobedeletedfirst. Ina neuralnetwork, however, itisnotatallobviousinwhichordertheparametersshouldbedeleted. 

Asimplestrategyconsistsindeletingparameterswithsmall "saliency", i.e. thosewhosedeletionwillhavetheleasteffectonthetrainingerror. Otherthingsbeingequal, small-magnitudeparameterswillhavetheleastsaliency, soa reasonableinitialstrategyistotrainthenetworkanddeletesmall-magnitudeparametersinorder. Afterdeletion, thenetworkshouldberetrained. Ofcoursethisprocedurecanbeiterated; inthelimititreducestocontinuousweight-decayduringtraining 
(usingdisproportionatelyrapiddecayofsmall-magnitudeparameters). Infact, severalnetworkminimizationschemeshavebeenimplementedusingnon-proportionalweightdecay (Rumelhart, 1988; Chauvin, 1989; HansonandPratt, 1989), or "gatingcoefficients" (MozerandSmolensky, 1989). Generalizationperformancehasbeenreportedtoincreasesignificantlyonthesomewhatsmallproblemsexamined. 

Twodrawbacksofthesetechniquesarethattheyrequirefine-tuningofthe "pruning" coefficientstoavoidcatastrophiceffects, andalsothatthelearningprocessissignificantlysloweddown. Suchmethodsincludetheimplicithypothesisthattheappropriatemeasureofnetworkcomplexityisthenumberofparameters (orsometimesthenumberofunits) inthenetwork. Oneofthemainpointsofthispaperistomovebeyondtheapproximationthat 
"magnitudeequalssaliency" , andproposea theoreticallyjustifiedsaliencymeasure. 

Ourtechniqueusesthesecondderivativeoftheobjectivefunctionwithrespecttotheparameterstocomputethesaliencies. Themethodwas ,,-alidatedusingourhandwrittendigitrecognitionnetworktrainedwithbackpropagation (LeCunetaI., 
1990b). 

# 2OptimalBrainDamageObjectivefunctionsplayacentralroleinthisfield; thereforeitismorethanreasonabletodefinethesaliencyofa parametertobethechangeintheobjectivefunctioncausedbydeletingthatparameter. Itwouldbeprohibiti,-elylaborioustoevaluatethesaliencydirectlyfromthisdefinition, i.e. bytemporarilydeletingeachparameterandreevaluatingtheobjectivefunction. 

Fortunately, itispossibletoconstructa localmodeloftheerrorfunctionandanalyticallypredicttheeffectofperturbingtheparametervector. "'eapproximatetheobjectivefunctionE bya Taylorseries. AperturbationlL~ oftheparametervectorwillchangetheobjectivefunctionby 

$$\deltaE=\sum_{i}g_{i}\deltau_{i}+\frac{1}{2}\sum_{i}h_{ii}\deltau_{i}^{2}+\frac{1}{2}\sum_{ij\neqj}h_{ij}\deltau_{i}\deltau_{j}+O(||\delta^{*}||^{3})\tag{1}$$

Here, the6ui'SarethecomponentsofflJ, thegi'sarethecomponentsofthegradientG ofE withrespecttoU, andthe *hi;'S* aretheelementsoftheHessianmatrixH ofE withrespecttoU: 

$$g_{i}=\frac{\partialE}{\partialu_{i}}\quad\quad\mbox{and}\quad\quadh_{ij}=\frac{\partial^{2}E}{\partialu_{i}\partialu_{j}}\tag{2}$$

Thegoalistofinda setofparameterswhosedeletionwillcausetheleastincreaseofE . Thisproblemispracticallyinsolubleinthegeneralcase. OnereasonisthatthematrixH isenormous (6.5x 106termsforour2600parameternetwork), andisverydifficulttocompute. Thereforewemustintroducesomesimplifyingapproximations. The "diagonal" approximationassumesthatthe6Ecausedbydeletingseveralparametersisthesumofthe *6E's* causedbydelet~ eachparameterindividually; crosstermsareneglected, sothirdtermofthenpthandsideofequation1 isdiscarded. The "extremal" approximationassumesthatparameterdeletionwillbeperformedaftertraininghasconverged. Theparametervectoristhenata (local) minimumofE andthefirsttermoftherighthandsideofequation1 canbeneglected. Furthermore, ata localminimum, allthehii'sarenon-negative, soanyperturbationoftheparameterswillcauseE toincreaseorstaythesame. 

Thirdly, the "quadratic" approximationassumesthatthecostfundionisnearlyquadratic80thatthelasttermintheequationcanbeneglected. Equation1 thenreducesto 

$\deltaE=\frac{1}{2}\sum_{i}h_{ii}\deltaa_{i}^{2}$
$$\left({\mathfrak{3}}\right)$$

#### 2.1ComputingTheSecondDerivativesNowweneedanefficientwayofcomputingthediagonalsecondderivativeshii . 

Sucha procedurewasderivedin (LeCun, 1987), andwasthebasisofa fastbackpropagationmethodusedextensivelyin \1lriousapplications (BeckerandLeCun, 1989; LeCun, 1989; LeCunetal., 1990a). Theprocedureisverysimilartotheback-propagationalgorithmusedforcomputingthefirstderivatives. Wewillonlyoutlinetheprocedure; detailscanbefoundinthereferences. Weassumetheobjectivefunctionistheusualmean-squarederror (MSE); generalizationtootheradditiveerrormeasuresisstraightforward. Thefollowingexpressionsapplytoa singleinputpattern; afterwardE andH mustbeaveragedoverthetrainingset. Thenetworkstateiscomputedusingthestandardformulae 

$x_{i}=f(a_{i})$ and $a_{i}=\sum_{j}w_{ij}x_{j}$
$$(4)$$
whereZiisthestateofuniti, aiitstotalinput (weightedsum), ! thesquashingfunctionandWijistheconnectiongoingfromunitj touniti. Ina shared-weightnetworklikeours, asingleparameterUkcancontroloneormoreconnections: Wij = 
Ukforall (i, j) E *Vk,* whereVkisa setofindexpairs. Bythechainrule, thediagonaltermsofH aregivenby 

$h_{kk}=\sum_{i}\frac{\partial^{2}E}{\partialw_{ij}^{2}}$ (i,j)$\epsilonV_{k}$
$$\left(5\right)$$
Thesummandcanbeexpanded (usingthebasicnetworkequations4) as: 

{J2ElPE 2 --=-z· {Jw~. ., {Ja~' . 
$$({\mathfrak{h}})$$
Thesecondderivativesareback-propagatedfromlayertolayer: 

$$\left(7\right)$$

Wealsoneedtheboundaryconditionattheoutputlayer, specifyingthesecondderivativeofE withrespecttothelast-layerweightedBUms: 

$\partial^{2}E=f^{\prime}(a_{i})^{2}\sum_{i}w_{ii}^{2}\frac{\partial^{2}E}{\partiala_{i}^{2}}-f^{\prime\prime}(a_{i})\frac{\partialE}{\partialx_{i}}$
$\partial^{2}E=2f^{\prime}(a_{i})^{2}-2(d_{i}-x_{i})f^{\prime\prime}(a_{i})$
$$(\mathbf{a})$$
forallunitsi intheoutputlayer. 

Ascanbeseen, computingthediagonalHessianisofthesameorderofcomplexityascomputingthegradient. Insomecases, thesecondtermoftherighthandsideofthelasttwoequations (involvingthesecondderivativeofI) canbeneglected. Thiscorrespondstothewell-knownLevenberg-Marquardtapproximation, andhastheinterestingpropertyofgivingguaranteedpositiveestimatesofthesecondderivative. 

#### 2.2TheRecipeTheOBOprocedurecanbecarriedoutasfollows: 
1. Choosea reasonablenetworkarchitecture2. Trainthenetworkuntila reasonablesolutionisobtained3. Computethesecondderivativeshuforeachparameter4. Computethesalienciesforeachparameter: Sk = huu~/25. Sorttheparametersbysaliencyanddeletesomelow-saliencyparameters6. Iteratetostep2 Deletinga parameterisdefinedassettingitto0 andfreezingitthere. Severalvariantsoftheprocedurecanbedevised, suchasdecreasingthe ... 41uesofthelowsaliencyparametersinsteadofsimplysettingthemto0, orallowingthedeletedparameterstoadaptagainaftertheyhavebeensettoo. 

#### 2.3ExperimentsThesimulationresultsgiveninthissectionwereobtainedusingback-propagationappliedtohandwrittendigitrecognition. Theinitialnetworkwashighlyconstrainedandsparselyconnected, having105connectionscontrolledby2578freeparameters. Itwastrainedona databaseofsegmentedhandwrittenzipcodedigitsandprinteddigitscontainingapproximately9300trainingexamplesand3350t.estexamples. 

Moredetailscanbeobtainedfromthecompanionpaper (LeCunetal., 1990b). 

![4_image_0.png](4_image_0.png)

Figure1: (a) Objectivefunction (indB) versusnumberofparamet.ersforOBn 
(lowercurve) andmagnitude-basedparameterdeletion (uppercurve). (b) Predictedandactualobjectivefunctionversusnumberofparameters. Thepredictedvalue (lowercurve) isthesumofthesalienciesofthedeletedparameters. 

Figurelashowshowtheobjectivefunctionincreases (fromrighttoleft) asthenumberofremainingparametersdecreases. Itisclearthatdeletin~ parametersbyorderofsaliencycausesa significantlysmallerincreaseoftheobjectivefunctionthandeletingthemaccordingtotheirmagnitude. Randomdeletionswerealsotestedforthesakeofcomparison, buttheperformancewassobadthatthecurvescannotbeshownonthesamescale. Figure1bshowshowtheobjectivefunctionincreases (fromrighttoleft) asthenumberofremainingparametersdecreases, comparedtotheincreasepredictedbytheQuadratic-Extremum-Diagonalapproximation. Goodagrementisobtainedforuptoapproximately800deletedparameters (approximately30% oftheparameters). Beyondthatpoint, thecurvesbegintosplit, forseveralreasons: theoff-diagonaltermsinequation1 becomedisproportionatelymoreimportantasthe *number* ofdeletedparametersincreases, andhigher-than-quadratictermsbecomemoreimportantwhen *larger-valued* parametersaredeleted. ' 

![5_image_0.png](5_image_0.png)

Figure2: Objectivefunction (indB) versusnumberofparameters, withoutretraining (uppercurve), andafterretraining (lowercurve). Curvesaregivenforthetrainingset (a) andthetestset (b). 

Figure2 showsthelog-MSEonthetrainingsetandtheonthetestsetbeforeandafterretraining. Theperformanceonthetrainingsetandonthetestset (afterretraining) staysalmostthesamewhenupto1500parameters (60% ofthetotal) aredeleted. 

WehavealsousedOBnasaninteractivetoolfornetworkdesignandanalysis. 

Thiscontrastswiththeusualviewofweightdeletionasa more-or-Iessautomaticprocedure. Specifically, wepreparedchartsdepictingthesaliencyofthe10,000parametersinthedigitrecognitionnetworkreportedlastyear (LeCunetaI., 1990b). 

Tooursurprise, severallargegroupsofparameterswereexpendable. Wewereabletoexcisethesecond-to-Iastlayer, therebyreducingthenumberofparametersbya factoroftwo. ThetrainingsetMSEincreasedbya factorof10, andthegeneralizationMSEincreasedbyonly50%. The10-categoryclassificationerroronthetestsetactuallydecreased (whichindicatesthatMSEisnottheoptimalobjectivefunctionforthistask). OBDmotivatedotherarchitecturalchanges, ascanbeseenbycomparingthe2600-parameternetworkin (LeCunetaI., 1990a) tothe1O,OOO-parameternetworkin (LeCunetaI., 1990b). 

### 3ConclusionsAndOutlookWehaveusedOptimalBrainDamageinteractivelytoreducethenumberofparametersina practicalneuralnetworkbya factoroffour. WeobtainedanadditionalfactorofmorethantwobyusingOBDtodeleteparametersautomatically. Thenetwork'sspeedimprovedsignificantly, anditsrecognitionaccuracyincreasedslightly. Weemphasizethatthestartingpointwasa state-of-the-artnetwork. Itwouldbetooeasytostartwitha foolishnetworkandmakelargeimprovements: atechniquethatcanhelpimproveanalready-goodnetworkisparticularlyvaluable. Webelievethatthetechniquespresentedhereonlyscratchthesurfaceoftheapplicationswheresecond-derivativeinformationcanandshouldbeused. Inparticular, wehavealsobeenabletomovebeyondtheapproximationthat "complexityequalsnumberoffreeparameters" byusingsecond-derivativeinformation. In (Denker, LeCunandSolla, 1990), weuseittotoderiveanimprovedmeasureofthenetwork'sinformationcontent, orcomplexity. Thisallowsustocomparenetworkarchitecturesona giventask, andmakescontactwiththenotionofMinimumDescriptionLength (MDL) (Rissanen, 1989). Themainideaisthata "simple" networkwhosedescriptionneedsa smallnumberofbitsismorelikelytogeneralizecorrectlythana morecomplexnetwork, becauseitpresumablyhasextractedtheessenceofthedataandremovedtheredundancyfromit. 

#### AcknowledgmentsWethanktheUSPostalServiceanditscontractorsforprovidinguswiththedatabase. WealsothankRichHowardandLarryJackelfortheirhelpfulcommentsandencouragements. WeespeciallythankDavidRumelhartforsharingunpublishedideas. 

### ReferencesAkaike, H. (1986). UseofStatisticalModelsforTimeSeriesAnalysis. InProceedingsICASSP86, pages3147-3155, Tokyo. IEEE. 

Baum, E. B. andHaussler, D. (1989). WhatSizeNetGivesValidGeneraliztion? 

NeuralComputation, 1:151-160. 

Becker, S. andLeCun, Y. (1989). ImprovingtheConvergenceofBack-PropagationLearningwithSecond-OrderMethods. InTouretzky, D., Hinton, G., andSejnowski, T., editors, *Proc. ofthe* 1988 *ConnectionistModel& S.mmerSchool,* 
pages29-37, SanMateo. MorganKaufman. 

Chauvin, Y. (1989). ABack-PropagationAlgorithmwithOptimalUseofHiddenUnits. InTouretzky, D., editor, *NeuralInformationProce$$ingS,&tems,* 
volume1, Denver, 1988. MorganKaufmann. 

Denker, J., Schwartz, D., Wittner, B., Solla, S. A., Howard, R., Jackel, L., andHopfield, J. (1987). LargeAutomaticLearning, RuleExtractionandGeneralization. *ComplexSystems, 1:877-922.* 
Denker, J. S., LeCun, Y., andSolla, S. A. (1990). OptimalBrainDamage. ToappearinComputerandSystemSciences. 

Hanson, S. J. andPratt, L. Y. (1989). SomeComparisonsofConstraintsforMinimalNetworkConstructionwithBack-Propagation. InTouretzky, D., editor, NeuralInformationProcessingSystems, volume1, Denver, 1988. MorganKaufmann. 

LeCun, Y. (1987). *ModelesConnexionnistesdel'Apprentissage.* PhDthesis, UniversitePierreetMarieCurie, Paris, France. 

LeCun, Y. (1989). GeneralizationandNetworkDesignStrategies. InPfeifer, R., 
Schreter, Z., Fogelman, F., andSteels, L., editors, ConnectionisminPerspective, Zurich, Switzerland. Elsevier. 

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., andJackel, L. D. (1990a). HandwrittenDigitRecognitionwitha Back-
PropagationNetwork. InTouretzky, D., editor, *NeuralInformationProcessing* Systems, volume2, Denver, 1989. MorganKaufman. 

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., 
andJackel, L. D. (1990b). Back-PropagationAppliedtoHandwrittenZipcodeRecognition. *NeuralComputation,* 1{ 4). 

Mozer, M. C. andSmolensky, P. (1989). Skeletonization: ATechniqueforTrimmingtheFatfroma NetworkviaRelevanceAssessment. InTouretzky, D., 
editor, *NeuralInformationProcessingSystefn$,* volume1, Denver, 1988. MorganKaufmann. 

Rissanen, J. (1989). *StochasticComplexityinStatisticalInquiry.* WorldScientific, Singapore. 

Rumeihart, D. E. (1988). personalcommunication. 

Solla, S. A., Schwartz, D. B., Tishby, N., andLevin, E. (1990). SupervisedLearning: aTheoreticalFramework. InTouretzky, D., editor, NeuralInformationProcessingSystems, volume2, Denver, 1989. MorganKaufman. 

Vapnik, V. N. (1989). InductivePrinciplesoftheSearchforEmpiricalDependences. 

InProceedingsofthesecondannualWorkshoponComputationalLearningTheory, pages3-21. MorganKaufmann. 

Vapnik, V. N. andChervonenkis, A. Y. (1971). OntheUniformConvergenceofRelativeFrequenciesofEventstoTheirProbabilities. Th. Pro6. anditsApplications, 17(2):264-280. 

Waibel, A. (1989). ConsonantRecognitionbyModularConstructionofLargePhonemicTime-DelayNeuralNetworks. InTouretzky, D., editor, NeuralInformationProcessingSystems, volume1, pages215-223, Denver, 1988. MorganKaufmann.