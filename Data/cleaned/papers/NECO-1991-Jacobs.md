InNeuralComputation, 3, pages **79-87.**
AdaptiveMixturesofLocal **Experts**
RobertA. **Jacobs** MichaelI. **Jordan**
DepartmentofBrain & Cognitive **Sciences**
MassachusettsInstituteof **Technology**
Cambridge, MA **02139**
StevenJ. **Nowlan**
GeoffreyE. **Hinton**
DepartmentofComputer **Science**
Universityof **Toronto**
Toronto, CanadaM5S1A4

## AbstractWepresenta newsupervisedlearningprocedureforsystemscomposedofmany **separate**
networks, eachofwhichlearnstohandlea subsetofthecompletesetoftraining **cases.**
Thenewprocedurecanbeviewedeitherasa modularversionofa multilayer **supervised** network, orasanassociativeversionofcompetitivelearning. Itthereforeprovidesa newlinkbetweenthesetwoapparentlydifferentapproaches. Wedemonstratethatthe **learning** proceduredividesupa voweldiscriminationtaskintoappropriatesubtasks, eachof **which** canbesolvedbya verysimpleexpert **network.**

## 1MakingAssociativeLearning **Competitive**

Ifbackpropagationisusedtotraina single, multilayernetworktoperformdifferent **subtasks** ondifferentoccasions, therewillgenerallybestronginterferenceeffectswhichleadto **slow** learningandpoorgeneralization. Ifweknowinadvancethata setoftrainingcasesmaybenaturallydividedintosubsetsthatcorrespondtodistinctsubtasks, interferencecanbereducedbyusinga systemcomposedofseveraldifferent "expert" networksplusa **gating**
networkthatdecideswhichoftheexpertsshouldbeusedforeachtrainingcase. 1 **Hampshire**

#### 

1ThisideawasfirstpresentedbyJacobsandHintonattheConnectionistSummerSchoolin **Pittsburgh**
in **1988.**
andWaibel (1989) havedescribeda systemofthiskindthatcanbeusedwhenthe **division** intosubtasksisknownpriortotraining, andJacobs, JordanandBarto (1990) have **described**
arelatedsystemthatlearnshowtoallocatecasestoexperts. Theideabehindsucha **system**
isthatthegatingnetworkallocatesa newcasetooneora fewexperts, and, ifthe **output** isincorrect, theweightchangesarelocalizedtotheseexperts (andthegatingnetwork). So **there** isnointerferencewiththeweightsofotherexpertsthatspecializeinquitedifferent **cases.** Theexpertsarethereforelocalinthesensethattheweightsinoneexpertare **decoupled** fromtheweightsinotherexperts. Inadditiontheywilloftenbelocalinthesensethat **each** expertwillbeallocatedtoonlya smalllocalregionofthespaceofpossibleinput **vectors.**
Unfortunately, bothHampshireandWaibelandJacobsetal. useanerrorfunction **which**
doesnotencouragelocalization. Theyassumethatthefinaloutputofthewhole **system** isa linearcombinationoftheoutputsofthelocalexperts, withthegatingnetwork **determining** theproportionofeachlocaloutputinthelinearcombination. Sothefinalerroron **case** cis

$$E^{c}=\|\vec{d^{c}}-\sum_{i}p_{i}^{c}\vec{o}_{i}^{c}\|^{2}$$
i
$$(1)$$

2(1)
where ~oc iistheoutputvectorofexperti on **case** c, pciistheproportional **contribution** ofexperti tothecombinedoutputvector, and ~dcisthedesiredoutputvectorin **case** c.

Thiserrormeasurecomparesthedesiredoutputwitha blendoftheoutputsofthe **local**
experts, so, tominimizetheerror, eachlocalexpertmustmakeitsoutputcancelthe **residual** errorthatisleftbythecombinedeffectsofalltheotherexperts. Whentheweightsinoneexpertchange, theresidualerrorchanges, andsotheerrorderivativesforalltheother **local** experts **change.** 
2Thisstrongcouplingbetweentheexpertscausesthemtocooperate **nicely,**
buttendstoleadtosolutionsinwhichmanyexpertsareusedforeachcase. Itis **possible** toencouragecompetitionbyaddingpenaltytermstotheobjectivefunctionto **encourage** solutionsinwhichonlyoneexpertisactive (Jacobs, Jordan, andBarto, 1990), buta **simpler** remedyistoredefinetheerrorfunctionsothatthelocalexpertsareencouragedto **compete** ratherthan **cooperate.**
Insteadoflinearlycombiningtheoutputsoftheseparateexperts, weimaginethatthegatingnetworkmakesa stochasticdecisionaboutwhichsingleexperttouseoneach **occasion** (seefigure1). Theerroristhentheexpectedvalueofthesquareddifferencebetweenthedesiredandactualoutput **vectors**

2ForHampshireandWaibel, thisproblemdoesnotarisebecausetheydonotlearnthetask **decomposition.**
Theytraineach **expert** separatelyonitsownpre-assigned **subtask.**

![2_image_0.png](2_image_0.png)

Figure1: Asystemofexpertandgatingnetworks. Eachexpertisa feedforwardnetworkandallexpertsreceivethesameinputandhavethesamenumberofoutputs. Thegatingnetworkisalsofeedforwardandtypicallyreceivesthesameinputastheexpertnetworks.

Ithasnormalizedoutputsp; = exp(x;}/ ∑;exp(x;), wherex; isthetotalweightedinputreceivedbyoutputunitj ofthegatingnetwork. Theselectoractslikea multipleinput, singleoutputstochasticswitch; theprobabilitythattheswitchwillselecttheoutputfromexpertj ispj.

$$(2)$$

$$E^{c}=<\!\!\|\vec{d}^{\vec{c}}-\vec{o}_{i}^{c}\|^{2}\!\!>\;=\sum_{i}p_{i}^{c}\|\vec{d}^{\vec{c}}-\vec{o}_{i}^{c}\|^{2}$$

coNoticethatinthisnewerrorfunction, eachexpertisrequiredtoproducethe **whole** oftheoutputvectorratherthana residual. Asa result, thegoalofa localexpertona **given** trainingcaseisnotdirectlyaffectedbytheweightswithinotherlocalexperts. Thereis **still** someindirectcouplingbecauseifsomeotherexpertchangesitsweights, itmaycausethegatingnetworktoaltertheresponsibilitiesthatgetassignedtotheexperts, butat **least** theseresponsibilitychangescannotalterthesignoftheerrorthata localexpert **senses** ona giventrainingcase. Ifboththegatingnetworkandthelocalexpertsare **trained** bygradientdescentinthisnewerrorfunction, thesystemtendstodevotea single **expert** toeachtrainingcase. Wheneveranexpertgiveslesserrorthantheweightedaverageoftheerrorsofalltheexperts (usingtheoutputsofthegatingnetworktodecidehowto **weight** eachexpert'serror) itsresponsibilityforthatcasewillbeincreased, andwheneverit **does** worsethantheweightedaverageitsresponsibilitywillbe **decreased.**
Theerrorfunctioninequation2 worksinpracticebutinthesimulationsreported **below**
weuseda differenterrorfunctionwhichgivesbetter **performance:**

$$E^{c}=-\log\sum_{i}p_{i}^{c}e^{-\frac{1}{2}\|\vec{d}^{\vec{c}}-\vec{d}_{i}^{c}\|^{2}}\tag{3}$$

Theerrordefinedinequation3 issimplythenegativelogprobabilityofgeneratingthedesiredoutputvectorunderthemixtureofgaussiansmodeldescribedattheendofthe **next** section. Toseewhythiserrorfunctionworksbetter, itishelpfultocomparethe **derivatives** ofthetwoerrorfunctionswithrespecttotheoutputofanexpert. Fromequation2 weget

$$\frac{\partialE^{c}}{\partial\vec{o}_{i}^{c}}=-2p_{i}^{c}(\vec{d}^{\vec{c}}-\vec{o}_{i}^{c})\tag{1}$$

whilefromequation3 weget

$$\frac{\partialE^{c}}{\partial\sigma_{i}^{c}}=-\left[\frac{p_{i}^{c}e^{-\frac{1}{2}\|\vec{d}^{c}-\vec{\sigma}_{i}^{c}\|^{2}}}{\sum_{j}p_{j}^{c}e^{-\frac{1}{2}\|\vec{d}^{c}-\vec{\sigma}_{j}^{c}\|^{2}}}\right](\vec{d}^{\vec{E}}-\vec{\sigma}_{i}^{c})\tag{5}$$
$$\left(4\right)$$

Inequation4 the **term** pc iisusedtoweightthederivativeforexperti. Inequation5 weusea weightingtermthattakesintoaccounthowwellexperti doesrelativetoother **experts.** Thisisa moreusefulmeasureoftherelevanceofexperti totrainingcasec, especially **early** inthetraining. Suppose, forexample, thatthegatingnetworkinitiallygivesequal **weights**
toallexpertsandk~dc − ~oc ik > 1foralltheexperts. Equation4 willadaptthe **best-fitting**
experttheslowest, whereasequation5 willadaptitthe **fastest.**

# 2MakingCompetitiveLearning **Associative**

Itisnaturaltothinkthatthe "data" vectorsonwhicha competitivenetworkis **trained**
playa rolesimilartotheinputvectorsofanassociativenetworkthatmapsinput **vectors**
tooutputvectors. Thiscorrespondenceisassumedinmodelsthatusecompetitive **learning** asa preprocessingstagewithinanassociativenetwork (MoodyandDarken, 1989). A **quite** differentviewisthatthedatavectorsusedincompetitivelearningcorrespondtotheoutputvectorsofanassociativenetwork. Thecompetitivenetworkcanthenbe **viewed** asaninputlessstochasticgeneratorofoutputvectorsandcompetitivelearningcanbe **viewed** asa procedureformakingthenetworkgenerateoutputvectorswitha distributionthat **matches** thedistributionofthe "data" vectors. Theweightvectorofeachcompetitivehidden **unit** representsthemeanofa multidimensionalgaussiandistribution, andoutputvectorsaregeneratedbyfirstpickinga hiddenunitandthenpickinganoutputvectorfromthe **gaussian** distributiondeterminedbytheweightvectorofthechosenhiddenunit. Thelog **probability** ofgeneratinganyparticularoutput **vector** ~ocis **then**

$$\logP^{c}=\log\sum_{i}p_{i}ke^{-\frac{1}{2}\|\vec{\mu}_{i}-\vec{\sigma}^{c}\|^{2}}$$
$\left(\vec{0}\right)$. 

(6)

wherei isanindexoverthehiddenunits, µ~iisthe "weight" vectorofthehidden **unit,** kisa normalizingconstant, andpiistheprobabilityofpickinghiddenuniti, sothepiareconstrainedtosumto1. Inthestatisticsliterature (McLachlanandBasford, 1988), thepiarecalled "mixing **proportions".**
"Soft" competitivelearningmodifiestheweights (andalsothevariancesandthe **mixing**
proportions) soastoincreasetheproductoftheprobabilities (i.e. thelikelihood) ofgeneratingtheoutputvectorsinthetrainingset (Nowlan, 1990). "Hard" competitive **learning** isa simpleapproximationtosoftcompetitivelearninginwhichweignorethepossibility **that** adatavectorcouldbegeneratedbyseveraldifferenthiddenunits. Instead, weassume **that** itmustbegeneratedbythehiddenunitwiththeclosestweightvector, soonlythis **weight** vectorneedstobemodifiedtoincreasetheprobabilityofgeneratingthedata **vector.**
Ifweviewa competitivenetworkasgeneratingoutputvectors, itisnot **immediately**
obviouswhatroleinputvectorscouldplay. However, competitivelearningcanbe **generalized** inmuchthesamewayasBarto (1985) hasgeneralizedlearningautomataby **adding** aninputvectorandmakingtheactionsoftheautomatonbeconditionalontheinput **vector.** Wereplaceeachhiddenunitina competitivenetworkbyanentireexpertnetwork **whose** outputvectorspecifiesthemeanofa multidimensionalgaussiandistribution. Sothe **means**

### 5arenowa functionofthecurrentinputvectorandarerepresentedbyactivitylevels **rather** thanweights. Inaddition, weusea gatingnetworkwhichallowsthemixing **proportions** oftheexpertstobedeterminedbytheinputvector. Thisgivesusa systemofcompeting **local** expertswiththeerrorfunctiondefinedinequation3. Wecouldalsointroducea **mechanism** toallowtheinputvectortodynamicallydeterminethecovariancematrixforthe **distribution** definedbyeachexpertnetwork, butwehavenotyetexperimentedwiththis **possibility.**

## 3ApplicationToMulti-SpeakerVowel **Recognition**

Themixtureofexpertsmodelwasevaluatedona speakerindependent, four-class, **vowel** discriminationproblem. Thedataconsistedofthefirstandsecondformantsofthevowels **[i],**
[I], [a], and [A] (usuallydenoted [Λ]) from75speakers (males, femalesandchildren) **uttered**
ina hVdcontext (Peterson & Barney, 1952). Thedataformstwopairsof **overlapping** classes, anddifferentexpertslearntoconcentrateononepairofclassesortheother **(figure** 2).

Wecomparedstandardback-propagationnetworkscontaininga singlehidden **layer** of6 or12unitswithmixturesof4 or8 verysimpleexperts. Thearchitectureofeach **expert** wasrestrictedsoitcouldformonlya lineardecisionsurfacewhichisdefinedasthesetofinputvectorsforwhichtheexpertgivesanoutputofexactly0.5. Allmodelswere **trained** withdatafromthefirst50speakersandtestedwithdatafromtheremaining25 **speakers.** Thesmallnumberofparametersforeachexpertallowsexcellentgeneralization **performance** (table1), andpermitsa graphicalrepresentationoftheprocessoftaskdecomposition **(figure** 3). Thenumberofhiddenunitsinthebackpropagationnetworkswaschosentogive **roughly** equalnumbersofparametersforthebackpropagationnetworksandmixturemodels. Allsimulationswereperformedusinga simplegradientdescentalgorithmwithfixedstep **size** .

Tosimplifythecomparisons, nomomentumorotheraccelerationtechniqueswereused. Thevalueofforeachsystemwaschosenbyperforminga limitedexplorationofthe **convergence** fromthesameinitialconditionsfora rangeof . Batchtrainingwasusedwithone **weight** updateforeachpassthroughthetrainingset (epoch). Eachsystemwastrained **until** anaveragesquarederrorof0.08overthetrainingsetwas **obtained.**
Themixturesofexpertsreachtheerrorcriterionsignificantlyfasterthanthebackpropagationnetworks (p0.999), requiringonlyabouthalfasmanyepochson **average** (table1). Thelearningtimeforthemixturemodelalsoscaleswellasthenumberof **experts** isincreased: Themixtureof8 expertshasa small, butstatisticallysignificant (p > 0.**95),** advantageintheaveragenumberofepochsrequiredtoreachtheerrorcriterion. In **contrast,**

### 6Figure2: Dataforvoweldiscriminationproblem, andexpertandgatingnetwork **decision**

![6_image_0.png](6_image_0.png) lines. Thehorizontalaxisisthefirstformantvalue, andtheverticalaxisisthe **second** formantvalue (theformantvalueshavebeenlinearlyscaledbydividingbya factorof **1000).** Eachexampleislabelledwithitscorrespondingvowelsymbol. Vowels [i] and [I] formoneoverlappingpairofclasses, vowels [a] and [A] formtheotherpair. ThelineslabelledNet0, 1and2 representthedecisionlinesfor3 expertnetworks. Ononesideoftheselinestheoutputofthecorrespondingexpertislessthan0.5, ontheothersidetheoutputis **greater** than0.5. Althoughthemixtureinthiscasecontained4 experts, oneoftheseexperts **made** nosignificantcontributiontothefinalmixturesinceitsmixingproportionpiwas **effectively** 0forallcases. ThelinelabelledGate0:2indicatesthedecisionbetweenexpert0 and **expert** 2madebythegatingnetwork. Totheleftofthislinep2 > p0, totherightofthis **line** p0 > p2. Theboundarybetweenclasses [a] and [A] isformedbythecombinationofthe **left** partofNet2'sdecisionlineandtherightpartofNet0'sdecisionline. Althoughthe **system** tendstouseasfewexpertsasitcantosolvea problem, itisalsosensitivetospecific **problem** featuressuchastheslightlycurvedboundarybetweenclasses [a] and **[A].**

| | System | | Train | % | Correct | Test | % Correct | Avg. # | Epochs | Std. | Dev. |
|----|----------|-----|---------|-----|-----------|--------|-------------|----------|----------|--------|--------|
| 4 | Experts | | | 88 | | | 90 | 1124 | | 23 | |
| 8 | Experts | | | 88 | | | 90 | 1083 | | 12 | |
| BP | 6 | Hid | | 88 | | | 90 | 2209 | | 83 | |
| BP | 12 | Hid | | 88 | | | 90 | 2435 | | 124 | |

Table1: Summaryofperformanceonvoweldiscriminationtask. Resultsare **based** on25simulationsforeachofthealternativemodels. Thefirstcolumnofthetableindicatesthesystemsimulated. Thesecondcolumngivesthepercentoftrainingcasesclassified **correctly** bythefinalsetofweights, whilethethirdcolumnindicatesthepercentoftesting **cases** classifiedcorrectly. Thelasttwocolumnscontaintheaveragenumberofepochs **required** toreachtheerrorcriterion, andthestandarddeviationofthedistributionof **convergence** times. Althoughthesquarederrorwasusedtodecidewhentostoptraining, the **criterion** forcorrectperformanceisbasedona weightedaverageoftheoutputsofallthe **experts.** Eachexpertassignsa probabilitydistributionovertheclassesandthesedistributionsarecombinedusingproportionsgivenbythegatingnetwork. Themostprobableclassis **then**
takentobetheresponseofthesystem. Theidenticalperformanceofallthesystemsisduetothefactthat, withthisdataset, thesetofmisclassifiedexamplesisnotsensitiveto **small** changesinthedecisionsurfaces. Also, thetestsetiseasierthanthetraining **set.**

Figure3: Thetrajectoriesofthedecisionlinesofsomeexpertsduringonesimulation. The

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png) horizontalaxisisthefirstformantvalue, andtheverticalaxisisthesecondformant **value.** Eachtrajectoryisrepresentedbya sequenceofdots, oneperepoch, eachdotmarkingtheintersectionoftheexpert'sdecisionlineandthenormaltothatlinepassingthroughtheorigin. Forclarity, only5 ofthe8 expertsareshownandthenumberoftheexpertis **shown**
atthestartofthetrajectory. ThepointlabelledT0indicatestheoptimaldecisionlinefora singleexperttrainedtodiscriminate [i] from [I]. Similarly, T1representstheoptimal **decision** linetodiscriminate [a] from [A]. ThepointlabelledX isthedecisionlinelearnedbya **single**
experttrainedwithdatafromall4 classes, andrepresentsa typeofaverage **solution.**

the12hiddenunitback-propagationnetworkrequiresmoreepochs (p > 0.95) toreachtheerrorcriterionthanthenetworkwith6 hiddenunits (table1). Allstatistical **comparisons**
arebasedona t-testwith48degreesoffreedomanda pooledvariance **estimator.**
Figure3 showshowthedecisionlinesofdifferentexpertsmovearoundasthe **system**
learnstoallocatepiecesofthetasktodifferentexperts. Thesystembeginsinan **unbiased** state, withthegatingnetworkassigningequalmixingproportionstoallexpertsinall **cases.** Asa result, eachexperttendstogeterrorsfromroughlyequalnumbersofcasesinall4 classes, andallexpertsheadtowardsthepointX, whichrepresentstheoptimal **decision** lineforanexpertthatmustdealwithallthecases. Onceoneormoreexperts **begin** toreceivemoreerrorfromcasesinoneclasspairthantheother, thissymmetryisbrokenandthetrajectoriesbegintodivergeasdifferentexpertsconcentrateononeclasspairortheother. Inthissimulation, expert5 learnstoconcentrateondiscriminatingclasses [i] and
[I] soitsdecisionlineapproachestheoptimallineforthisdiscrimination (T0). **Experts** 4and6 bothconcentrateondiscriminatingclasses [a] and [A], sotheirtrajectories **approach**
theoptimalsingleline (T1) andthensplittoforma piecewiselinearapproximationtotheslightlycurvedoptimaldecisionsurface (seefigure2). Onlyexperts4, 5, and6 are **active** inthefinalmixture. Thissolutionistypical - inallsimulationswithmixturesof4 or8 **experts** allbut2 or3 expertshadmixingproportionsthatwereeffectively0 forall **cases.**

# AcknowledgementsJordanandJacobswerefundedbygrantsfromSiemensandtheMcDonnell-Pew **program**
inCognitiveNeuroscience. HintonandNowlanwerefundedbygrantsfromthe **Ontario** InformationTechnologyResearchCenterandtheCanadianNaturalScienceand **Engineering** ResearchCouncil. Hintonisa fellowoftheCanadianInstituteforAdvanced **Research.**

### ReferencesBarto, A. G. (1985) Learningbystatisticalcooperationofself-interestedneuron-likecomputingelements. HumanNeurobiology, **4:229-256.** Hampshire, J. andWaibel, A. (1989) TheMeta-Pinetwork: Buildingdistributed **knowledge** representationsforrobustpatternrecognition, TechnicalReportCMU-CS-89-166, **Carnegie** MellonUniversity, Pittsburgh, PA. Jacobs, R.A. & Jordan, M.I. (1991) Learningpiecewisecontrolstrategiesina modularconnectionistarchitecture, in **preparation**. Jacobs, R. A., Jordan, M. I. andBarto, A. G. (1991) Taskdecompositionthroughcompetitionina modularconnectionistarchitecture: Thewhatandwherevisiontasks. **Cognitive** Science, in **press**. McLachlan, G. J. andBasford, K. E. (1988) Mixturemodels: Inferenceand **applications** toclustering. MarcelDekker, **Inc.**
Moody, J. andDarken, C. (1989) Fastlearninginnetworksoflocally-tunedprocessing **units.**
NeuralComputation, 1**(2):281-294.**
Nowlan, S. J. (1990) MaximumLikelihoodCompetitiveLearning. InD. S. Touretzky **(ed.),**
AdvancesinNeuralInformationProcessingSystems2, pp. 574-582. SanMateo, CA: **Morgan**
Kaufmann. Nowlan, S. J. (1990) Competingexperts: Anexperimentalinvestigationofassociativemixturemodels. TechnicalReportCRG-TR-90-5, UniversityofToronto, Toronto, **Canada.** Peterson, G. E. andBarney, H. L. (1952) ControlMethodsUsedina Studyofthe **Vowels,** J. Acoust. Soc. Am., vol. 24, pp. **175-184.**