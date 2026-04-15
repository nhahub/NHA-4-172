# ImprovingLanguageUnderstandingByGenerativePre-Training

| AlecRadford | KarthikNarasimhan | TimSalimans | IlyaSutskever |
|-----------------|----------------------|----------------|-------------------|
| OpenAI | OpenAI | OpenAI | OpenAI |
| alec@openai.com | karthikn@openai.com | tim@openai.com | ilyasu@openai.com |

# AbstractNaturallanguageunderstandingcomprisesa widerangeofdiversetaskssuchastextualentailment, questionanswering, semanticsimilarityassessment, anddocumentclassification. Althoughlargeunlabeledtextcorporaareabundant, labeleddataforlearningthesespecifictasksisscarce, makingitchallengingfordiscriminativelytrainedmodelstoperformadequately. Wedemonstratethatlargegainsonthesetaskscanberealizedby *generativepre-training* ofa languagemodelona diversecorpusofunlabeledtext, followedby *discriminativefine-tuning* oneachspecifictask. Incontrasttopreviousapproaches, wemakeuseoftask-awareinputtransformationsduringfine-tuningtoachieveeffectivetransferwhilerequiringminimalchangestothemodelarchitecture. Wedemonstratetheeffectivenessofourapproachona widerangeofbenchmarksfornaturallanguageunderstanding.

Ourgeneraltask-agnosticmodeloutperformsdiscriminativelytrainedmodelsthatusearchitecturesspecificallycraftedforeachtask, significantlyimprovinguponthestateoftheartin9 outofthe12tasksstudied. Forinstance, weachieveabsoluteimprovementsof8.9% oncommonsensereasoning (StoriesClozeTest), 5.7% onquestionanswering (RACE), and1.5% ontextualentailment (MultiNLI).

# 1IntroductionTheabilitytolearneffectivelyfromrawtextiscrucialtoalleviatingthedependenceonsupervisedlearninginnaturallanguageprocessing (NLP). Mostdeeplearningmethodsrequiresubstantialamountsofmanuallylabeleddata, whichrestrictstheirapplicabilityinmanydomainsthatsufferfroma dearthofannotatedresources [61]. Inthesesituations, modelsthatcanleveragelinguisticinformationfromunlabeleddataprovidea valuablealternativetogatheringmoreannotation, whichcanbetime-consumingandexpensive. Further, evenincaseswhereconsiderablesupervisionisavailable, learninggoodrepresentationsinanunsupervisedfashioncanprovidea significantperformanceboost. Themostcompellingevidenceforthissofarhasbeentheextensiveuseofpretrainedwordembeddings [10, 39, 42] toimproveperformanceona rangeofNLPtasks [8, 11, 26, 45]. Leveragingmorethanword-levelinformationfromunlabeledtext, however, ischallengingfortwomainreasons. First, itisunclearwhattypeofoptimizationobjectivesaremosteffectiveatlearningtextrepresentationsthatareusefulfortransfer. Recentresearchhaslookedatvariousobjectivessuchaslanguagemodeling [44], machinetranslation [38], anddiscoursecoherence [22], witheachmethodoutperformingtheothersondifferenttasks.1Second, thereisnoconsensusonthemosteffectivewaytotransfertheselearnedrepresentationstothetargettask. Existingtechniquesinvolvea combinationofmakingtask-specificchangestothemodelarchitecture [43, 44], usingintricatelearningschemes [21] andaddingauxiliarylearningobjectives [50]. Theseuncertaintieshavemadeitdifficulttodevelopeffectivesemi-supervisedlearningapproachesforlanguageprocessing.

1https://gluebenchmark.com/leaderboardInthispaper, weexplorea semi-supervisedapproachforlanguageunderstandingtasksusinga combinationofunsupervisedpre-trainingandsupervisedfine-tuning. Ourgoalistolearna universalrepresentationthattransferswithlittleadaptationtoa widerangeoftasks. Weassumeaccesstoa largecorpusofunlabeledtextandseveraldatasetswithmanuallyannotatedtrainingexamples (targettasks). Oursetupdoesnotrequirethesetargettaskstobeinthesamedomainastheunlabeledcorpus. Weemploya two-stagetrainingprocedure. First, weusea languagemodelingobjectiveontheunlabeleddatatolearntheinitialparametersofa neuralnetworkmodel. Subsequently, weadapttheseparameterstoa targettaskusingthecorrespondingsupervisedobjective. Forourmodelarchitecture, weusethe *Transformer* [62], whichhasbeenshowntoperformstronglyonvarioustaskssuchasmachinetranslation [62], documentgeneration [34], andsyntacticparsing [29]. Thismodelchoiceprovidesuswitha morestructuredmemoryforhandlinglong-termdependenciesintext, comparedtoalternativeslikerecurrentnetworks, resultinginrobusttransferperformanceacrossdiversetasks. Duringtransfer, weutilizetask-specificinputadaptationsderivedfromtraversal-styleapproaches [52], whichprocessstructuredtextinputasa singlecontiguoussequenceoftokens. Aswedemonstrateinourexperiments, theseadaptationsenableustofine-tuneeffectivelywithminimalchangestothearchitectureofthepre-trainedmodel.

Weevaluateourapproachonfourtypesoflanguageunderstandingtasks - naturallanguageinference, questionanswering, semanticsimilarity, andtextclassification. Ourgeneraltask-agnosticmodeloutperformsdiscriminativelytrainedmodelsthatemployarchitecturesspecificallycraftedforeachtask, significantlyimprovinguponthestateoftheartin9 outofthe12tasksstudied. Forinstance, weachieveabsoluteimprovementsof8.9% oncommonsensereasoning (StoriesClozeTest) [40],
5.7% onquestionanswering (RACE) [30], 1.5% ontextualentailment (MultiNLI) [66] and5.5% ontherecentlyintroducedGLUEmulti-taskbenchmark [64]. Wealsoanalyzedzero-shotbehaviorsofthepre-trainedmodelonfourdifferentsettingsanddemonstratethatitacquiresusefullinguisticknowledgefordownstreamtasks.

# 2RelatedWorkSemi-supervisedlearningforNLPOurworkbroadlyfallsunderthecategoryofsemi-supervisedlearningfornaturallanguage. Thisparadigmhasattractedsignificantinterest, withapplicationstotaskslikesequencelabeling [24, 33, 57] ortextclassification [41, 70]. Theearliestapproachesusedunlabeleddatatocomputeword-levelorphrase-levelstatistics, whichwerethenusedasfeaturesina supervisedmodel [33]. Overthelastfewyears, researchershavedemonstratedthebenefitsofusingwordembeddings [11, 39, 42], whicharetrainedonunlabeledcorpora, toimproveperformanceona varietyoftasks [8, 11, 26, 45]. Theseapproaches, however, mainlytransferword-levelinformation, whereasweaimtocapturehigher-levelsemantics. Recentapproacheshaveinvestigatedlearningandutilizingmorethanword-levelsemanticsfromunlabeleddata. Phrase-levelorsentence-levelembeddings, whichcanbetrainedusinganunlabeledcorpus, havebeenusedtoencodetextintosuitablevectorrepresentationsforvarioustargettasks [28, 32, 1, 36, 22, 12, 56, 31].

Unsupervisedpre-trainingUnsupervisedpre-trainingisa specialcaseofsemi-supervisedlearningwherethegoalistofinda goodinitializationpointinsteadofmodifyingthesupervisedlearningobjective. Earlyworksexploredtheuseofthetechniqueinimageclassification [20, 49, 63] andregressiontasks [3]. Subsequentresearch [15] demonstratedthatpre-trainingactsasa regularizationscheme, enablingbettergeneralizationindeepneuralnetworks. Inrecentwork, themethodhasbeenusedtohelptraindeepneuralnetworksonvarioustaskslikeimageclassification [69], speechrecognition [68], entitydisambiguation [17] andmachinetranslation [48]. Theclosestlineofworktooursinvolvespre-traininga neuralnetworkusinga languagemodelingobjectiveandthenfine-tuningitona targettaskwithsupervision. Daietal. [13] andHowardandRuder [21] followthismethodtoimprovetextclassification. However, althoughthepre-trainingphasehelpscapturesomelinguisticinformation, theirusageofLSTMmodelsrestrictstheirpredictionabilitytoa shortrange. Incontrast, ourchoiceoftransformernetworksallowsustocapturelongerrangelinguisticstructure, asdemonstratedinourexperiments. Further, wealsodemonstratetheeffectivenessofourmodelona widerrangeoftasksincludingnaturallanguageinference, paraphrasedetectionandstorycompletion. Otherapproaches [43, 44, 38] usehiddenrepresentationsfroma pre-trainedlanguageormachinetranslationmodelasauxiliaryfeatureswhiletraininga supervisedmodelonthetargettask. Thisinvolvesa substantialamountofnewparametersforeachseparatetargettask, whereaswerequireminimalchangestoourmodelarchitectureduringtransfer. AuxiliarytrainingobjectivesAddingauxiliaryunsupervisedtrainingobjectivesisanalternativeformofsemi-supervisedlearning. EarlyworkbyCollobertandWeston [10] useda widevarietyofauxiliaryNLPtaskssuchasPOStagging, chunking, namedentityrecognition, andlanguagemodelingtoimprovesemanticrolelabeling. Morerecently, Rei [50] addedanauxiliarylanguagemodelingobjectivetotheirtargettaskobjectiveanddemonstratedperformancegainsonsequencelabelingtasks. Ourexperimentsalsouseanauxiliaryobjective, butasweshow, unsupervisedpre-trainingalreadylearnsseverallinguisticaspectsrelevanttotargettasks.

# 3FrameworkOurtrainingprocedureconsistsoftwostages. Thefirststageislearninga high-capacitylanguagemodelona largecorpusoftext. Thisisfollowedbya fine-tuningstage, whereweadaptthemodeltoa discriminativetaskwithlabeleddata.

### 3.1UnsupervisedPre-TrainingGivenanunsupervisedcorpusoftokensU = {u1*, . . . , u*n}, weusea standardlanguagemodelingobjectivetomaximizethefollowinglikelihood:

$$L_{1}({\mathcal{U}})=\sum_{i}\logP(u_{i}|u_{i-k},\ldots,u_{i-1};\Theta)$$
iwherek isthesizeofthecontextwindow, andtheconditionalprobabilityP ismodeledusinga neuralnetworkwithparametersΘ. Theseparametersaretrainedusingstochasticgradientdescent [51]. Inourexperiments, weusea multi-layer *Transformerdecoder* [34] forthelanguagemodel, whichisa variantofthetransformer [62]. Thismodelappliesa multi-headedself-attentionoperationovertheinputcontexttokensfollowedbyposition-wisefeedforwardlayerstoproduceanoutputdistributionovertargettokens:

$h_{0}=UW_{e}+W_{p}$ $h_{l}=$transformer_block$(h_{l-1})\foralli\in[1,n]$ $P(u)=$softmax$(h_{n}W_{e}^{T})$
$$(1)$$
$$(2)$$
$$({\mathfrak{I}})$$
$$(4)$$
whereU = (u−k*, . . . , u*−1) isthecontextvectoroftokens, nisthenumberoflayers, Weisthetokenembeddingmatrix, andWpisthepositionembeddingmatrix.

### 3.2SupervisedFine-TuningAftertrainingthemodelwiththeobjectiveinEq. 1, weadapttheparameterstothesupervisedtargettask. Weassumea labeleddatasetC, whereeachinstanceconsistsofa sequenceofinputtokens, x1*, . . . , x*m, alongwitha labely. Theinputsarepassedthroughourpre-trainedmodeltoobtainthefinaltransformerblock'sactivationh ml, whichisthenfedintoanaddedlinearoutputlayerwithparametersWytopredicty:

$\frac{1}{2}$
$$\Gamma$$
P(y|x

$$x^{1},\ldots,x^{m})={\tts of tm ax}(h_{l}^{m}W_{y}).$$
lWy). (3)

Thisgivesusthefollowingobjectivetomaximize:

effectivetomaximize: $L_{2}(C)=\sum_{(x,y)}\logP(y|x^{1},\ldots,x^{m})$. 
1*, . . . , x*m). (4)
Weadditionallyfoundthatincludinglanguagemodelingasanauxiliaryobjectivetothefine-tuninghelpedlearningby (a) improvinggeneralizationofthesupervisedmodel, and (b) acceleratingconvergence. Thisisinlinewithpriorwork [50, 43], whoalsoobservedimprovedperformancewithsuchanauxiliaryobjective. Specifically, weoptimizethefollowingobjective (withweightλ):
L3(C) = L2(C) + λ ∗ L1(C) (5)
Overall, theonlyextraparameterswerequireduringfine-tuningareWy, andembeddingsfordelimitertokens (describedbelowinSection3.3).

Figure1: **(left)** Transformerarchitectureandtrainingobjectivesusedinthiswork. **(right)** Input

![3_image_0.png](3_image_0.png)

transformationsforfine-tuningondifferenttasks. Weconvertallstructuredinputsintotokensequencestobeprocessedbyourpre-trainedmodel, followedbya linear+softmaxlayer.

### 3.3Task-SpecificInputTransformationsForsometasks, liketextclassification, wecandirectlyfine-tuneourmodelasdescribedabove.

Certainothertasks, likequestionansweringortextualentailment, havestructuredinputssuchasorderedsentencepairs, ortripletsofdocument, question, andanswers. Sinceourpre-trainedmodelwastrainedoncontiguoussequencesoftext, werequiresomemodificationstoapplyittothesetasks. Previousworkproposedlearningtaskspecificarchitecturesontopoftransferredrepresentations [44]. Suchanapproachre-introducesa significantamountoftask-specificcustomizationanddoesnotusetransferlearningfortheseadditionalarchitecturalcomponents. Instead, weusea traversal-styleapproach [52], whereweconvertstructuredinputsintoanorderedsequencethatourpre-trainedmodelcanprocess. Theseinputtransformationsallowustoavoidmakingextensivechangestothearchitectureacrosstasks. Weprovidea briefdescriptionoftheseinputtransformationsbelowandFigure1 providesa visualillustration. Alltransformationsincludeaddingrandomlyinitializedstartandendtokens (hsi, hei). TextualentailmentForentailmenttasks, weconcatenatethepremisep andhypothesish tokensequences, witha delimitertoken ($) inbetween.

SimilarityForsimilaritytasks, thereisnoinherentorderingofthetwosentencesbeingcompared. Toreflectthis, wemodifytheinputsequencetocontainbothpossiblesentenceorderings (witha delimiterinbetween) andprocesseachindependentlytoproducetwosequencerepresentationsh ml whichareaddedelement-wisebeforebeingfedintothelinearoutputlayer.

QuestionAnsweringandCommonsenseReasoningForthesetasks, wearegivena contextdocumentz, aquestionq, anda setofpossibleanswers {ak}. Weconcatenatethedocumentcontextandquestionwitheachpossibleanswer, addinga delimitertokeninbetweentoget [z; q; $; ak]. Eachofthesesequencesareprocessedindependentlywithourmodelandthennormalizedviaa softmaxlayertoproduceanoutputdistributionoverpossibleanswers.

# 4Experiments

### 4.1SetupUnsupervisedpre-trainingWeusetheBooksCorpusdataset [71] fortrainingthelanguagemodel. Itcontainsover7,000uniqueunpublishedbooksfroma varietyofgenresincludingAdventure, Fantasy, andRomance. Crucially, itcontainslongstretchesofcontiguoustext, whichallowsthegenerativemodeltolearntoconditiononlong-rangeinformation. Analternativedataset, the1BWordBenchmark, whichisusedbya similarapproach, ELMo [44], isapproximatelythesamesizeTable1: Alistofthedifferenttasksanddatasetsusedinourexperiments.

| Task | Datasets |
|----------------------------|-------------------------------------------------------------------------|
| Naturallanguageinference | SNLI [5], MultiNLI [66], QuestionNLI [64], RTE [4], SciTail [25] |
| QuestionAnswering | RACE [30], StoryCloze [40] |
| Sentencesimilarity | MSRParaphraseCorpus [14], QuoraQuestionPairs [9], STSBenchmark [6] |
| Classification | StanfordSentimentTreebank\-2 [54], CoLA [65] |

butisshuffledata sentencelevel - destroyinglong-rangestructure. Ourlanguagemodelachievesa verylowtokenlevelperplexityof18.4onthiscorpus. ModelspecificationsOurmodellargelyfollowstheoriginaltransformerwork [62]. Wetraineda 12-layerdecoder-onlytransformerwithmaskedself-attentionheads (768dimensionalstatesand12attentionheads). Fortheposition-wisefeed-forwardnetworks, weused3072dimensionalinnerstates. WeusedtheAdamoptimizationscheme [27] witha maxlearningrateof2.5e-4. Thelearningratewasincreasedlinearlyfromzerooverthefirst2000updatesandannealedto0 usinga cosineschedule. Wetrainfor100epochsonminibatchesof64randomlysampled, contiguoussequencesof512tokens.

Sincelayernorm [2] isusedextensivelythroughoutthemodel, asimpleweightinitializationofN(0, 0.02) wassufficient. Weuseda bytepairencoding (BPE) vocabularywith40,000merges [53] andresidual, embedding, andattentiondropoutswitha rateof0.1forregularization. Wealsoemployeda modifiedversionofL2regularizationproposedin [37], withw = 0.01onallnonbiasorgainweights. Fortheactivationfunction, weusedtheGaussianErrorLinearUnit (GELU) [18]. Weusedlearnedpositionembeddingsinsteadofthesinusoidalversionproposedintheoriginalwork.

Weusethe *ftfy* library2tocleantherawtextinBooksCorpus, standardizesomepunctuationandwhitespace, andusethe *spaCy* tokenizer.3Fine-tuningdetailsUnlessspecified, wereusethehyperparametersettingsfromunsupervisedpre-training. Weadddropouttotheclassifierwitha rateof0.1. Formosttasks, weusea learningrateof6.25e-5anda batchsizeof32. Ourmodelfinetunesquicklyand3 epochsoftrainingwassufficientformostcases. Weusea linearlearningratedecayschedulewithwarmupover0.2% oftraining. λwassetto0.5.

### 4.2SupervisedFine-TuningWeperformexperimentsona varietyofsupervisedtasksincludingnaturallanguageinference, questionanswering, semanticsimilarity, andtextclassification. SomeofthesetasksareavailableaspartoftherecentlyreleasedGLUEmulti-taskbenchmark [64], whichwemakeuseof. Figure1 providesanoverviewofallthetasksanddatasets. NaturalLanguageInferenceThetaskofnaturallanguageinference (NLI), alsoknownasrecognizingtextualentailment, involvesreadinga pairofsentencesandjudgingtherelationshipbetweenthemfromoneofentailment, contradictionorneutral. Althoughtherehasbeena lotofrecentinterest [58, 35, 44], thetaskremainschallengingduetothepresenceofa widevarietyofphenomenalikelexicalentailment, coreference, andlexicalandsyntacticambiguity. Weevaluateonfivedatasetswithdiversesources, includingimagecaptions (SNLI), transcribedspeech, popularfiction, andgovernmentreports (MNLI), Wikipediaarticles (QNLI), scienceexams (SciTail) ornewsarticles (RTE). Table2 detailsvariousresultsonthedifferentNLItasksforourmodelandpreviousstate-of-the-artapproaches. Ourmethodsignificantlyoutperformsthebaselinesonfourofthefivedatasets, achievingabsoluteimprovementsofupto1.5% onMNLI, 5% onSciTail, 5.8% onQNLIand0.6% onSNLIoverthepreviousbestresults. Thisdemonstratesourmodel'sabilitytobetterreasonovermultiplesentences, andhandleaspectsoflinguisticambiguity. OnRTE, oneofthesmallerdatasetsweevaluateon (2490examples), weachieveanaccuracyof56%, whichisbelowthe61.7% reportedbya multi-taskbiLSTMmodel. GiventhestrongperformanceofourapproachonlargerNLIdatasets, itislikelyourmodelwillbenefitfrommulti-tasktrainingaswellbutwehavenotexploredthiscurrently.

2https://ftfy.readthedocs.io/en/latest/
3https://spacy.io/
Table2: Experimentalresultsonnaturallanguageinferencetasks, comparingourmodelwithcurrentstate-of-the-artmethods. 5xindicatesanensembleof5 models. Alldatasetsuseaccuracyastheevaluationmetric.

| Method | MNLI\-m | MNLI\-mm | SNLI | SciTail | QNLI | RTE |
|-------------------------------------|-----------|------------|--------|-----------|--------|-------|
| ESIM + ELMo [44] (5x) | \- | \- | 89.3 | \- | \- | \- |
| CAFE [58] (5x) | 80.2 | 79.0 | 89.3 | \- | \- | \- |
| StochasticAnswerNetwork [35] (3x) | 80.6 | 80.1 | \- | \- | \- | \- |
| CAFE [58] | 78.7 | 77.9 | 88.5 | 83.3 | | |
| GenSen [64] | 71.4 | 71.3 | \- | \- | 82.3 | 59.2 |
| Multi\-taskBiLSTM + Attn [64] | 72.2 | 72.1 | \- | \- | 82.1 | 61.7 |
| FinetunedTransformerLM (ours) | 82.1 | 81.4 | 89.9 | 88.3 | 88.1 | 56.0 |

Table3: Resultsonquestionansweringandcommonsensereasoning, comparingourmodelwith

| Method | StoryCloze | RACE\-m | RACE\-h | RACE |
|---------------------------------|---------------|-----------|-----------|--------|
| val\-LS\-skip [55] | 76.5 | \- | \- | \- |
| HiddenCoherenceModel [7] | 77.6 | \- | \- | \- |
| DynamicFusionNet [67] (9x) | \- | 55.6 | 49.4 | 51.2 |
| BiAttentionMRU [59] (9x) | \- | 60.2 | 50.3 | 53.3 |
| FinetunedTransformerLM (ours) | 86.5 | 62.9 | 57.4 | 59.0 |

currentstate-of-the-artmethods.. 9xmeansanensembleof9 models.

QuestionansweringandcommonsensereasoningAnothertaskthatrequiresaspectsofsingleandmulti-sentencereasoningisquestionanswering. WeusetherecentlyreleasedRACEdataset [30], consistingofEnglishpassageswithassociatedquestionsfrommiddleandhighschoolexams. ThiscorpushasbeenshowntocontainmorereasoningtypequestionsthatotherdatasetslikeCNN [19] orSQuaD [47], providingtheperfectevaluationforourmodelwhichistrainedtohandlelong-rangecontexts. Inaddition, weevaluateontheStoryClozeTest [40], whichinvolvesselectingthecorrectendingtomulti-sentencestoriesfromtwooptions. Onthesetasks, ourmodelagainoutperformsthepreviousbestresultsbysignificantmargins - upto8.9% onStoryCloze, and5.7% overallonRACE. Thisdemonstratestheabilityofourmodeltohandlelong-rangecontextseffectively.

SemanticSimilaritySemanticsimilarity (orparaphrasedetection) tasksinvolvepredictingwhethertwosentencesaresemanticallyequivalentornot. Thechallengeslieinrecognizingrephrasingofconcepts, understandingnegation, andhandlingsyntacticambiguity. Weusethreedatasetsforthistask - theMicrosoftParaphrasecorpus (MRPC) [14] (collectedfromnewssources), theQuoraQuestionPairs (QQP) dataset [9], andtheSemanticTextualSimilaritybenchmark (STS-B) [6]. Weobtainstate-of-the-artresultsontwoofthethreesemanticsimilaritytasks (Table4) witha 1pointabsolutegainonSTS-B. TheperformancedeltaonQQPissignificant, witha 4.2% absoluteimprovementoverSingle-taskBiLSTM + ELMo + Attn.

ClassificationFinally, wealsoevaluateontwodifferenttextclassificationtasks. TheCorpusofLinguisticAcceptability (CoLA) [65] containsexpertjudgementsonwhethera sentenceisgrammaticalornot, andteststheinnatelinguisticbiasoftrainedmodels. TheStanfordSentimentTreebank (SST-2) [54], ontheotherhand, isa standardbinaryclassificationtask. Ourmodelobtainsanscoreof45.4onCoLA, whichisanespeciallybigjumpoverthepreviousbestresultof35.0, showcasingtheinnatelinguisticbiaslearnedbyourmodel. Themodelalsoachieves91.3% accuracyonSST-2, whichiscompetitivewiththestate-of-the-artresults. Wealsoachieveanoverallscoreof72.8ontheGLUEbenchmark, whichissignificantlybetterthanthepreviousbestof68.9.

Table4: Semanticsimilarityandclassificationresults, comparingourmodelwithcurrentstate-of-theartmethods. AlltaskevaluationsinthistableweredoneusingtheGLUEbenchmark. (mc= Mathewscorrelation, acc=Accuracy, pc=Pearsoncorrelation)

| Method | Classification | | | SemanticSimilarity | | GLUE |
|----------------------------------------|------------------|-------|------|-----------------------|------|--------|
| | CoLA | SST2 | MRPC | STSB | QQP | |
| | (mc) | (acc) | (F1) | (pc) | (F1) | |
| SparsebytemLSTM [16] | \- | 93.2 | \- | \- | \- | \- |
| TF\-KLD [23] | \- | \- | 86.0 | \- | \- | \- |
| ECNU (mixedensemble) [60] | \- | \- | \- | 81.0 | \- | \- |
| Single\-taskBiLSTM + ELMo + Attn [64] | 35.0 | 90.2 | 80.2 | 55.5 | 66.1 | 64.8 |
| Multi\-taskBiLSTM + ELMo + Attn [64] | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| FinetunedTransformerLM (ours) | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 72.8 |

Overall, ourapproachachievesnewstate-of-the-artresultsin9 outofthe12datasetsweevaluateon, outperformingensemblesinmanycases. Ourresultsalsoindicatethatourapproachworkswellacrossdatasetsofdifferentsizes, fromsmallerdatasetssuchasSTS-B (≈5.7ktrainingexamples) - tothelargestone - SNLI (≈550ktrainingexamples).

# 5AnalysisImpactofnumberoflayerstransferredWeobservedtheimpactoftransferringa variablenumber

![6_image_0.png](6_image_0.png)

oflayersfromunsupervisedpre-trainingtothesupervisedtargettask. Figure2(left) illustratestheperformanceofourapproachonMultiNLIandRACEasa functionofthenumberoflayerstransferred.

Weobservethestandardresultthattransferringembeddingsimprovesperformanceandthateachtransformerlayerprovidesfurtherbenefitsupto9% forfulltransferonMultiNLI. Thisindicatesthateachlayerinthepre-trainedmodelcontainsusefulfunctionalityforsolvingtargettasks.

Figure2: (**left**) Effectoftransferringincreasingnumberoflayersfromthepre-trainedlanguagemodelonRACEandMultiNLI. (**right**) Plotshowingtheevolutionofzero-shotperformanceondifferenttasksasa functionofLMpre-trainingupdates. Performancepertaskisnormalizedbetweena randomguessbaselineandthecurrentstate-of-the-artwitha singlemodel.

Zero-shotBehaviorsWe'dliketobetterunderstandwhylanguagemodelpre-trainingoftransformersiseffective. Ahypothesisisthattheunderlyinggenerativemodellearnstoperformmanyofthetasksweevaluateoninordertoimproveitslanguagemodelingcapabilityandthatthemorestructured

| Method | Avg. Score | CoLA | SST2 | MRPC | STSB | QQP | MNLI | QNLI | RTE |
|-------------------------------|--------------|--------|--------|--------|--------|-------|--------|--------|-------|
| | | (mc) | (acc) | (F1) | (pc) | (F1) | (acc) | (acc) | (acc) |
| Transformerw/ auxLM (full) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| Transformerw/opre\-training | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| Transformerw/oauxLM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTMw/ auxLM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

Table5: Analysisofvariousmodelablationsondifferenttasks. Avg. scoreisa unweightedaverageofalltheresults. (mc= Mathewscorrelation, acc=Accuracy, pc=Pearsoncorrelation)

attentionalmemoryofthetransformerassistsintransfercomparedtoLSTMs. Wedesigneda seriesofheuristicsolutionsthatusetheunderlyinggenerativemodeltoperformtaskswithoutsupervisedfinetuning. Wevisualizetheeffectivenessoftheseheuristicsolutionsoverthecourseofgenerativepre-traininginFig2(right). Weobservetheperformanceoftheseheuristicsisstableandsteadilyincreasesovertrainingsuggestingthatgenerativepretrainingsupportsthelearningofa widevarietyoftaskrelevantfunctionality. WealsoobservetheLSTMexhibitshighervarianceinitszero-shotperformancesuggestingthattheinductivebiasoftheTransformerarchitectureassistsintransfer.

ForCoLA (linguisticacceptability), examplesarescoredastheaveragetokenlog-probabilitythegenerativemodelassignsandpredictionsaremadebythresholding. ForSST-2 (sentimentanalysis),
weappendthetoken *very* toeachexampleandrestrictthelanguagemodel'soutputdistributiontoonlythewords *positive* and *negative* andguessthetokenitassignshigherprobabilitytoastheprediction. ForRACE (questionanswering), wepicktheanswerthegenerativemodelassignsthehighestaveragetokenlog-probabilitywhenconditionedonthedocumentandquestion. ForDPRD [46] (winogradschemas), wereplacethedefinitepronounwiththetwopossiblereferrentsandpredicttheresolutionthatthegenerativemodelassignshigheraveragetokenlog-probabilitytotherestofthesequenceafterthesubstitution. AblationstudiesWeperformthreedifferentablationstudies (Table5). First, weexaminetheperformanceofourmethodwithouttheauxiliaryLMobjectiveduringfine-tuning. WeobservethattheauxiliaryobjectivehelpsontheNLItasksandQQP. Overall, thetrendsuggeststhatlargerdatasetsbenefitfromtheauxiliaryobjectivebutsmallerdatasetsdonot. Second, weanalyzetheeffectoftheTransformerbycomparingitwitha singlelayer2048unitLSTMusingthesameframework. Weobservea 5.6averagescoredropwhenusingtheLSTMinsteadoftheTransformer. TheLSTMonlyoutperformstheTransformerononedataset - MRPC. Finally, wealsocomparewithourtransformerarchitecturedirectlytrainedonsupervisedtargettasks, withoutpre-training. Weobservethatthelackofpre-traininghurtsperformanceacrossallthetasks, resultingina 14.8% decreasecomparedtoourfullmodel.

# 6ConclusionWeintroduceda frameworkforachievingstrongnaturallanguageunderstandingwitha singletask-agnosticmodelthroughgenerativepre-traininganddiscriminativefine-tuning. Bypre-trainingona diversecorpuswithlongstretchesofcontiguoustextourmodelacquiressignificantworldknowledgeandabilitytoprocesslong-rangedependencieswhicharethensuccessfullytransferredtosolvingdiscriminativetaskssuchasquestionanswering, semanticsimilarityassessment, entailmentdetermination, andtextclassification, improvingthestateofthearton9 ofthe12datasetswestudy. Usingunsupervised (pre-)trainingtoboostperformanceondiscriminativetaskshaslongbeenanimportantgoalofMachineLearningresearch. Ourworksuggeststhatachievingsignificantperformancegainsisindeedpossible, andoffershintsastowhatmodels (Transformers) anddatasets (textwithlongrangedependencies) workbestwiththisapproach. Wehopethatthiswillhelpenablenewresearchintounsupervisedlearning, forbothnaturallanguageunderstandingandotherdomains, furtherimprovingourunderstandingofhowandwhenunsupervisedlearningworks.

# References

[1] S. Arora, Y. Liang, andT. Ma. Asimplebuttough-to-beatbaselineforsentenceembeddings. 2016.

[2] J. L. Ba, J. R. Kiros, andG. E. Hinton. Layernormalization. *arXivpreprintarXiv:1607.06450*, 2016. [3] Y. Bengio, P. Lamblin, D. Popovici, andH. Larochelle. Greedylayer-wisetrainingofdeepnetworks. InAdvancesinneuralinformationprocessingsystems, pages153-160, 2007.

[4] L. Bentivogli, P. Clark, I. Dagan, andD. Giampiccolo. Thefifthpascalrecognizingtextualentailmentchallenge. InTAC, 2009.

[5] S. R. Bowman, G. Angeli, C. Potts, andC. D. Manning. Alargeannotatedcorpusforlearningnaturallanguageinference. *EMNLP*, 2015.

[6] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, andL. Specia. Semeval-2017task1: Semantictextualsimilarity-multilingualandcross-lingualfocusedevaluation. *arXivpreprintarXiv:1708.00055*, 2017.

[7] S. Chaturvedi, H. Peng, andD. Roth. Storycomprehensionforpredictingwhathappensnext. InProceedingsofthe2017ConferenceonEmpiricalMethodsinNaturalLanguageProcessing, pages1603-1614, 2017.

[8] D. ChenandC. Manning. Afastandaccuratedependencyparserusingneuralnetworks. In *Proceedings* ofthe2014conferenceonempiricalmethodsinnaturallanguageprocessing (EMNLP), pages740-750, 2014.

[9] Z. Chen, H. Zhang, X. Zhang, andL. Zhao. Quoraquestionpairs. https://data.quora.com/First-Quora-
Dataset-Release-Question-Pairs, 2018.

[10] R. CollobertandJ. Weston. Aunifiedarchitecturefornaturallanguageprocessing: Deepneuralnetworkswithmultitasklearning. In *Proceedingsofthe25thinternationalconferenceonMachinelearning*, pages160-167. ACM, 2008.

[11] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, andP. Kuksa. Naturallanguageprocessing
(almost) fromscratch. *JournalofMachineLearningResearch*, 12(Aug):2493-2537, 2011.

[12] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, andA. Bordes. Supervisedlearningofuniversalsentencerepresentationsfromnaturallanguageinferencedata. *EMNLP*, 2017.

[13] A. M. DaiandQ. V. Le. Semi-supervisedsequencelearning. In *AdvancesinNeuralInformationProcessing* Systems, pages3079-3087, 2015.

[14] W. B. DolanandC. Brockett. Automaticallyconstructinga corpusofsententialparaphrases. InProceedingsoftheThirdInternationalWorkshoponParaphrasing (IWP2005), 2005.

[15] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, andS. Bengio. Whydoesunsupervisedpre-traininghelpdeeplearning? *JournalofMachineLearningResearch*, 11(Feb):625-660, 2010.

[16] S. Gray, A. Radford, andK. P. Diederik. Gpukernelsforblock-sparseweights. 2017.

[17] Z. He, S. Liu, M. Li, M. Zhou, L. Zhang, andH. Wang. Learningentityrepresentationforentitydisambiguation. InProceedingsofthe51stAnnualMeetingoftheAssociationforComputationalLinguistics
(Volume2: ShortPapers), volume2, pages30-34, 2013.

[18] D. HendrycksandK. Gimpel. Bridgingnonlinearitiesandstochasticregularizerswithgaussianerrorlinearunits. *arXivpreprintarXiv:1606.08415*, 2016.

[19] K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, andP. Blunsom. Teachingmachinestoreadandcomprehend. In *AdvancesinNeuralInformationProcessingSystems*, pages1693-
1701, 2015.

[20] G. E. Hinton, S. Osindero, andY.-W. Teh. Afastlearningalgorithmfordeepbeliefnets. Neuralcomputation, 18(7):1527-1554, 2006.

[21] J. HowardandS. Ruder. Universallanguagemodelfine-tuningfortextclassification. *Associationfor* ComputationalLinguistics (ACL), 2018.

[22] Y. Jernite, S. R. Bowman, andD. Sontag. Discourse-basedobjectivesforfastunsupervisedsentencerepresentationlearning. *arXivpreprintarXiv:1705.00557*, 2017.

[23] Y. JiandJ. Eisenstein. Discriminativeimprovementstodistributionalsentencesimilarity. InProceedingsofthe2013ConferenceonEmpiricalMethodsinNaturalLanguageProcessing, pages891-896, 2013.

[24] F. Jiao, S. Wang, C.-H. Lee, R. Greiner, andD. Schuurmans. Semi-supervisedconditionalrandomfieldsforimprovedsequencesegmentationandlabeling. InProceedingsofthe21stInternationalConferenceonComputationalLinguisticsandthe44thannualmeetingoftheAssociationforComputationalLinguistics, pages209-216. AssociationforComputationalLinguistics, 2006.

[25] T. Khot, A. Sabharwal, andP. Clark. Scitail: Atextualentailmentdatasetfromsciencequestionanswering.

In *ProceedingsofAAAI*, 2018.

[26] Y. Kim. Convolutionalneuralnetworksforsentenceclassification. *EMNLP*, 2014. [27] D. P. KingmaandJ. Ba. Adam: Amethodforstochasticoptimization. *arXivpreprintarXiv:1412.6980*,
2014.

[28] R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, andS. Fidler. Skip-thoughtvectors. In *Advancesinneuralinformationprocessingsystems*, pages3294-3302, 2015.

[29] N. KitaevandD. Klein. Constituencyparsingwitha self-attentiveencoder. ACL, 2018. [30] G. Lai, Q. Xie, H. Liu, Y. Yang, andE. Hovy. Race: Large-scalereadingcomprehensiondatasetfromexaminations. *EMNLP*, 2017.

[31] G. Lample, L. Denoyer, andM. Ranzato. Unsupervisedmachinetranslationusingmonolingualcorporaonly. *ICLR*, 2018.

[32] Q. LeandT. Mikolov. Distributedrepresentationsofsentencesanddocuments. InInternationalConferenceonMachineLearning, pages1188-1196, 2014.

[33] P. Liang. *Semi-supervisedlearningfornaturallanguage*. PhDthesis, MassachusettsInstituteofTechnology, 2005.

[34] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, andN. Shazeer. Generatingwikipediabysummarizinglongsequences. ICLR, 2018.

[35] X. Liu, K. Duh, andJ. Gao. Stochasticanswernetworksfornaturallanguageinference. arXivpreprintarXiv:1804.07888, 2018.

[36] L. LogeswaranandH. Lee. Anefficientframeworkforlearningsentencerepresentations. *ICLR*, 2018. [37] I. LoshchilovandF. Hutter. Fixingweightdecayregularizationinadam. *arXivpreprintarXiv:1711.05101*,
2017.

[38] B. McCann, J. Bradbury, C. Xiong, andR. Socher. Learnedintranslation: Contextualizedwordvectors. InAdvancesinNeuralInformationProcessingSystems, pages6297-6308, 2017.

[39] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, andJ. Dean. Distributedrepresentationsofwordsandphrasesandtheircompositionality. In *Advancesinneuralinformationprocessingsystems*, pages3111-3119, 2013.

[40] N. Mostafazadeh, M. Roth, A. Louis, N. Chambers, andJ. Allen. Lsdsem2017sharedtask: Thestoryclozetest. In *Proceedingsofthe2ndWorkshoponLinkingModelsofLexical, SententialandDiscourse-level* Semantics, pages46-51, 2017.

[41] K. Nigam, A. McCallum, andT. Mitchell. Semi-supervisedtextclassificationusingem. *Semi-Supervised* Learning, pages33-56, 2006.

[42] J. Pennington, R. Socher, andC. Manning. Glove: Globalvectorsforwordrepresentation. InProceedingsofthe2014conferenceonempiricalmethodsinnaturallanguageprocessing (EMNLP), pages1532-1543, 2014.

[43] M. E. Peters, W. Ammar, C. Bhagavatula, andR. Power. Semi-supervisedsequencetaggingwithbidirectionallanguagemodels. ACL, 2017.

[44] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, andL. Zettlemoyer. Deepcontextualizedwordrepresentations. *NAACL*, 2018.

[45] Y. Qi, D. S. Sachan, M. Felix, S. J. Padmanabhan, andG. Neubig. Whenandwhyarepre-trainedwordembeddingsusefulforneuralmachinetranslation? *NAACL*, 2018.

[46] A. RahmanandV. Ng. Resolvingcomplexcasesofdefinitepronouns: thewinogradschemachallenge. InProceedingsofthe2012JointConferenceonEmpiricalMethodsinNaturalLanguageProcessingandComputationalNaturalLanguageLearning, pages777-789. AssociationforComputationalLinguistics, 2012.

[47] P. Rajpurkar, J. Zhang, K. Lopyrev, andP. Liang. Squad: 100,000+ questionsformachinecomprehensionoftext. *EMNLP*, 2016.

[48] P. Ramachandran, P. J. Liu, andQ. V. Le. Unsupervisedpretrainingforsequencetosequencelearning.

arXivpreprintarXiv:1611.02683, 2016.

[49] M. Ranzato, C. Poultney, S. Chopra, andY. LeCun. Efficientlearningofsparserepresentationswithanenergy-basedmodel. In *Advancesinneuralinformationprocessingsystems*, pages1137-1144, 2007.

[50] M. Rei. Semi-supervisedmultitasklearningforsequencelabeling. ACL, 2017. [51] H. RobbinsandS. Monro. Astochasticapproximationmethod. *Theannalsofmathematicalstatistics*,
pages400-407, 1951.

[52] T. Rocktäschel, E. Grefenstette, K. M. Hermann, T. Kociskˇ y, andP. Blunsom. Reasoningaboutentailment `
withneuralattention. *arXivpreprintarXiv:1509.06664*, 2015.

[53] R. Sennrich, B. Haddow, andA. Birch. Neuralmachinetranslationofrarewordswithsubwordunits. arXivpreprintarXiv:1508.07909, 2015.

[54] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, andC. Potts. Recursivedeepmodelsforsemanticcompositionalityovera sentimenttreebank. InProceedingsofthe2013conferenceonempiricalmethodsinnaturallanguageprocessing, pages1631-1642, 2013.

[55] S. Srinivasan, R. Arora, andM. Riedl. Asimpleandeffectiveapproachtothestoryclozetest. *arXiv* preprintarXiv:1803.05547, 2018.

[56] S. Subramanian, A. Trischler, Y. Bengio, andC. J. Pal. Learninggeneralpurposedistributedsentencerepresentationsvialargescalemulti-tasklearning. *arXivpreprintarXiv:1804.00079*, 2018.

[57] J. SuzukiandH. Isozaki. Semi-supervisedsequentiallabelingandsegmentationusinggiga-wordscaleunlabeleddata. *ProceedingsofACL-08: HLT*, pages665-673, 2008.

[58] Y. Tay, L. A. Tuan, andS. C. Hui. Acompare-propagatearchitecturewithalignmentfactorizationfornaturallanguageinference. *arXivpreprintarXiv:1801.00102*, 2017.

[59] Y. Tay, L. A. Tuan, andS. C. Hui. Multi-rangereasoningformachinecomprehension. arXivpreprintarXiv:1803.09074, 2018.

[60] J. Tian, Z. Zhou, M. Lan, andY. Wu. Ecnuatsemeval-2017task1: Leveragekernel-basedtraditionalnlpfeaturesandneuralnetworkstobuilda universalmodelformultilingualandcross-lingualsemantictextualsimilarity. In *Proceedingsofthe11thInternationalWorkshoponSemanticEvaluation (SemEval-2017)*, pages191-197, 2017.

[61] Y. Tsvetkov. Opportunitiesandchallengesinworkingwithlow-resourcelanguages. CMU, 2017. [62] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, andI. Polosukhin.

Attentionisallyouneed. In *AdvancesinNeuralInformationProcessingSystems*, pages6000-6010, 2017.

[63] P. Vincent, H. Larochelle, Y. Bengio, andP.-A. Manzagol. Extractingandcomposingrobustfeatureswithdenoisingautoencoders. In *Proceedingsofthe25thinternationalconferenceonMachinelearning*, pages1096-1103. ACM, 2008.

[64] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, andS. R. Bowman. Glue: Amulti-taskbenchmarkandanalysisplatformfornaturallanguageunderstanding. *arXivpreprintarXiv:1804.07461*, 2018.

[65] A. Warstadt, A. Singh, andS. R. Bowman. Corpusoflinguisticacceptability. http://nyu-mll.github.io/cola, 2018.

[66] A. Williams, N. Nangia, andS. R. Bowman. Abroad-coveragechallengecorpusforsentenceunderstandingthroughinference. *NAACL*, 2018.

[67] Y. Xu, J. Liu, J. Gao, Y. Shen, andX. Liu. Towardshuman-levelmachinereadingcomprehension:
Reasoningandinferencewithmultiplestrategies. *arXivpreprintarXiv:1711.04964*, 2017.

[68] D. Yu, L. Deng, andG. Dahl. Rolesofpre-trainingandfine-tuningincontext-dependentdbn-hmmsforreal-worldspeechrecognition. InProc. NIPSWorkshoponDeepLearningandUnsupervisedFeatureLearning, 2010.

[69] R. Zhang, P. Isola, andA. A. Efros. Split-brainautoencoders: Unsupervisedlearningbycross-channelprediction. InCVPR, volume1, page6, 2017.

[70] X. Zhu. Semi-supervisedlearningliteraturesurvey. 2005. [71] Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, andS. Fidler. Aligningbooksandmovies: Towardsstory-likevisualexplanationsbywatchingmoviesandreadingbooks. InProceedingsoftheIEEEinternationalconferenceoncomputervision, pages19-27, 2015.