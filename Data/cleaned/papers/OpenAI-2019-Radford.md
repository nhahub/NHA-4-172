# 

AlecRadford * 1JeffreyWu * 1RewonChild1 DavidLuan1 DarioAmodei ** 1**IlyaSutskever** ** 1

# AbstractNaturallanguageprocessingtasks, suchasquestionanswering, machinetranslation, readingcomprehension, andsummarization, aretypicallyapproachedwithsupervisedlearningontaskspecificdatasets. Wedemonstratethatlanguagemodelsbegintolearnthesetaskswithoutanyexplicitsupervisionwhentrainedona newdatasetofmillionsofwebpagescalledWebText. Whenconditionedona documentplusquestions, theanswersgeneratedbythelanguagemodelreach55F1ontheCoQAdataset - matchingorexceedingtheperformanceof3 outof4 baselinesystemswithoutusingthe127,000+ trainingexamples. Thecapacityofthelanguagemodelisessentialtothesuccessofzero-shottasktransferandincreasingitimprovesperformanceina log-linearfashionacrosstasks. Ourlargestmodel, GPT-2, isa 1.5BparameterTransformerthatachievesstateoftheartresultson7 outof8 testedlanguagemodelingdatasetsina zero-shotsettingbutstillunderfitsWebText. Samplesfromthemodelreflecttheseimprovementsandcontaincoherentparagraphsoftext. Thesefindingssuggesta promisingpathtowardsbuildinglanguageprocessingsystemswhichlearntoperformtasksfromtheirnaturallyoccurringdemonstrations.

# 1. IntroductionMachinelearningsystemsnowexcel (inexpectation) attaskstheyaretrainedforbyusinga combinationoflargedatasets, high-capacitymodels, andsupervisedlearning (Krizhevskyetal., 2012) (Sutskeveretal., 2014) (Amodeietal., 2016). Yetthesesystemsarebrittleandsensitivetoslightchangesinthedatadistribution (Rechtetal., 2018) andtaskspecification (Kirkpatricketal., 2017). Currentsystemsarebettercharacterizedasnarrowexpertsratherthancompetentgeneralists. Wewouldliketomovetowardsmoregeneralsystemswhichcanperformmanytasks - eventuallywithouttheneedtomanuallycreateandlabela trainingdatasetforeachone. ThedominantapproachtocreatingMLsystemsistocollecta datasetoftrainingexamplesdemonstratingcorrectbehaviorfora desiredtask, traina systemtoimitatethesebehaviors, andthentestitsperformanceonindependentandidenticallydistributed (IID) held-outexamples. Thishasservedwelltomakeprogressonnarrowexperts. Buttheoftenerraticbehaviorofcaptioningmodels (Lakeetal., 2017), readingcomprehensionsystems (Jia & Liang, 2017), andimageclassifiers (Alcornetal., 2018) onthediversityandvarietyofpossibleinputshighlightssomeoftheshortcomingsofthisapproach. Oursuspicionisthattheprevalenceofsingletasktrainingonsingledomaindatasetsisa majorcontributortothelackofgeneralizationobservedincurrentsystems. Progresstowardsrobustsystemswithcurrentarchitecturesislikelytorequiretrainingandmeasuringperformanceona widerangeofdomainsandtasks. Recently, severalbenchmarkshavebeenproposedsuchasGLUE (Wangetal., 2018) anddecaNLP (McCannetal., 2018) tobeginstudyingthis.

Multitasklearning (Caruana, 1997) isa promisingframeworkforimprovinggeneralperformance. However, multitasktraininginNLPisstillnascent. Recentworkreportsmodestperformanceimprovements (Yogatamaetal., 2019) andthetwomostambitiouseffortstodatehavetrainedona totalof10and17 (dataset, objective) pairsrespectively (McCannetal., 2018) (Bowmanetal., 2018). Froma meta-learningperspective, each (dataset, objective) pairisa singletrainingexamplesampledfromthedistributionofdatasetsandobjectives. CurrentMLsystemsneedhundredstothousandsofexamplestoinducefunctionswhichgeneralizewell. Thissuggeststhatmultitasktrainingmanyneedjustasmanyeffectivetrainingpairstorealizeitspromisewithcurrentapproaches. Itwillbeverydifficulttocontinuetoscalethecreationofdatasetsandthedesignofobjectivestothedegreethatmayberequiredtobruteforceourwaytherewithcurrenttechniques. Thismotivatesexploringadditionalsetupsforperformingmultitasklearning. Thecurrentbestperformingsystemsonlanguagetasks

*, **Equalcontribution1OpenAI, SanFrancisco, California, UnitedStates. Correspondenceto: AlecRadford <alec@openai.com>.

![1_image_0.png](1_image_0.png)

Figure1. Zero-shottaskperformanceofWebTextLMsasa functionofmodelsizeonmanyNLPtasks. ReadingComprehensionresultsareonCoQA (Reddyetal., 2018), translationonWMT-14Fr-En (Artetxeetal., 2017), summarizationonCNNandDailyMail (Seeetal., 2017), andQuestionAnsweringonNaturalQuestions (Kwiatkowskietal., 2019). Section3 containsdetaileddescriptionsofeachresult.

utilizea combinationofpre-trainingandsupervisedfinetuning. Thisapproachhasa longhistorywitha trendtowardsmoreflexibleformsoftransfer. First, wordvectorswerelearnedandusedasinputstotask-specificarchitectures (Mikolovetal., 2013) (Collobertetal., 2011), thenthecontextualrepresentationsofrecurrentnetworksweretransferred (Dai & Le, 2015) (Petersetal., 2018), andrecentworksuggeststhattask-specificarchitecturesarenolongernecessaryandtransferringmanyself-attentionblocksissufficient (Radfordetal., 2018) (Devlinetal., 2018). Thesemethodsstillrequiresupervisedtraininginordertoperforma task. Whenonlyminimalornosuperviseddataisavailable, anotherlineofworkhasdemonstratedthepromiseoflanguagemodelstoperformspecifictasks, suchascommonsensereasoning (Schwartzetal., 2017) andsentimentanalysis (Radfordetal., 2017). Inthispaper, weconnectthesetwolinesofworkandcontinuethetrendofmoregeneralmethodsoftransfer. Wedemonstratelanguagemodelscanperformdown-streamtasksina zero-shotsetting - withoutanyparameterorarchitecturemodification. Wedemonstratethisapproachshowspotentialbyhighlightingtheabilityoflanguagemodelstoperforma widerangeoftasksina zero-shotsetting. Weachievepromising, competitive, andstateoftheartresultsdependingonthetask.

# 2. ApproachAtthecoreofourapproachislanguagemodeling. Languagemodelingisusuallyframedasunsuperviseddistributionestimationfroma setofexamples (x1, x2*, ..., x*n)
eachcomposedofvariablelengthsequencesofsymbols
(s1, s2*, ..., s*n). Sincelanguagehasa naturalsequentialordering, itiscommontofactorizethejointprobabilitiesoversymbolsastheproductofconditionalprobabilities (Jelinek & Mercer, 1980) (Bengioetal., 2003):

$$p(x)=\prod_{i=1}^{n}p(s_{n}|s_{1},...,s_{n-1})\qquad\qquad(1)$$

Thisapproachallowsfortractablesamplingfromandestimationofp(x) aswellasanyconditionalsoftheformp(sn−k, ..., sn|s1*, ..., s*n−k−1). Inrecentyears, therehavebeensignificantimprovementsintheexpressivenessofmodelsthatcancomputetheseconditionalprobabilities, suchasself-attentionarchitecturesliketheTransformer (Vaswanietal., 2017). Learningtoperforma singletaskcanbeexpressedina probabilisticframeworkasestimatinga conditionaldistributionp(output|*input*). Sincea generalsystemshouldbeabletoperformmanydifferenttasks, evenforthesameinput, itshouldconditionnotonlyontheinputbutalsoonthetasktobeperformed. Thatis, itshouldmodelp(output|*input, task*). Thishasbeenvariouslyformalizedinmultitaskandmeta-learningsettings. Taskconditioningisoftenimplementedatanarchitecturallevel, suchasthetaskspecificencodersanddecodersin (Kaiseretal., 2017) oratanalgorithmiclevelsuchastheinnerandouterloopoptimizationframeworkofMAML (Finnetal., 2017). ButasexemplifiedinMcCannetal. (2018), languageprovidesa flexiblewaytospecifytasks, inputs, andoutputsallasa sequenceofsymbols. Forexample, atranslationtrainingexamplecanbewrittenasthesequence (translatetofrench, englishtext, frenchtext). Likewise, areadingcomprehensiontrainingexamplecanbewrittenas (answerthequestion, document, question, answer). McCannetal. (2018) demonstrateditwaspossibletotraina singlemodel, theMQAN,
toinferandperformmanydifferenttasksonexampleswiththistypeofformat. Languagemodelingisalsoableto, inprinciple, learnthetasksofMcCannetal. (2018) withouttheneedforexplicitsupervisionofwhichsymbolsaretheoutputstobepredicted. Sincethesupervisedobjectiveisthethesameastheunsupervisedobjectivebutonlyevaluatedona subsetofthesequence, theglobalminimumoftheunsupervisedobjectiveisalsotheglobalminimumofthesupervisedobjective. Inthisslightlytoysetting, theconcernswithdensityestimationasa principledtrainingobjectivediscussedin (Sutskeveretal., 2015) aresidestepped. Theprobleminsteadbecomeswhetherweareableto, inpractice, optimizetheunsupervisedobjectivetoconvergence. Preliminaryexperimentsconfirmedthatsufficientlylargelanguagemodelsareabletoperformmultitasklearninginthistoy-ishsetupbutlearningismuchslowerthaninexplicitlysupervisedapproaches. Whileitisa largestepfromthewell-posedsetupdescribedabovetothemessinessof "languageinthewild", Weston (2016) argues, inthecontextofdialog, fortheneedtodevelopsystemscapableoflearningfromnaturallanguagedirectlyanddemonstrateda proofofconcept - learninga QAtaskwithouta rewardsignalbyusingforwardpredictionofa teacher'soutputs. Whiledialogisanattractiveapproach, weworryitisoverlyrestrictive. Theinternetcontainsa vastamountofinformationthatispassivelyavailablewithouttheneedforinteractivecommunication. Ourspeculationisthata languagemodelwithsufficientcapacitywillbegintolearntoinferandperformthetasksdemonstratedinnaturallanguagesequencesinordertobetterpredictthem, regardlessoftheirmethodofprocurement. Ifa languagemodelisabletodothisitwillbe, ineffect, performingunsupervisedmultitasklearning. Wetestwhetherthisisthecasebyanalyzingtheperformanceoflanguagemodelsina zero-shotsettingona widevarietyoftasks.

### 2.1. TrainingDatasetMostpriorworktrainedlanguagemodelsona singledomainoftext, suchasnewsarticles (Jozefowiczetal., 2016),
Wikipedia (Merityetal., 2016), orfictionbooks (Kirosetal., 2015). Ourapproachmotivatesbuildingaslargeanddiversea datasetaspossibleinordertocollectnaturallanguagedemonstrationsoftasksinasvariedofdomainsandcontextsaspossible. ApromisingsourceofdiverseandnearlyunlimitedtextiswebscrapessuchasCommonCrawl. Whilethesearchivesaremanyordersofmagnitudelargerthancurrentlanguagemodelingdatasets, theyhavesignificantdataqualityissues. Trinh & Le (2018) usedCommonCrawlintheirworkoncommonsensereasoningbutnoteda largeamountofdocuments "whosecontentaremostlyunintelligible". Weobservedsimilardataissuesinourinitialexperimentswith
"I'mnotthecleverestmanintheworld, butliketheysayinFrench: **Jenesuispasunimbecile [I'mnota fool].** Ina now-deletedpostfromAug. 16, SoheilEid, TorycandidateintheridingofJoliette, wroteinFrench: "**Mentezmentez,** ilenresteratoujoursquelquechose," whichtranslatesas,
"**Lielieandsomethingwillalwaysremain.**"
"Ihatetheword '**perfume**,"' Burrsays. 'It'ssomewhatbetterinFrench: '**parfum**.' Iflistenedcarefullyat29:55, aconversationcanbeheardbetweentwoguysinFrench: "**-Commentonfaitpouraller** del'autrecote? -Quelautrecot ´ e?´ ", whichmeans "**- How** doyougettotheotherside? - Whatside?". Ifthissoundslikea bitofa stretch, considerthisquestioninFrench: **As-tualleraucinema?** ´ , or **Didyougoto**
themovies?, whichliterallytranslatesasHave-youtogotomovies/theater? "**BrevetSansGarantieDuGouvernement**", translatedtoEnglish: "**Patentedwithoutgovernmentwarranty**".

Table1. ExamplesofnaturallyoccurringdemonstrationsofEnglishtoFrenchandFrenchtoEnglishtranslationfoundthroughouttheWebTexttrainingset.

CommonCrawl. Trinh & Le (2018)'sbestresultswereachievedusinga smallsubsampleofCommonCrawlwhichincludedonlydocumentsmostsimilartotheirtargetdataset, theWinogradSchemaChallenge. Whilethisisa pragmaticapproachtoimproveperformanceona specifictask, wewanttoavoidmakingassumptionsaboutthetaskstobeperformedaheadoftime. Instead, wecreateda newwebscrapewhichemphasizesdocumentquality. Todothisweonlyscrapedwebpageswhichhavebeencurated/filteredbyhumans. Manuallyfilteringa fullwebscrapewouldbeexceptionallyexpensivesoasa startingpoint, wescrapedalloutboundlinksfromReddit, asocialmediaplatform, whichreceivedatleast3 karma. Thiscanbethoughtofasa heuristicindicatorforwhetherotherusersfoundthelinkinteresting, educational, orjustfunny. Theresultingdataset, WebText, containsthetextsubsetofthese45millionlinks. ToextractthetextfromHTMLresponsesweusea combinationoftheDragnet (Peters &
Lecocq, 2013) andNewspaper1contentextractors. Allresultspresentedinthispaperusea preliminaryversionofWebTextwhichdoesnotincludelinkscreatedafterDec2017andwhichafterde-duplicationandsomeheuristicbasedcleaningcontainsslightlyover8 milliondocumentsfora totalof40GBoftext. WeremovedallWikipediadocumentsfromWebTextsinceitisa commondatasourceforotherdatasetsandcouldcomplicateanalysisduetoover-

1https://github.com/codelucas/newspaperlappingtrainingdatawithtestevaluationtasks.

#### 2.2. InputRepresentationA generallanguagemodel (LM) shouldbeabletocomputetheprobabilityof (andalsogenerate) anystring. CurrentlargescaleLMsincludepre-processingstepssuchaslowercasing, tokenization, andout-of-vocabularytokenswhichrestrictthespaceofmodel-ablestrings. WhileprocessingUnicodestringsasa sequenceofUTF-8byteselegantlyfulfillsthisrequirementasexemplifiedinworksuchasGillicketal. (2015), currentbyte-levelLMsarenotcompetitivewithword-levelLMsonlargescaledatasetssuchastheOneBillionWordBenchmark (Al-Rfouetal., 2018). Weobserveda similarperformancegapinourownattemptstotrainstandardbyte-levelLMsonWebText. BytePairEncoding (BPE) (Sennrichetal., 2015) isa practicalmiddlegroundbetweencharacterandwordlevellanguagemodelingwhicheffectivelyinterpolatesbetweenwordlevelinputsforfrequentsymbolsequencesandcharacterlevelinputsforinfrequentsymbolsequences. Despiteitsname, referenceBPEimplementationsoftenoperateonUnicodecodepointsandnotbytesequences. TheseimplementationswouldrequireincludingthefullspaceofUnicodesymbolsinordertomodelallUnicodestrings. Thiswouldresultina basevocabularyofover130,000beforeanymulti-symboltokensareadded. Thisisprohibitivelylargecomparedtothe32,000to64,000tokenvocabulariesoftenusedwithBPE. Incontrast, abyte-levelversionofBPEonlyrequiresa basevocabularyofsize256. However, directlyapplyingBPEtothebytesequenceresultsinsuboptimalmergesduetoBPEusinga greedyfrequencybasedheuristicforbuildingthetokenvocabulary. WeobservedBPEincludingmanyversionsofcommonwordslikedogsincetheyoccurinmanyvariationssuchasdog. dog! dog? . Thisresultsina sub-optimalallocationoflimitedvocabularyslotsandmodelcapacity. Toavoidthis, wepreventBPEfrommergingacrosscharactercategoriesforanybytesequence. Weaddanexceptionforspaceswhichsignificantlyimprovesthecompressionefficiencywhileaddingonlyminimalfragmentationofwordsacrossmultiplevocabtokens.

Thisinputrepresentationallowsustocombinetheempiricalbenefitsofword-levelLMswiththegeneralityofbyte-levelapproaches. Sinceourapproachcanassigna probabilitytoanyUnicodestring, thisallowsustoevaluateourLMsonanydatasetregardlessofpre-processing, tokenization, orvocabsize.

### 2.3. ModelWeusea Transformer (Vaswanietal., 2017) basedarchitectureforourLMs. ThemodellargelyfollowsthedetailsoftheOpenAIGPTmodel (Radfordetal., 2018) witha

| Parameters | Layers | dmodel |
|--------------|----------|----------|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| 1542M | 48 | 1600 |

Table2. Architecturehyperparametersforthe4 modelsizes.

fewmodifications. Layernormalization (Baetal., 2016) wasmovedtotheinputofeachsub-block, similartoa pre-activationresidualnetwork (Heetal., 2016) andanadditionallayernormalizationwasaddedafterthefinalselfattentionblock. Amodifiedinitializationwhichaccountsfortheaccumulationontheresidualpathwithmodeldepthisused. Wescaletheweightsofresiduallayersatinitializationbya factorof1/
√NwhereN isthenumberofresiduallayers. Thevocabularyisexpandedto50,257. Wealsoincreasethecontextsizefrom512to1024tokensanda largerbatchsizeof512isused.

# 3. ExperimentsWetrainedandbenchmarkedfourLMswithapproximatelylog-uniformlyspacedsizes. ThearchitecturesaresummarizedinTable2. ThesmallestmodelisequivalenttotheoriginalGPT, andthesecondsmallestequivalenttothelargestmodelfromBERT (Devlinetal., 2018). Ourlargestmodel, whichwecallGPT-2, hasoveranorderofmagnitudemoreparametersthanGPT. Thelearningrateofeachmodelwasmanuallytunedforthebestperplexityona 5%
held-outsampleofWebText. AllmodelsstillunderfitWeb-
Textandheld-outperplexityhasasofyetimprovedgivenmoretrainingtime.

### 3.1. LanguageModelingAsaninitialsteptowardszero-shottasktransfer, weareinterestedinunderstandinghowWebTextLM'sperformatzero-shotdomaintransferontheprimarytasktheyaretrainedfor - languagemodeling. Sinceourmodeloperatesona bytelevelanddoesnotrequirelossypre-processingortokenization, wecanevaluateitonanylanguagemodelbenchmark. Resultsonlanguagemodelingdatasetsarecommonlyreportedina quantitywhichisa scaledorexponentiatedversionoftheaveragenegativelogprobabilitypercanonicalpredictionunit - usuallya character, abyte, ora word. Weevaluatethesamequantitybycomputingthelog-probabilityofa datasetaccordingtoa WebTextLManddividingbythenumberofcanonicalunits. Formanyofthesedatasets, WebTextLMswouldbetestedsignificantlyoutof-distribution, havingtopredictaggressivelystandardizedtext, tokenizationartifactssuchasdisconnectedpunctuationandcontractions, shuffledsentences, andeventhestring

| | LAMBADA | LAMBADA | CBT\-CN | CBT\-NE | WikiText2 | PTB | enwik8 | text8 | WikiText103 | 1BW |
|-------|-----------|-----------|-----------|-----------|-------------|-------|----------|---------|---------------|--------|
| | (PPL) | (ACC) | (ACC) | (ACC) | (PPL) | (PPL) | (BPB) | (BPC) | (PPL) | (PPL) |
| SOTA | 99.8 | 59.23 | 85.7 | 82.3 | 39.14 | 46.54 | 0.99 | 1.08 | 18.3 | 21.8 |
| 117M | 35.13 | 45.99 | 87.65 | 83.4 | 29.41 | 65.85 | 1.16 | 1.17 | 37.50 | 75.20 |
| 345M | 15.60 | 55.48 | 92.35 | 87.1 | 22.76 | 47.33 | 1.01 | 1.06 | 26.37 | 55.72 |
| 762M | 10.87 | 60.12 | 93.45 | 88.0 | 19.93 | 40.31 | 0.97 | 1.02 | 22.05 | 44.575 |
| 1542M | 8.63 | 63.24 | 93.30 | 89.05 | 18.34 | 35.76 | 0.93 | 0.98 | 17.48 | 42.16 |

Table3. Zero-shotresultsonmanydatasets. Notrainingorfine-tuningwasperformedforanyoftheseresults. PTBandWikiText-2resultsarefrom (Gongetal., 2018). CBTresultsarefrom (Bajgaretal., 2016). LAMBADAaccuracyresultisfrom (Hoangetal., 2018) andLAMBADAperplexityresultisfrom (Graveetal., 2016). Otherresultsarefrom (Daietal., 2019).

<UNK> whichisextremelyrareinWebText - occurringonly26timesin40billionbytes. WereportourmainresultsinTable3 usinginvertiblede-tokenizerswhichremoveasmanyofthesetokenization / pre-processingartifactsaspossible. Sincethesede-tokenizersareinvertible, wecanstillcalculatethelogprobabilityofa datasetandtheycanbethoughtofasa simpleformofdomainadaptation. Weobservegainsof2.5to5 perplexityforGPT-2withthesede-tokenizers.

WebTextLMstransferwellacrossdomainsanddatasets, improvingthestateofthearton7 outofthe8 datasetsina zero-shotsetting. LargeimprovementsarenoticedonsmalldatasetssuchasPennTreebankandWikiText-2whichhaveonly1 to2 milliontrainingtokens. Largeimprovementsarealsonoticedondatasetscreatedtomeasurelong-termdependencieslikeLAMBADA (Papernoetal., 2016) andtheChildren'sBookTest (Hilletal., 2015). OurmodelisstillsignificantlyworsethanpriorworkontheOneBillionWordBenchmark (Chelbaetal., 2013). Thisislikelyduetoa combinationofitbeingboththelargestdatasetandhavingsomeofthemostdestructivepre-processing - 1BW'ssentencelevelshufflingremovesalllong-rangestructure.

3.2. Children'sBookTest

![4_image_0.png](4_image_0.png)

Figure2. PerformanceontheChildren'sBookTestasa functionofmodelcapacity. HumanperformancearefromBajgaretal. (2016), insteadofthemuchlowerestimatesfromtheoriginalpaper.

TheChildren'sBookTest (CBT) (Hilletal., 2015) wascreatedtoexaminetheperformanceofLMsondifferentcategoriesofwords: namedentities, nouns, verbs, andprepositions. Ratherthanreportingperplexityasanevaluationmetric, CBTreportsaccuracyonanautomaticallyconstructedclozetestwherethetaskistopredictwhichof10possiblechoicesforanomittedwordiscorrect. FollowingtheLMapproachintroducedintheoriginalpaper, wecomputetheprobabilityofeachchoiceandtherestofthesentenceconditionedonthischoiceaccordingtotheLM, andpredicttheonewiththehighestprobability. AsseeninFigure2 performancesteadilyimprovesasmodelsizeisincreasedandclosesthemajorityofthegaptohumanperformanceonthistest. DataoverlapanalysisshowedoneoftheCBTtestsetbooks, TheJungleBookbyRudyardKipling, isinWebText, sowereportresultsonthevalidationsetwhichhasnosignificantoverlap. GPT-2achievesnewstateoftheartresultsof93.3% oncommonnounsand89.1% onnamedentities. Ade-tokenizerwasappliedtoremovePTBstyletokenizationartifactsfromCBT.

### 3.3. LambadaTheLAMBADAdataset (Papernoetal., 2016) teststheabilityofsystemstomodellong-rangedependenciesintext. Thetaskistopredictthefinalwordofsentenceswhichrequireatleast50tokensofcontextfora humantosuccessfullypredict. GPT-2improvesthestateoftheartfrom99.8 (Graveetal., 2016) to8.6perplexityandincreasestheaccuracyofLMsonthistestfrom19% (Dehghanietal., 2018) to52.66%. InvestigatingGPT-2'serrorsshowedmostpredictionsarevalidcontinuationsofthesentence, butarenotvalidfinalwords. ThissuggeststhattheLMisnotusingtheadditionalusefulconstraintthatthewordmustbethefinalofthesentence. Addinga stop-wordfilterasanapproximationtothisfurtherincreasesaccuracyto63.24%, improvingtheoverallstateoftheartonthistaskby4%. Thepreviousstateoftheart (Hoangetal., 2018) useda differentrestrictedpredictionsettingwheretheoutputsofthemodelwereconstrainedtoonlywordsthatappearedinthecontext. ForGPT-2, thisrestrictionisharmfulratherthanhelpfulsince19% ofanswersarenotincontext. Weusea versionofthedatasetwithoutpreprocessing.

3.4. WinogradSchemaChallenge

![5_image_0.png](5_image_0.png)

Figure3. PerformanceontheWinogradSchemaChallengeasa functionofmodelcapacity.

TheWinogradSchemachallenge (Levesqueetal., 2012) wasconstructedtomeasurethecapabilityofa systemtoperformcommonsensereasoningbymeasuringitsabilitytoresolveambiguitiesintext. RecentlyTrinh & Le (2018) demonstratedsignificantprogressonthischallengeusingLMs, bypredictingtheresolutionoftheambiguitywithhigherprobability. WefollowtheirproblemformulationandvisualizetheperformanceofourmodelswithbothfullandpartialscoringtechniquesinFigure3. GPT-2improvesstateoftheartaccuracyby7%, achieving70.70%. Thedatasetisquitesmallwithonly273examplessowerecommendreadingTrichelairetal. (2018) tohelpcontextualizethisresult.

### 3.5. ReadingComprehensionTheConversationQuestionAnsweringdataset (CoQA) Reddyetal. (2018) consistsofdocumentsfrom7 differentdomainspairedwithnaturallanguagedialoguesbetweena questionaskeranda questionanswereraboutthedocument. CoQAtestsreadingcomprehensioncapabilitiesandalsotheabilityofmodelstoanswerquestionsthatdependonconversationhistory (suchas "Why?"). GreedydecodingfromGPT-2whenconditionedona document, thehistoryoftheassociatedconversation, anda finaltokenA: achieves55F1onthedevelopmentset. Thismatchesorexceedstheperformanceof3 outof4 baselinesystemswithoutusingthe127,000+ manuallycollectedquestionanswerpairsthosebaselinesweretrainedon. ThesupervisedSOTA, aBERTbasedsystem (Devlinetal.,

| | R\-1 | R\-2 | R\-L | R\-AVG |
|----------------|--------|--------|--------|----------|
| Bottom\-UpSum | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede\-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| GPT\-2TL;DR: | 29.34 | 8.27 | 26.58 | 21.40 |
| Random\-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT\-2nohint | 21.58 | 4.03 | 19.47 | 15.03 |

Table4. SummarizationperformanceasmeasuredbyROUGEF1metricsontheCNNandDailyMaildataset. Bottom-UpSumistheSOTAmodelfrom (Gehrmannetal., 2018)

2018), isnearingthe89F1performanceofhumans. WhileGPT-2'sperformanceisexcitingfora systemwithoutanysupervisedtraining, someinspectionofitsanswersanderrorssuggestsGPT-2oftenusessimpleretrievalbasedheuristicssuchas *answerwitha namefromthedocumentinresponse* toa whoquestion.

### 3.6. SummarizationWetestGPT-2'sabilitytoperformsummarizationontheCNNandDailyMaildataset (Nallapatietal., 2016). ToinducesummarizationbehaviorweaddthetextTL;DR: afterthearticleandgenerate100tokenswithTop-krandomsampling (Fanetal., 2018) withk = 2whichreducesrepetitionandencouragesmoreabstractivesummariesthangreedydecoding. Weusethefirst3 generatedsentencesinthese100tokensasthesummary. Whilequalitativelythegenerationsresemblesummaries, asshowninTable14, theyoftenfocusonrecentcontentfromthearticleorconfusespecificdetailssuchashowmanycarswereinvolvedina crashorwhethera logowasona hatorshirt. OnthecommonlyreportedROUGE1,2,Lmetricsthegeneratedsummariesonlybegintoapproachtheperformanceofclassicneuralbaselinesandjustbarelyoutperformsselecting3 randomsentencesfromthearticle. GPT-2'sperformancedropsby6.4pointsontheaggregatemetricwhenthetaskhintisremovedwhichdemonstratestheabilitytoinvoketaskspecificbehaviorina languagemodelwithnaturallanguage.

### 3.7. TranslationWetestwhetherGPT-2hasbeguntolearnhowtotranslatefromonelanguagetoanother. Inordertohelpitinferthatthisisthedesiredtask, weconditionthelanguagemodelona contextofexamplepairsoftheformatenglishsentence = frenchsentenceandthenaftera finalpromptofenglishsentence = wesamplefromthemodelwithgreedydecodingandusethefirstgeneratedsentenceasthetranslation. OntheWMT-14English-Frenchtestset, GPT-2gets5 BLEU, whichisslightlyworsethana word-by-wordsubstitutionwitha bilinguallexiconinferredinpreviousworkonunsupervisedwordtranslation

| Question | GeneratedAnswer | Correct | Probability |
|-----------------------------------------------------------------------------------|-----------------------|-----------|---------------|
| Whowrotethebooktheoriginofspecies? | CharlesDarwin | ✓ | 83.4% |
| Whoisthefounderoftheubuntuproject? | MarkShuttleworth | ✓ | 82.0% |
| Whoisthequarterbackforthegreenbaypackers? | AaronRodgers | ✓ | 81.1% |
| Pandaisa nationalanimalofwhichcountry? | China | ✓ | 76.8% |
| Whocameupwiththetheoryofrelativity? | AlbertEinstein | ✓ | 76.4% |
| Whenwasthefirststarwarsfilmreleased? | 1977 | ✓ | 71.4% |
| Whatisthemostcommonbloodtypeinsweden? | A | ✗ | 70.6% |
| Whoisregardedasthefounderofpsychoanalysis? | SigmundFreud | ✓ | 69.3% |
| Whotookthefirststepsonthemoonin1969? | NeilArmstrong | ✓ | 66.8% |
| Whoisthelargestsupermarketchainintheuk? | Tesco | ✓ | 65.3% |
| Whatisthemeaningofshalominenglish? | peace | ✓ | 64.0% |
| Whowastheauthoroftheartofwar? | SunTzu | ✓ | 59.6% |
| Largeststateintheusbylandmass? | California | ✗ | 59.2% |
| Greenalgaeisanexampleofwhichtypeofreproduction? | parthenogenesis | ✗ | 56.5% |
| Vikramsamvatcalenderisofficialinwhichcountry? | India | ✓ | 55.6% |
| Whoismostlyresponsibleforwritingthedeclarationofindependence? | ThomasJefferson | ✓ | 53.3% |
| Whatusstateformsthewesternboundaryofmontana? | Montana | ✗ | 52.3% |
| Whoplaysserdavosingameofthrones? | PeterDinklage | ✗ | 52.1% |
| Whoappointsthechairofthefederalreservesystem? | JanetYellen | ✗ | 51.5% |
| Statetheprocessthatdividesonenucleusintotwogeneticallyidenticalnuclei? | mitosis | ✓ | 50.7% |
| Whowonthemostmvpawardsinthenba? | MichaelJordan | ✗ | 50.2% |
| Whatriverisassociatedwiththecityofrome? | theTiber | ✓ | 48.6% |
| Whoisthefirstpresidenttobeimpeached? | AndrewJohnson | ✓ | 48.3% |
| Whoistheheadofthedepartmentofhomelandsecurity2017? | JohnKelly | ✓ | 47.0% |
| Whatisthenamegiventothecommoncurrencytotheeuropeanunion? | Euro | ✓ | 46.8% |
| Whatwastheemperornameinstarwars? | Palpatine | ✓ | 46.5% |
| Doyouhavetohavea gunpermittoshootata range? | No | ✓ | 46.4% |
| Whoproposedevolutionin1859asthebasisofbiologicaldevelopment? | CharlesDarwin | ✓ | 45.7% |
| Nuclearpowerplantthatblewupinrussia? | Chernobyl | ✓ | 45.7% |
| Whoplayedjohnconnorintheoriginalterminator? | ArnoldSchwarzenegger | ✗ | 45.2% |

Table5. The30mostconfidentanswersgeneratedbyGPT-2onthedevelopmentsetofNaturalQuestionssortedbytheirprobabilityaccordingtoGPT-2. NoneofthesequestionsappearinWebTextaccordingtotheproceduredescribedinSection4.

(Conneauetal., 2017b). OntheWMT-14French-Englishtestset, GPT-2isabletoleverageitsverystrongEnglishlanguagemodeltoperformsignificantlybetter, achieving11.5BLEU. Thisoutperformsseveralunsupervisedmachinetranslationbaselinesfrom (Artetxeetal., 2017) and (Lampleetal., 2017) butisstillmuchworsethanthe33.5BLEUofthecurrentbestunsupervisedmachinetranslationapproach (Artetxeetal., 2019). Performanceonthistaskwassurprisingtous, sincewedeliberatelyremovednon-EnglishwebpagesfromWebTextasa filteringstep. Inordertoconfirmthis, werana byte-levellanguagedetector2onWebTextwhichdetectedonly10MBofdataintheFrenchlanguagewhichisapproximately500xsmallerthanthemonolingualFrenchcorpuscommoninpriorunsupervisedmachinetranslationresearch.

#### 3.8. QuestionAnsweringA potentialwaytotestwhatinformationiscontainedwithina languagemodelistoevaluatehowoftenitgeneratesthecorrectanswertofactoid-stylequestions. Previousshowcasingofthisbehaviorinneuralsystemswhereallinformationisstoredinparameterssuchas *ANeuralConversational* Model (Vinyals & Le, 2015) reportedqualitativeresultsduetothelackofhigh-qualityevaluationdatasets. TherecentlyintroducedNaturalQuestionsdataset (Kwiatkowskietal.,
2019) isa promisingresourcetotestthismorequantitatively. Similartotranslation, thecontextofthelanguagemodelisseededwithexamplequestionanswerpairswhichhelpsthemodelinfertheshortanswerstyleofthedataset. GPT-2answers4.1% ofquestionscorrectlywhenevaluatedbytheexactmatchmetriccommonlyusedonreadingcomprehensiondatasetslikeSQUAD.3Asa comparisonpoint, thesmallestmodeldoesnotexceedthe1.0% accuracyofanincrediblysimplebaselinewhichreturnsthemostcommonanswerforeachquestiontype (who, what, where, etc...). GPT-2answers5.3timesmorequestionscorrectly, suggestingthatmodelcapacityhasbeena majorfactorinthepoorperformanceofneuralsystemsonthiskindoftaskasofyet. TheprobabilityGPT-2assignstoitsgeneratedanswersiswellcalibratedandGPT-2hasanaccuracyof63.1% onthe1% ofquestionsitismostconfidentin. The30mostconfidentanswersgeneratedbyGPT-2ondevelopmentsetquestionsareshowninTable5. TheperformanceofGPT-2isstillmuch, much, worsethanthe30to50% rangeofopendomainquestionansweringsystemswhichhybridizeinformationretrievalwithextractivedocumentquestionanswering (Albertietal., 2019).

2https://github.com/CLD2Owners/cld23Alec, whopreviouslythoughtofhimselfasgoodatrandomtrivia, answered17of100randomlysampledexamplescorrectlywhentestedinthesamesettingasGPT-2. Heactuallyonlygot14rightbutheshouldhavegottenthoseother3

| | PTB | WikiText\-2 | enwik8 | text8 | Wikitext\-103 | 1BW |
|---------------|-------|---------------|----------|---------|-----------------|--------|
| Datasettrain | 2.67% | 0.66% | 7.50% | 2.34% | 9.09% | 13.19% |
| WebTexttrain | 0.88% | 1.63% | 6.31% | 3.94% | 2.42% | 3.75% |

Table6. Percentageoftestset8 gramsoverlappingwithtrainingsets.

# 4. GeneralizationVsMemorization

### GaveAwayTheAnswer.

Recentworkincomputervisionhasshownthatcommonimagedatasetscontaina non-trivialamountofnear-duplicateimages. ForinstanceCIFAR-10has3.3% overlapbetweentrainandtestimages (Barz & Denzler, 2019). Thisresultsinanover-reportingofthegeneralizationperformanceofmachinelearningsystems. Asthesizeofdatasetsincreasesthisissuebecomesincreasinglylikelywhichsuggestsa similarphenomenacouldbehappeningwithWebText. Thereforeitisimportanttoanalyzehowmuchtestdataalsoshowsupinthetrainingdata. TostudythiswecreatedBloomfilterscontaining8-gramsofWebTexttrainingsettokens. Toimproverecall, stringswerenormalizedtocontainonlylower-casedalphanumericwordswitha singlespaceasa delimiter. TheBloomfilterswereconstructedsuchthatthefalsepositiverateisupperboundedby1 108 . Wefurtherverifiedthelowfalsepositiveratebygenerating1Mstrings, ofwhichzerowerefoundbythefilter.

TheseBloomfiltersletuscalculate, givena dataset, thepercentageof8-gramsfromthatdatasetthatarealsofoundintheWebTexttrainingset. Table6 showsthisoverlapanalysisforthetestsetsofcommonLMbenchmarks. CommonLMdatasets' testsetshavebetween1-6% overlapwithWeb- Texttrain, withanaverageofoverlapof3.2%. Somewhatsurprisingly, manydatasetshavelargeroverlapswiththeirowntrainingsplits, withanaverageof5.9% overlap. Ourapproachoptimizesforrecall, andwhilemanualinspectionoftheoverlapsshowsmanycommonphrases, therearemanylongermatchesthatareduetoduplicateddata. ThisisnotuniquetoWebText. Forinstance, wediscoveredthatthetestsetofWikiText-103hasanarticlewhichisalsointhetrainingdataset. Sincethereareonly60articlesinthetestsetthereisatleastanoverlapof1.6%.4Potentiallymoreworryingly, 1BWhasanoverlapofnearly13.2% withitsowntrainingsetaccordingtoourprocedure. FortheWinogradSchemaChallenge, wefoundonly10schematawhichhadany8-gramoverlapswiththeWebTexttrainingset. Ofthese, 2werespuriousmatches. Oftheremaining8, only1 schemaappearedinanycontextsthatForCoQA, about15% ofdocumentsinthenewsdomainarealreadyinWebTextandthemodelperformsabout3 F1betteronthese. CoQA'sdevelopmentsetmetricreportstheaverageperformanceover5 differentdomainsandwemeasurea gainofabout0.5-1.0F1duetooverlapacrossthevariousdomains. However, noactualtrainingquestionsoranswersareinWebTextsinceCoQAwasreleasedafterthecutoffdateforlinksinWebText.

OnLAMBADA, theaverageoverlapis1.2%. GPT-2performsabout2 perplexitybetteronexampleswithgreaterthan15% overlap. Recalculatingmetricswhenexcludingallexampleswithanyoverlapshiftsresultsfrom8.6to8.7perplexityandreducesaccuracyfrom63.2% to62.9%. Thisverysmallchangeinoverallresultsislikelyduetoonly1 in200exampleshavingsignificantoverlap. Overall, ouranalysissuggeststhatdataoverlapbetweenWebTexttrainingdataandspecificevaluationdatasetsprovidesa smallbutconsistentbenefittoreportedresults. However, formostdatasetswedonotnoticesignificantlylargeroverlapsthanthosealreadyexistingbetweenstandardtrainingandtestsets, asTable6 highlights. Understandingandquantifyinghowhighlysimilartextimpactsperformanceisanimportantresearchquestion. Betterde-duplicationtechniquessuchasscalablefuzzymatchingcouldalsohelpbetteranswerthesequestions. Fornow, werecommendtheuseofn-gramoverlapbasedde-duplicationasanimportantverificationstepandsanitycheckduringthecreationoftrainingandtestsplitsfornewNLPdatasets.

AnotherpotentialwayofdeterminingwhethertheperformanceofWebTextLMsisattributabletomemorizationisinspectingtheirperformanceontheirownheld-outset. AsshowninFigure4, performanceonboththetrainingandtestsetsofWebTextaresimilarandimprovetogetherasmodelsizeisincreased. ThissuggestsevenGPT-2isstillunderfittingonWebTextinmanyways. GPT-2isalsoabletowritenewsarticlesaboutthediscoveryoftalkingunicorns. AnexampleisprovidedinTable13.

# 5. RelatedWorkA significantportionofthisworkmeasuredtheperformanceoflargerlanguagemodelstrainedonlargerdatasets. This4Asignificantportionofadditionaloverlapisduetoeditorsreusingsomeparagraphsacrossmultiplearticleswitha sharedthemesuchasvariousbattlesintheKoreanWar.

![8_image_0.png](8_image_0.png)

Figure4. TheperformanceofLMstrainedonWebTextasa functionofmodelsize.

issimilartotheworkofJozefowiczetal. (2016) whichscaledRNNbasedlanguagemodelsonthe1 BillionWordBenchmark. Bajgaretal. (2016) alsopreviouslyimprovedresultsontheChildren'sBookTestbycreatinga muchlargertrainingdatasetoutofProjectGutenbergtosupplementthestandardtrainingdataset. Hestnessetal. (2017) conducteda thoroughanalysisofhowtheperformanceofvariousdeeplearningmodelschangesasa functionofbothmodelcapacityanddatasetsize. Ourexperiments, whilemuchnoisieracrosstasks, suggestsimilartrendsholdforsub-tasksofanobjectiveandcontinueintothe1B+ parameterregime. InterestinglearnedfunctionalityingenerativemodelshasbeendocumentedbeforesuchasthecellsinanRNNlanguagemodelperformingline-widthtrackingandquote/commentdetectionKarpathyetal. (2015). MoreinspirationaltoourworkwastheobservationofLiuetal. (2018) thata modeltrainedtogenerateWikipediaarticlesalsolearnedtotranslatenamesbetweenlanguages. Previousworkhasexploredalternativeapproachestofilteringandconstructinga largetextcorpusofwebpages, suchastheiWebCorpus (Davies, 2018). Therehasbeenextensiveworkonpre-trainingmethodsforlanguagetasks. Inadditiontothosementionedintheintroduction, GloVe (Penningtonetal., 2014) scaledwordvectorrepresentationlearningtoallofCommonCrawl. Aninfluentialearlyworkondeeprepresentationlearningfortextwas *Skip-thoughtVectors* (Kirosetal., 2015). McCannetal. (2017) exploredtheuseofrepresentationsderivedfrommachinetranslationmodelsandHoward & Ruder (2018)
improvedtheRNNbasedfine-tuningapproachesof (Dai & Le, 2015). (Conneauetal., 2017a) studiedthetransferperformanceofrepresentationslearnedbynaturallanguageinferencemodelsand (Subramanianetal., 2018) exploredlarge-scalemultitasktraining. (Ramachandranetal., 2016) demonstratedthatseq2seqmodelsbenefitfrombeinginitializedwithpre-trainedlanguagemodelsasencodersanddecoders. MorerecentworkhasshownthatLMpre-trainingishelpfulwhenfine-tunedfordifficultgenerationtaskslikechit-chatdialoganddialogbasedquestionansweringsystemsaswell (Wolfetal., 2019) (Dinanetal., 2018).

# 6. DiscussionMuchresearchhasbeendedicatedtolearning (Hilletal., 2016), understanding (Levy & Goldberg, 2014), andcriticallyevaluating (Wieting & Kiela, 2019) therepresentationsofbothsupervisedandunsupervisedpre-trainingmethods. Ourresultssuggestthatunsupervisedtasklearningisanadditionalpromisingareaofresearchtoexplore. Thesefindingspotentiallyhelpexplainthewidespreadsuccessofpre-trainingtechniquesfordown-streamNLPtasksasweshowthat, inthelimit, oneofthesepre-trainingtechniquesbeginstolearntoperformtasksdirectlywithouttheneedforsupervisedadaptionormodification. OnreadingcomprehensiontheperformanceofGPT-2iscompetitivewithsupervisedbaselinesina zero-shotsetting. However, onothertaskssuchassummarization, whileitisqualitativelyperformingthetask, itsperformanceisstillonlyrudimentaryaccordingtoquantitativemetrics. Whilesuggestiveasa researchresult, intermsofpracticalapplications, thezero-shotperformanceofGPT-2isstillfarfromuse-able. Wehavestudiedthezero-shotperformanceofWebTextLMsonmanycanonicalNLPtasks, buttherearemanyadditionaltasksthatcouldbeevaluated. ThereareundoubtedlymanypracticaltaskswheretheperformanceofGPT-2isstillnobetterthanrandom. Evenoncommontasksthatweevaluatedon, suchasquestionansweringandtranslation, languagemodelsonlybegintooutperformtrivialbaselineswhentheyhavesufficientcapacity.

Whilezero-shotperformanceestablishesa baselineofthepotentialperformanceofGPT-2onmanytasks, itisnotclearwheretheceilingiswithfinetuning. Onsometasks, GPT-2'sfullyabstractiveoutputisa significantdeparturefromtheextractivepointernetwork (Vinyalsetal., 2015) basedoutputswhicharecurrentlystateoftheartonmanyquestionansweringandreadingcomprehensiondatasets. Giventhepriorsuccessoffine-tuningGPT, weplantoinvestigatefine-tuningonbenchmarkssuchasdecaNLPandGLUE, especiallysinceitisunclearwhethertheadditionaltrainingdataandcapacityofGPT-2issufficienttoovercometheinefficienciesofuni-directionalrepresentationsdemonstratedbyBERT (Devlinetal., 2018).

# 7. ConclusionWhena largelanguagemodelistrainedona sufficientlylargeanddiversedatasetitisabletoperformwellacrossmanydomainsanddatasets. GPT-2zero-shotstostateoftheartperformanceon7 outof8 testedlanguagemodelingdatasets. Thediversityoftasksthemodelisabletoperformina zero-shotsettingsuggeststhathigh-capacitymodelstrainedtomaximizethelikelihoodofa sufficientlyvariedtextcorpusbegintolearnhowtoperforma surprisingamountoftaskswithouttheneedforexplicitsupervision.5

# AcknowledgementsThankstoeveryonewhowrotethetext, sharedthelinks, andupvotedthecontentinWebText. ManymillionsofpeoplewereinvolvedincreatingthedatathatGPT-2wastrainedon. AlsothankstoalltheGooglerswhohelpeduswithtraininginfrastructure, includingZakStone, JSRiehl, JonathanHseu, RussellPower, YoulongCheng, NoamShazeer, SolomonBoulos, MichaelBanfield, AmanGupta, DanielSohn, andmanymore. Finallythankstothepeoplewhogavefeedbackondraftsofthepaper: JacobSteinhardt, SamBowman, GeoffreyIrving, andMadisonMay.

# ReferencesAl-Rfou, R., Choe, D., Constant, N., Guo, M., andJones, L.

Character-levellanguagemodelingwithdeeperself-attention. arXivpreprintarXiv:1808.04444, 2018.

Alberti, C., Lee, K., andCollins, M. Abertbaselineforthenaturalquestions. *arXivpreprintarXiv:1901.08634*, 2019.

Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.-S., andNguyen, A. Strike (with) apose: Neuralnetworksareeasilyfooledbystrangeposesoffamiliarobjects. *arXivpreprint* arXiv:1811.11553, 2018.

Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., Casper, J., Catanzaro, B., Cheng, Q., Chen, G., etal. Deepspeech2: End-to-endspeechrecognitioninenglishandmandarin. InInternationalConferenceonMachineLearning, pp. 173-182, 2016.

Artetxe, M., Labaka, G., Agirre, E., andCho, K. Unsupervisedneuralmachinetranslation. *arXivpreprintarXiv:1710.11041*,
2017.

Artetxe, M., Labaka, G., andAgirre, E. Aneffectiveapproachtounsupervisedmachinetranslation. *arXivpreprint* arXiv:1902.01313, 2019.

5Preliminarycodefordownloadingandusingthesmallmodelisavailableathttps://github.com/openai/gpt-2Ba, J. L., Kiros, J. R., andHinton, G. E. Layernormalization.

arXivpreprintarXiv:1607.06450, 2016.

Bajgar, O., Kadlec, R., andKleindienst, J. Embracingdataabundance: Booktestdatasetforreadingcomprehension. arXivpreprintarXiv:1610.00956, 2016.

Barz, B. andDenzler, J. Dowetrainontestdata? purgingcifarofnear-duplicates. *arXivpreprintarXiv:1902.00423*, 2019.

Bengio, Y., Ducharme, R., Vincent, P., andJauvin, C. Aneuralprobabilisticlanguagemodel. *Journalofmachinelearning* research, 3(Feb):1137-1155, 2003.

Bowman, S. R., Pavlick, E., Grave, E., VanDurme, B., Wang, A.,
Hula, J., Xia, P., Pappagari, R., McCoy, R. T., Patel, R., etal. Lookingforelmo'sfriends: Sentence-levelpretrainingbeyondlanguagemodeling. *arXivpreprintarXiv:1812.10860*, 2018.

Caruana, R. Multitasklearning. *Machinelearning*, 28(1):41-75, 1997.

Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., andRobinson, T. Onebillionwordbenchmarkformeasuringprogressinstatisticallanguagemodeling. arXivpreprintarXiv:1312.3005, 2013.

Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., andKuksa, P. Naturallanguageprocessing (almost) fromscratch. *JournalofMachineLearningResearch*, 12(Aug):2493-
2537, 2011.

Conneau, A., Kiela, D., Schwenk, H., Barrault, L., andBordes, A. Supervisedlearningofuniversalsentencerepresentationsfromnaturallanguageinferencedata. *arXivpreprint* arXiv:1705.02364, 2017a.

Conneau, A., Lample, G., Ranzato, M., Denoyer, L., andJegou, ´
H. Wordtranslationwithoutparalleldata. *arXivpreprint* arXiv:1710.04087, 2017b.

Dai, A. M. andLe, Q. V. Semi-supervisedsequencelearning. InAdvancesinneuralinformationprocessingsystems, pp. 3079- 3087, 2015.

Dai, Z., Yang, Z., Yang, Y., Cohen, W. W., Carbonell, J., Le, Q. V., andSalakhutdinov, R. Transformer-xl: Attentivelanguagemodelsbeyonda fixed-lengthcontext. arXivpreprintarXiv:1901.02860, 2019.

Davies, M. The14billionwordiwebcorpus.

https://corpus.byu.edu/iWeb/, 2018.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., andKaiser, Ł. Universaltransformers. *arXivpreprintarXiv:1807.03819*, 2018.

Devlin, J., Chang, M.-W., Lee, K., andToutanova, K. Bert: Pretrainingofdeepbidirectionaltransformersforlanguageunderstanding. *arXivpreprintarXiv:1810.04805*, 2018.

Dinan, E., Roller, S., Shuster, K., Fan, A., Auli, M., andWeston, J. Wizardofwikipedia: Knowledge-poweredconversationalagents. *arXivpreprintarXiv:1811.01241*, 2018.

Fan, A., Lewis, M., andDauphin, Y. Hierarchicalneuralstorygeneration. *arXivpreprintarXiv:1805.04833*, 2018.

Finn, C., Abbeel, P., andLevine, S. Model-agnosticmetalearningforfastadaptationofdeepnetworks. arXivpreprintarXiv:1703.03400, 2017.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska- Barwinska, A., etal. Overcomingcatastrophicforgettinginneuralnetworks. Proceedingsofthenationalacademyofsciences, pp. 201611835, 2017.

Gehrmann, S., Deng, Y., andRush, A. M. Bottom-upabstractivesummarization. *arXivpreprintarXiv:1808.10792*, 2018.

Kiros, R., Zhu, Y., Salakhutdinov, R. R., Zemel, R., Urtasun, R.,
Torralba, A., andFidler, S. Skip-thoughtvectors. InAdvancesinneuralinformationprocessingsystems, pp. 3294-3302, 2015.

Gillick, D., Brunk, C., Vinyals, O., andSubramanya, A. Multilinguallanguageprocessingfrombytes. *arXivpreprint* arXiv:1512.00103, 2015.

Krizhevsky, A., Sutskever, I., andHinton, G. E. Imagenetclassificationwithdeepconvolutionalneuralnetworks. InAdvancesinneuralinformationprocessingsystems, pp. 1097-1105, 2012.

Gong, C., He, D., Tan, X., Qin, T., Wang, L., andLiu, T.-Y. Frage:
frequency-agnosticwordrepresentation. InAdvancesinNeuralInformationProcessingSystems, pp. 1341-1352, 2018.

Kwiatkowski, T., Palomaki, J., Rhinehart, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Kelcey, M., Devlin, J., etal. Naturalquestions: abenchmarkforquestionansweringresearch. 2019.

Grave, E., Joulin, A., andUsunier, N. Improvingneurallanguagemodelswitha continuouscache. arXivpreprintarXiv:1612.04426, 2016.

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., andGershman, S. J.

Buildingmachinesthatlearnandthinklikepeople. *Behavioral* andBrainSciences, 40, 2017.

He, K., Zhang, X., Ren, S., andSun, J. Identitymappingsindeepresidualnetworks. In *Europeanconferenceoncomputervision*,
pp. 630-645. Springer, 2016.

Lample, G., Conneau, A., Denoyer, L., andRanzato, M. Unsupervisedmachinetranslationusingmonolingualcorporaonly. arXivpreprintarXiv:1711.00043, 2017.

Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., andZhou, Y. Deeplearningscalingispredictable, empirically. *arXivpreprint* arXiv:1712.00409, 2017.

Levesque, H., Davis, E., andMorgenstern, L. Thewinogradschemachallenge. In *ThirteenthInternationalConferenceon* thePrinciplesofKnowledgeRepresentationandReasoning, 2012.

Hill, F., Bordes, A., Chopra, S., andWeston, J. Thegoldilocksprinciple: Readingchildren'sbookswithexplicitmemoryrepresentations. *arXivpreprintarXiv:1511.02301*, 2015.

Levy, O. andGoldberg, Y. Neuralwordembeddingasimplicitmatrixfactorization. InAdvancesinneuralinformationprocessingsystems, pp. 2177-2185, 2014.

Hill, F., Cho, K., andKorhonen, A. Learningdistributedrepresentationsofsentencesfromunlabelleddata. *arXivpreprint* arXiv:1602.03483, 2016.

Liu, P. J., Saleh, M., Pot, E., Goodrich, B., Sepassi, R., Kaiser, L.,
andShazeer, N. Generatingwikipediabysummarizinglongsequences. *arXivpreprintarXiv:1801.10198*, 2018.

Hoang, L., Wiseman, S., andRush, A. M. Entitytrackingimprovescloze-stylereadingcomprehension. arXivpreprintarXiv:1810.02891, 2018.

McCann, B., Bradbury, J., Xiong, C., andSocher, R. Learnedintranslation: Contextualizedwordvectors. InAdvancesinNeuralInformationProcessingSystems, pp. 6294-6305, 2017.

Howard, J. andRuder, S. Universallanguagemodelfine-tuningfortextclassification. InProceedingsofthe56thAnnualMeetingoftheAssociationforComputationalLinguistics (Volume1: LongPapers), volume1, pp. 328-339, 2018.

McCann, B., Keskar, N. S., Xiong, C., andSocher, R. Thenaturallanguagedecathlon: Multitasklearningasquestionanswering. arXivpreprintarXiv:1806.08730, 2018.

Jelinek, F. andMercer, R. L. Interpolatedestimationofmarkovsourceparametersfromsparsedata. InProceedingsoftheWorkshoponPatternRecognitioninPractice, Amsterdam, TheNetherlands: North-Holland, May., 1980.

Merity, S., Xiong, C., Bradbury, J., andSocher, R. Pointersentinelmixturemodels. *arXivpreprintarXiv:1609.07843*, 2016.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., andDean, J. Distributedrepresentationsofwordsandphrasesandtheircompositionality. InAdvancesinneuralinformationprocessingsystems, pp. 3111-3119, 2013.

Jia, R. andLiang, P. Adversarialexamplesforevaluatingreadingcomprehensionsystems. *arXivpreprintarXiv:1707.07328*, 2017.

Nallapati, R., Zhou, B., Gulcehre, C., Xiang, B., etal. Abstractivetextsummarizationusingsequence-to-sequencernnsandbeyond. *arXivpreprintarXiv:1602.06023*, 2016.

Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., andWu, Y. Exploringthelimitsoflanguagemodeling. arXivpreprintarXiv:1602.02410, 2016.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., andFernandez, R. The ´ lambadadataset: Wordpredictionrequiringa broaddiscoursecontext. *arXivpreprintarXiv:1606.06031*, 2016.

Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N.,
Jones, L., andUszkoreit, J. Onemodeltolearnthemall. arXivpreprintarXiv:1706.05137, 2017.

Pennington, J., Socher, R., andManning, C. Glove: Globalvectorsforwordrepresentation. InProceedingsofthe2014conferenceonempiricalmethodsinnaturallanguageprocessing (EMNLP), pp. 1532-1543, 2014.

Karpathy, A., Johnson, J., andFei-Fei, L. Visualizingandunderstandingrecurrentnetworks. *arXivpreprintarXiv:1506.02078*, 2015.

Peters, M. E. andLecocq, D. Contentextractionusingdiversefeaturesets. InProceedingsofthe22ndInternationalConferenceonWorldWideWeb, pp. 89-90. ACM, 2013.

Vinyals, O., Fortunato, M., andJaitly, N. Pointernetworks. InAdvancesinNeuralInformationProcessingSystems, pp. 2692- 2700, 2015.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C.,
Lee, K., andZettlemoyer, L. Deepcontextualizedwordrepresentations. *arXivpreprintarXiv:1802.05365*, 2018.

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., andBowman, S. R. Glue: Amulti-taskbenchmarkandanalysisplatformfornaturallanguageunderstanding. arXivpreprintarXiv:1804.07461, 2018.

Radford, A., Jozefowicz, R., andSutskever, I. Learningtogeneratereviewsanddiscoveringsentiment. arXivpreprintarXiv:1704.01444, 2017.

Weston, J. E. Dialog-basedlanguagelearning. InAdvancesinNeuralInformationProcessingSystems, pp. 829-837, 2016.

Radford, A., Narasimhan, K., Salimans, T., andSutskever, I.

Improvinglanguageunderstandingbygenerativepre-training. 2018.

Wieting, J. andKiela, D. Notrainingrequired: Exploringrandomencodersforsentenceclassification. arXivpreprintarXiv:1901.10444, 2019.

Ramachandran, P., Liu, P. J., andLe, Q. V. Unsupervisedpretrainingforsequencetosequencelearning. arXivpreprintarXiv:1611.02683, 2016.

Wolf, T., Sanh, V., Chaumond, J., andDelangue, C. Transfertransfo: Atransferlearningapproachforneuralnetworkbasedconversationalagents. *arXivpreprintarXiv:1901.08149*, 2019.

Recht, B., Roelofs, R., Schmidt, L., andShankar, V. Docifar-10classifiersgeneralizetocifar-10? arXivpreprintarXiv:1806.00451, 2018.

Yogatama, D., d'Autume, C. d. M., Connor, J., Kocisky, T.,
Chrzanowski, M., Kong, L., Lazaridou, A., Ling, W., Yu, L., Dyer, C., etal. Learningandevaluatinggenerallinguisticintelligence. *arXivpreprintarXiv:1901.11373*, 2019.

Reddy, S., Chen, D., andManning, C. D. Coqa: Aconversationalquestionansweringchallenge. *arXivpreprintarXiv:1808.07042*, 2018.

Schwartz, R., Sap, M., Konstas, I., Zilles, L., Choi, Y., andSmith, N. A. Storyclozetask: Uwnlpsystem. InProceedingsofthe2ndWorkshoponLinkingModelsofLexical, SententialandDiscourse-levelSemantics, pp. 52-55, 2017.

See, A., Liu, P. J., andManning, C. D. Gettothepoint: Summarizationwithpointer-generatornetworks. *arXivpreprint* arXiv:1704.04368, 2017.

Sennrich, R., Haddow, B., andBirch, A. Neuralmachinetranslationofrarewordswithsubwordunits. arXivpreprintarXiv:1508.07909, 2015.

Subramanian, S., Trischler, A., Bengio, Y., andPal, C. J. Learninggeneralpurposedistributedsentencerepresentationsvialargescalemulti-tasklearning. *arXivpreprintarXiv:1804.00079*, 2018.

Sutskever, I., Vinyals, O., andLe, Q. V. Sequencetosequencelearningwithneuralnetworks. InAdvancesinneuralinformationprocessingsystems, pp. 3104-3112, 2014.

Sutskever, I., Jozefowicz, R., Gregor, K., Rezende, D., Lillicrap, T., andVinyals, O. Towardsprincipledunsupervisedlearning. arXivpreprintarXiv:1511.06440, 2015.

Trichelair, P., Emami, A., Cheung, J. C. K., Trischler, A., Suleman, K., andDiaz, F. Ontheevaluationofcommon-sensereasoninginnaturallanguageunderstanding. *arXivpreprint* arXiv:1811.01778, 2018.

Trinh, T. H. andLe, Q. V. Asimplemethodforcommonsensereasoning. *arXivpreprintarXiv:1806.02847*, 2018.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, Ł., andPolosukhin, I. Attentionisallyouneed. InAdvancesinNeuralInformationProcessingSystems, pp. 5998-6008, 2017.

Vinyals, O. andLe, Q. Aneuralconversationalmodel. arXivpreprintarXiv:1506.05869, 2015.

# 8. AppendixA: Samples

### 8.1. ModelCapacityTocomplementthereportedperplexitygainsofbiggerLMsonWebTextshowinFigure4, Tables7 through11showside-by-sidecompletionsofthesmallestWebTextLMandGPT-2onrandomunseenWebTexttestsetarticles.

### 8.2. TextMemorizationWeobservesomememorizingbehaviorinGPT-2onlongerstringsthatarerepeatedmanytimesinthedatasetsuchasfamousquotesorspeeches. Forexample, whenconditionedonthefirstsentenceanda halfoftheGettysburgAddress (whichoccursapproximately40timesthroughoutWebText), anargmaxdecodefromGPT-2recoversthespeech. Evenwhensamplingwithouttruncation, wefindthatthemodelcopiesthespeechforawhilebeforedrifting, albeitina similarstyle. Ittypicallydriftswithin100-200tokens, anddisplayswideningdiversityonceitdrifts.

Toquantifyhowoftenexactmemorizationshowsupinsamples,

![12_image_0.png](12_image_0.png) wegeneratedsamplesfromGPT-2conditionedonWebTexttestsetarticlesandcomparedtheoverlapratesofGPT-2'sgenerationstotheoverlapratesoftheground-truthcompletions. TheresultsofthisanalysisareshownbelowandsuggestthatGPT-2repeatstextfromthetrainingsetlessoftenthenthebaselinerateofheld-outarticles.

### 8.3. DiversityTable12showsmultiplecompletionsofthesamerandomWebTexttestsetcontext, showingthediversityofcompletionswithstandardsamplingsettings.

### 8.4. RobustnessTable13showsthepreviouslymentionedtalkingunicornsnewsarticle. Wefindthemodeltobecapableofhandlingoutofdistributioncontexts, butthequalityofthesesamplesisgenerallylower.

Figure5. CDFofpercentage8-gramoverlapwithWebTexttrainingset, forbothWebTexttestsetandsamples (conditionedonWebTexttestset, withtop-ktruncatedrandomsamplingwithk = 40). Mostsampleshavelessthan1% overlap, includingover30% ofsampleswithnooverlap, whereasthemedianfortestsetis2.6% overlap.

Outfit1: Typical

### Context (WebtextTest)

ThispairingwasthefirstoutfitI thoughtofwhenI boughttheshoes. It'slikea summerversionofthisJakeGranthamoutfit; infact, myshoesareclosetothecolorsofhisNikeRacers! Insteadofa heavyHarrisTweedjacketanddenimshirt, I'mwearinga cottonDBjacketandanda linenshirt. Bothfabrics (inthesecolors) areanabsolutemustforsummer, astheygowithbothdarkandandlightpants! Asyoucansee, theypairwonderfullywiththedarkjeansandshoes. It'sa pseudomenswear/prepoutfit. Overall, thisisa verycasualoutfitwhichiswhyI pairedmysneakerswithit. I'mnotaboutwearinga fullwoolsuitwithsneakers (asGQshowsa lot) butI'mdefinitelyopentokeepingthingscasual, likethiscottonDB. Casualfabricsarekeytopullingoffyoursneakersina dresseddownmenswearoutfit. I'devensuggesttowearthesesneakerswitha khakichinosuitora whitelinensuit. Justbesuretoditchthetieorweara teeorpolo;

| wearinga tiewithsneakersisa bittoomuch | |
|--------------------------------------------------------------|--------------------------------------------------------------|
| Smallmodelcompletion | GPT\-2completion |
| . | forme! |
| Outfit2: HighEnd | Outfit2: CasualJacket |
| Thisoutfitwasjustintheeyeofthestorm. It'sa | Iknowthatsomepeoplemightbeopposedtowearing |
| hybridofwhatASICSdoeswithstreetwear (likeJ. Crew) | sneakersina turtleneck, butI wantedtobetruetomyself, |
| andwhatJ. Crewdoeswithcasualpiecesthatarea bit | soI wentwitha slimfittingturtleneckina colormore |
| moretraditional. Again, asI mentionedbefore, it'sa low | similartomyfavoritecoloroftheday. Overall, it'sa |
| budgetitemandtheycanbeveryexpensive. Thesewill | classicturtleneck, withnothingtooflashy. Ididswapmy |
| definitelybeanissueifyou'retryingtoworkasa casual | shoesfortheseASICSTigerKillshotsandI wishI hadn't |
| sneakermaker, andthat'sOKbutdon'thesitateifyou | didthat. Theshoeitselfisverycomfortable, however, |
| wanttotrythelowerpriceversionsoftheiroutfits! | Ifoundthatwearingtheminthesesneakersmademy |
| | feetlooka bitlarger. I'dwearthemwithdenim, chinos, |
| Thisversionfeaturesa verylowcut\-cutneckline | whatever, butwouldneverrecommendwearingthem |
| (whichisnota badthing), slightlylongerlegs, anda pair | alone. Therearejusttoomanyvariablesinvolved. |
| oflonglegstokeepthingsnice. Mypreferredcolorof | |
| colortheseitemsareblue\-gray (whichismyfirstcolors, | Outfit3: Dress |
| astheyarea lightwhite), orblack (whichismyfavorite | |
| ofmydesigns). Asalways, mypickofcolorsfromthis | ThisisthefinaloutfitI woretheASICSTiger |
| storehavenothingtodowiththatotherthanthede | Killshots. ThisversionoftheASICSisa muchmore |
| sign. Fora highend, Iwouldrecommendthistoeveryone! | modernandstreamlinedversion. Theshoehasthemost |
| | modernandminimaldesignthatI canthinkof! Witha lot |
| Outfit3: HighLight | ofblack, thesesneakerslookverycool. They'lldefinitely |
| | beverycomfortableforsummer. |
| Thiswasa "highlight" lookwhileI hada chance | |
| togetsomecolor. Ithinkthecolorisbasedon | Tome, wearingthemwitha suitistheeasiestway |
| | tohavea goodpairofsneakershoes. That'sprettymuch |
| | it! If |

Table7. Randomunseencontexts (top), andnon-cherry-pickedcompletionsfromboththesmallest (left) andlargest (right) models. Contextsare768tokens, withapproximately256tokensworthofparagraphsshown. Completionsare256tokensandfullyshown. Top-krandomsamplingwithk = 40wasusedforgeneration.

### Context (WebtextTest)

SomeofthemostglorioushistoricalattractionsinSpaindatefromtheperiodofMuslimrule, includingTheMezquita, builtastheGreatMosqueofCordobaandtheMedinaAzahara, alsoinC ´ ordobaandnowinruinsbut ´ stillvisitableassuchandbuiltastheMadinatal-Zahra, thePalaceofal-Andalus; andtheAlhambrainGranada, asplendid, intactpalace. TherearealsotwosynagoguesstillstandingthatwerebuiltduringtheeraofMuslimSpain: SantaMar´ıalaBlancainToledoandtheSynagogueofCordoba, intheOldCity. ´

| ReconquistaandImperialera [ edit ] | |
|---------------------------------------------------------------|--------------------------------------------------------------|
| Smallmodelcompletion | GPT\-2completion |
| TheeraofMuslimSpainisverydifferentfromthatof | TheReconquista (literally, "Thereturntothecountry") |
| otherpartsoftheworld, witha ratherextensivemedieval | of1092and1095sawtheestablishmentintheIberian |
| history. WhiletheSpanishEmpireexistedduringthe | peninsulaofa unitedkingdomofCastileandAragonin |
| secondhalfofthe9thcenturyBCitwasquiteslowly | whicha Muslimdynastywouldexist. Thiswasthelast |
| transformedtoa small, secularizedempireby1150A.D., | timeinEuropewhereMuslimrulewouldbeconsidered |
| ina muchmoreprosperousanddiversestate. In1050CE | "native" andthelastoccasionwhereChristian\-inflamed |
| theemperorofNaples, atleastpartiallyincharacterisedby | andorganizedconflictswouldmakethemselvesknownin |
| aseriesofwarsandinvasionsfromthenorth\-westerncoast | Europe. |
| tothewesterncoastoftheregion (e.g., ontheDanube | |
| RiverwhichextendedallthewayfromGibraltarintothe | ThereignofMuslimMoorsinSpainisoftenre |
| southofSpain, butwasquicklydestroyedonthesame | ferredto, especiallyamongnon\-Muslims, asthe |
| daybytheRomanarmy), assumedthetitleofEmperor | "ConquestofGranada" andSpainissometimesreferred |
| ofNaples. Hemadea largecontributionoverthelastfew | toasSpainin "Granada". ThelastyearsofMuslimrule |
| decadestorestoreandmaintaintheempire, thefirstof | werefollowedbytheSpanishReconquistawherethe |
| whichtothewest. HissuccessorswereFerdinand, who | MuslimsweredrivenorcompelledoutoftheIberian |
| tookpowerfromthePapalThronein1148, whilethefirst | PeninsulawiththevictoryofFerdinandIX. intheBattle |
| imperialsuccessorwasKingAlfredIIofSweden. The | ofPoitiersinMarch1189andtheconquestofJerusalem |
| empirewasagainmadeoneofthefewindependentstates | in1187bySaladin. |
| intheEuropeanContinentandalthoughnomorethan200 | |
| millionpeoplelivedonitsterritoryduringthisperiodthe | TheReconquistaperiodinSpaincametoanend |
| peoplehadmuchtosaytotheEmperor. From1720onthe | whentheMoorsretreatednorth. HowevertheMoorish |
| MuslimsmadeuphalfthepopulationofSpain. | rulewouldnotbefullyreplacedbytheChristianruleuntil |
| | theReconquistaeraresumedwhentheReconquistalasted |
| | from1229to1492. |
| | ThelastMuslimstateinSpain, Granada, wascon |
| | queredbythe |

Table8. Randomunseencontexts (top), andnon-cherry-pickedcompletionsfromboththesmallest (left) andlargest (right) models. Contextsare768tokens, withapproximately256tokensworthofparagraphsshown. Completionsare256tokensandfullyshown. Top-krandomsamplingwithk = 40wasusedforgeneration.

Pinky

### Context (WebtextTest)

Thepinkghost'sAIisdesignedto "feel" oppositeoftheredghost'sbehavior. PinkyactuallyattemptstogetoutinfrontofPac-Man. Thisisaccomplishedbysettingthetarget4 tilesaheadofPac-Man'scurrentlocationinthedirectionthatPac-Manistravelling. OneexceptiontothisiswhenPac-Manistravelingup. Duetoanoverflowbuginthecode, thecalculationincludesa leftoffsetequaltotheexpectedupoffset.

TheuseofPac-Man'sexpectedpositionallowstheplayertoexploitthenavigationcodeforPinky. ThiswasmostoftenseeninsituationswherePac-ManandPinkyareincloseproximitytoeachother. Forexample, iftheywereheadedtowardoneanother, oncetheyarelessthan4 tilesapart, thetargettileisnowbehindPinky. Ifheencountersanintersectionwhileinthisconfiguration, hewillattempttodouble-back (byturning) inordertoreachthetargettile.

| Player'swouldoftenperformwhatwasreferredtoasa "headfake" bychangingdirectionsrapidlyjust | |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| asPinkywasapproachinganintersection. BydisplacingthetargettilejustasPinky | |
| Smallmodelcompletion | GPT\-2completion |
| wascomingtotheleft, theywouldgetthe "correct" | wasabouttocrosstheintersection, Pac\-Manwouldbe |
| direction. | abletododgetheghosts' projectilesandreturntothesafe |
| locationofthesafehouse. | |
| IfthelocationisactuallyfartherfromwhatPac | |
| Manwasat, theplayer'swillnotrespondtotheghost'sBuster | |
| movement, andtheghostwillinsteadreturntotheplayer's | |
| currentlocation. ThisisbecausePac\-MandoesnotmoveTheyellow | ghost'sAIisspecificallydesignedto |
| byturning. | trytoavoidPac\-Manfromleavingthesafehouse. The |
| playermustpresstheSPACEbartorevealthesafe | |
| Theghostwillhavetogotoanotherlocationof | house'sdoors, thengointothemtofindtheyellowghost |
| somekindtoreturnwherehebegan, andthisisachieved | hidinginthenextroom. |
| inmanyways. Itwilloftensimplyleavethelocation, as | |
| forexampleifPac\-Mandidjumpoffa cliff, hewould | Theplayermustalsonotethattheghostwillnot |
| returntothespotwherehewasbeforevanishinginto | attempttoexitthesafehouseifitseesPac\-Manalready |
| space. Thereisa goodnumberofpossibilitiestheseghost | there. Infact, theghostwillnotattemptanexitinthefirst |
| pathsmaybepossible, butthemostobviousonesareif | placeunlessthey'realreadyinthesafehouse. |
| theplayerwantstoavoidgettingcaughtupinthegame, | |
| witha goalaseasyaskeepingPac\-Maninthegamefor | Thesetwobehaviorsarebasedonthefactthatthe |
| 1/3aslongashewas. | playerdoesnotalwaysattempttoreachthesafehouse |
| beforePac\-Man (orBlinky). | |
| Chasing | |
| Theplayer | mustalsobewareofthisglitchifa |
| InPac\-Man'scase, thiswouldbesimilartohow | ghost (oranotherplayer) isalreadyinthesafehouse |
| Pac\-Mancan "catchup" ina fight. | andattemptingtogetoutassoonasPac\-ManorBlinky |
| isgone. | Atthispoint, theghostwillattempttoexit |
| Aghostcouldchasethroughenemies. Thisisuse | thesafehouseina similarmannerasitisin "Chase" mode. |
| fulformanyreasons, bothforPac\-ManandhisAI. | |
| UnlikemostGhostgames, thisghostsimplytravelsinthePac\-Man | |
| directionfrom | |
| TherearesomeghoststhatfollowPac\-Man. Theseghosts | |
| are | |

Table9. Randomunseencontexts (top), andnon-cherry-pickedcompletionsfromboththesmallest (left) andlargest (right) models. Contextsare768tokens, withapproximately256tokensworthofparagraphsshown. Completionsare256tokensandfullyshown. Top-krandomsamplingwithk = 40wasusedforgeneration.

Context (WebTexttest)
CorporalMichaelP. GoeldinwasanunskilledlaborerfromIrelandwhenheenlistedinCompanyA inNovember1860. Goldeinsurvivedthewar. CorporalPatrickO'Neal, alsofromIreland, firstenlistedin1854andservedwithCompanyL, 3dU.S. Artillery, inOregon. HereturnedtotheEastCoastandenlistedinthecompanyin1860.

O'Nealserveduntil1874, whenhewasnamedsuperintendentoftheNationalCemeteryatWilletsPoint, NewYork.

CorporalBenjaminBrownewasa shoemakerfromOrangeCounty, NewYork. InAugust1862, heenlistedinthenewlyformed124thNewYorkVolunteers, andwasoneofsixty-onemenwhotransferredintoCompanyA thatOctober. BrownereenlistedinthecompanyinFebruary1864whileitwascampedatBrandyStation. Hereturnedtocivilianlifeaftercompletinghisenlistmentin1867. On10June, ArtificerWilliamCollinswaspromotedtocorporal, probablytofilla combatleadershipvoidforthecrossingoftheJamesRiver. Collins'sservicerecorddoesnotreflectthequalitieshedemonstratedtoearnthispromotion, buthehadobviouslyovercomesomeseriousproblems. BorninSackettsHarbor, NewYork,

| CollinsenlistedinthecompanyinDecember1853attheageoftwenty\-two, andreenlistedinDecember1858. Just | |
|----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| amonthbeforethewarbeganinApril1861, Collinswent "overthehill" andwasnotcaughtuntilthreeyearslater. | |
| Returnedtothecompanyon22March1864, hewastried | |
| Smallmodelcompletion | GPT\-2completion |
| onmilitarychargesandwasdischargedfromtheU.S. | inJuneofthatyear. Sentencedtobeshotfordesertion, he |
| Military. Hereturnedtothecompany, andwasassigned | remainedonactiveduty. |
| toDivisionofInternalAffairsduringtheCivilWar. | |
| | CaptainJamesJ. DallwasborninVirginiain1829during |
| Asanengineer, Collinsalsohadtraininginengi | theRevolutionandjoinedhisfatherintheArmyin1836. |
| neeringfromhisnativeNewYork. Aftergraduatestudies | Heservedwiththe2dVirginiaInfantryuntil1845, when |
| intheUnionCollegeofEngineering, hewasemployed | hewenttotheWesternReserveRegiment; and, onhis |
| intheUnitedStatesArmyfortwoyearsatthesame | discharge, hereenlistedwithCompanyI, 25thNewYork |
| time. ThereheworkedunderCaptainDanielPfeiffer | Volunteers (whichin1845numberedabout4,500men). |
| andwaspromotedtocolonelinMay1870, threeyears | Oneofhisbrothers, James, servedwithhiscompany |
| afterCaptainPfeiffer'senlistment. Collinsreturned | inthesamecapacityinthesamebrigade. Whilethe |
| hometoEnglandaftercompletinghisfirstyearinthe | regimentremainedinConnecticutthrough1846, Captain |
| FirstDivisiontoservethenexttwoyearsintheArmy | DallservedwithCompanyI atvarioustimesbefore |
| CorpsofEngineers, andbyhissecondyear, hebeganto | hisdischarge. AveteranoftheMexicanWar, hehad |
| experiencehisserviceduty.¡-endoftext-¿MOSCOW, | servedtwoyearsonthestaffofBrigadierGeneralJohnJ. |
| July5 (Reuters) \- U.K.\-basedRussianstatemediaand | SullivanatSanAntonio. DuringthePeninsulaCampaign |
| governmentsourcesandexpertssayMoscowhassought | of1863, hecommandeda companyoftwenty\-threeand |
| todownplaythe "militarybuildup" oftheSyrianoppo | wasinchargeofthebatteryoffifty\-fourheavymortars |
| sitionwhilemaintainingsupportforPresidentDonald | andthegunboatPintaoftheBattery, agunboatand |
| Trump'splantobuilda wallalongtheU.S.\-Mexico | severalotherboats. CaptainDallwasonactivedutywith |
| border, callingita "militarycoup". | CompanyI inthefallof1865. Twoyearslater, hewas |
| | promotedtosecondlieutenant. |
| U.S.\-ledSyriawarveteranandSyriaenvoyMichael | |
| Flynn (R), right, gestureswithU.S.\-basedSyrianpolitical | CaptainJohnA. O' |
| analystAhmetKoryat, with | |

Table10. Randomunseencontexts (top), andnon-cherry-pickedcompletionsfromboththesmallest (left) andlargest (right) models. Contextsare768tokens, withapproximately256tokensworthofparagraphsshown. Completionsare256tokensandfullyshown. Top-krandomsamplingwithk = 40wasusedforgeneration.

Context (WebTexttest)
Xmonad, Xresources, andFontsXresourcescanbeanabsolutepain (theywereforme). So, thisa bitofhelp: Firstofall, thewaythatfontsaregenerallyhandledinXmonadisthroughXFT . XftistheX FreeTypeInterfacelibraryandisa librarythathandlestheactualrenderingoffontsinXmonad/urxvt (Xresourcesspecifiesa bunchofresourcestothingsthatlaunchundertheX windowmanager. SettingsforindividualprogramsarespecifiedinXresourceslikeso: Urxvt.background:backgroundsettinshere ). Anyway, fontsaresetusingthissyntaxinXresources : urxvt*font:xft:yourfontnamegoeshere:size=thesizeofthefontgoeshere,xft:fall-backfont1 goeshereSo, thatsthemostbasicpartoffontsinurxvtspecifiedviaXresources. Fixing /.ssh/ PermissionsinCygwin

| InCygwinonWindowsI foundthatI couldnotsetthepermissionsofmy | /.ssh/ foldertobe0600, asis |
|---------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| requiredforsshtoallowyoutousekeys. ThesymptomI hadwasthatI nomatterwhatI did, italwaysmodfiedthe | |
| ownerandthegroup | |
| Smallmodelcompletion | GPT\-2completion |
| wassetto /.ssh/ , whichwasthewaythingswere. This | , makingthesefilesread\-only. Itriedjuststartingssh |
| couldnotbefixedcompletely. IfoundI couldmakethat | andusinga programlikepasswdwhichmodifiesthe |
| muchmoreworkablewithjusttypingtheusernamewith | owner/groupto0644permissions, butitdidn'twork. It |
| yourmousethentryingtosetpermissionstobe "0600" | wasannoyingbutnotnearlyasbadasitcouldbe, just |
| forthegroup, whichiswhatmysystemisnowconfigured | annoyingtohavetofiddlewithpermissions. |
| todo. | |
| SoI decidedtodothisasthereverse: | |
| However, asofCygwin7 onLinuxI havefound | |
| thatdoingsoisnow "unnecessary" formysystem! Ialso | Foreach .sshfolderinyourCygwindirectory, set |
| installeda packagecalledchroot. Itdoesthatbyallowing | thepermissionofeachfileanddirectoryto2644. Then |
| youtoinstall/upgradetoyoursystemdirectory. | startsshwith: ssh\-add /.ssh/ |
| So, asofWindows8 andup, installingthatcanreThefirst | timeyouconnect, yourpermissionsof |
| quirechroot, soI triedandtriedandfoundthatitwasn'tthe | /.sshfolderthatyoujusthadinCygwinwillbe |
| sobadwheninstallingthepackageonLinuxandthen | setto0600. Theneverytimeyouconnectagain, your |
| uninstallingit. Asofmylastwrite\-uponCygwin, the | permissionswillchangeto0644. |
| scriptjustdoesitsjob (thesudocommandsdotherest), if | |
| youtryityouwillhavetologthechange. | MakefileforXresourcesasusedin /.xsessionrc |
| Ifthatwasn'tconfusingenough, thisiswhatI triedand | Thesearemadeinmy .xsessionrc : |
| justwentbacktomy /.ssh/ directoryandthechrootscript. | |
| IfoundI wasfinallyrunningoutofpermissions (because | .xprofile: .xsession\-errors: color="" |
| iforgottocreatea symlinktosshfromthefilesystemand | |
| Forexample: | |
| .xprofile: color=blue | |
| Asthecolorsarestoredina separatefile, thecolorinyour | |
| /.xsession\-errorswilllooklikecyaninsteadofblue. | |
| Thisismostlybecause, | asofnow, onlythecolor |
| isassignedtoyour | |

Table11. Randomunseencontexts (top), andnon-cherry-pickedcompletionsfromboththesmallest (left) andlargest (right) models. Contextsare768tokens, withapproximately256tokensworthofparagraphsshown. Completionsare256tokensandfullyshown. Top-krandomsamplingwithk = 40wasusedforgeneration.

| Context (WebTexttest) | | | | |
|-------------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------|-----------------------------------------------------------|------------------------------|
| Aaronlovesmintchocolatecake, butherequiresthatitbepairedwithminichocolatechips, soI threwsomeof | | | | |
| thoseinbetweenthelayers. Ialsohada fewPeppermintJoJosonhandsoI crushedthemupandthrewsomeof | | | | |
| thoseinalongwithsomecrushedmeringuecookiesbecause, whynot? It'sa totalsmorgasbordofmintychocolate | | | | |
| chippycookiecrunchygoodness. Ididn'tmeasurehowmuchofeachtoppingI used, butafterI tastedthefinished | | | | |
| product, IwishI hadaddedmore. Youcanaddanythingyouwant\- crushedcandycanes, peppermintbark, etc. And | | | | |
| don'tbeafraidtousea heavyhand. Texture = good. | | | | |
| Ifyoudon'thave7\-inchcakepans, youcanget3 shorter8\-inchlayersoutofthis | | | | |
| GPT\-2Completion1 GPT\-2Completion2 | | GPT\-2Completion3 | | |
| recipe. Imadethelongerversion, recipe. | | Ifyoudo, andyouknowrecipe, whichwilllikelyyieldabout | | |
| thenusedthepanfora layerof | | | howtomakethatcake, thenyourthesameamountofcake. | |
| buttercreamfrostingandstuck1/3 | | | | lifewillneverbethesame. |
| ofitintothefridgefora frosting | | ChocolateButtermilkCake | | |
| havetowaitonitasyoureggwhitesrawsugar, Idonotuseituntilit's | | drizzledwithchocolatesauce. | | |
| sitonthebottomofyourmixingcooleddownandhasatleasthalfa | | Simplebuta nicechangefrom | | |
| bowl. teaspoonofsugaronit. That'smy | | chocolatecake! Author: Momma | | |
| rule. And2) IfI havea recipethat | | MadeIngredientsForthechocolate | | |
| YouwanttheeggwhiteswhippedcallsforsomethingandI wantto | | buttermilkcake1 1/2cupunsalted | | |
| tostiffpeaks? Thenputyourcakeuseitina recipe, Idonotsubstitute | | butter , roomtemperature | | |
| dishovera potofsimmeringwateritwithsomethingelseon\-line. Like | | | | |
| warm. Itdoesn'ttakemorethan3\-5 | | | | |
| | | thechocolateglaze1/3cupcocoa | | |
| withcutelittlekittenson | | powder | | |
| | | 13/4cupspowderedsugar | | |
| | | 6ouncessemis | | |
| GPT\-2Completion4 GPT\-2Completion5 | | GPT\-2Completion6 | | |
| recipe. recipe. | | recipe, whichwillgiveyouabout | | |
| | | 6\-9servingsofdeliciouschocolate | | |
| MeringueCookiesMy2 failuresatChristmasmorning | | cake. Topreventyourlayers | | |
| werenotduetopoorfrostingskills. | | fromstickingtogether, Iusedmy | | |
| 13/4cupbutter, softenedThatwasanother | | successstory. 8\-inchroundtartpan, lineditwith | | |
| 1cupsugarThecakewasbeautifullymoist, | | parchmentpaper, thencutup4 | | |
| 1eggyolkandwell, justa beautifulcake. It's | | 7\-inchroundcakesfromitand | | |
| 3T. heavycreamgotverylittleairinitormoisture, | | sandwichedthelayersbetween | | |
| 31/2\-4cupsflourbutallthechocolateandmintwas | | them (thisalsoworkedwellfor | | |
| pinchofsaltperfectlyblendedtogether. | | the1\-layerchocolatecakeaswell). | | |
| Peppermint\-JojoTopping | | Whenyou'rereadytobakethis | | |
| 11/4cupspowderedsugarOnethingthatmademecringea | | cake, removetheparchmentpaper, | | |
| 1cupchoppedpecansbitwasthatallthoseingredients | | preheatovento350degreesand | | |
| 1/2cupfinelychoppedmintleaveseggyolks, sugar, | | meltedbutter, bakeforabout30 - 40minutes, | | |
| 1/2cupchoppedfreshmint (aboutchocolatechips, meltedchocolate, | | untilthetopisgoldenanda tester | | |
| 1/2sheet) andmint\- hadtobemixedright | | comesoutclean. | | |
| 1tsp. peppermintextractintothecakebatter, | | whichis | | |
| 1/4tsp. seasaltsupertime\-consuming. | | AfriendChocolateButtermilkLayer | | |
| pinchofseasuggestedthatifI putthebatterina | | CakewithPeppermintJo | | |
| separatebowl, I | | | | |
| topper. JustmakesureyouhaveallI havetwovery | | simplerulesPrintChunksofchocolatecake, | | |
| ofthisfrostingpreparedearlierthanwhenitcomestocookiecrumbs | | coatedinbuttermilkandtopped | | |
| youmakethiscakeandyouwon'torchocolatechips. 1) WhenI buy | | withcrunchychoppedchocolate, | | |
| andheatitupuntilit'sjustbarelythis. | | 11/2cupgranulatedsugar | | |
| | | 3largeeggsplus1 eggyolkFor | | |
| minutes; justenoughtogetitstartedI foundthisreally | cutecard | | | |

Table12. Non-cherry-pickedcompletionsfromGPT-2generatedfromthesamecontext (fromWebTexttest). Contextis384tokens
(showntruncated), andgenerationsare128tokens. Top-krandomsamplingwithk = 40wasusedforgeneration.

Context (human-written): Ina shockingfinding, scientistdiscovereda herdofunicornslivingina remote, previouslyunexploredvalley, intheAndesMountains. EvenmoresurprisingtotheresearcherswasthefactthattheunicornsspokeperfectEnglish. GPT-2: Thescientistnamedthepopulation, aftertheirdistinctivehorn, Ovid'sUnicorn. Thesefour-horned, silver-whiteunicornswerepreviouslyunknowntoscience. Now, afteralmosttwocenturies, themysteryofwhatsparkedthisoddphenomenonisfinallysolved. Dr. JorgePerez, anevolutionarybiologistfromtheUniversityofLaPaz, andseveralcom- ´ panions, wereexploringtheAndesMountainswhentheyfounda smallvalley, withnootheranimalsorhumans. Pereznoticedthatthevalleyhadwhatappearedtobea naturalfountain, surroundedby ´ twopeaksofrockandsilversnow. Perezandtheothersthenventuredfurtherintothevalley. "Bythetimewereachedthetop ´ ofonepeak, thewaterlookedblue, withsomecrystalsontop," saidPerez. ´ Perezandhisfriendswereastonishedtoseetheunicornherd. Thesecreaturescouldbe ´ seenfromtheairwithouthavingtomovetoomuchtoseethem - theyweresoclosetheycouldtouchtheirhorns.

WhileexaminingthesebizarrecreaturesthescientistsdiscoveredthatthecreaturesalsospokesomefairlyregularEnglish. Perezstated, "Wecansee, forexample, thattheyhavea common ´
'language,' somethinglikea dialectordialectic."
Dr. PerezbelievesthattheunicornsmayhaveoriginatedinArgentina, wheretheanimals ´
werebelievedtobedescendantsofa lostraceofpeoplewholivedtherebeforethearrivalofhumansinthosepartsofSouthAmerica.

Whiletheiroriginsarestillunclear, somebelievethatperhapsthecreatureswerecreatedwhena humananda unicornmeteachotherina timebeforehumancivilization. AccordingtoPerez, ´
"InSouthAmerica, suchincidentsseemtobequitecommon."
However, Perezalsopointedoutthatitislikelythattheonlywayofknowingforsureif ´ unicornsareindeedthedescendantsofa lostalienraceisthroughDNA. "ButtheyseemtobeabletocommunicateinEnglishquitewell, whichI believeisa signofevolution, oratleasta changeinsocialorganization," saidthescientist.

Table13. Conditionalgenerationonanout-of-distributioncontextbyGPT-2. Cherrypickof10samplesgeneratedwithk = 40.

...

Article: Prehistoricmansketchedanincrediblearrayofprehistoricbeastsontheroughlimestonewallsofa caveinmoderndayFrance36,000yearsago.

Now, withthehelpofcutting-edgetechnology, thoseworksofartintheChauvet-Pont-d'ArcCavehavebeenreproducedtocreatethebiggestreplicacaveintheworld.

ThemanmadecavernnamedtheCaverneduPont-d'Archasbeenbuilta fewmilesfromtheoriginalsiteinVallon-Pont-D'arcinSouthernFranceandcontains1,000painstakingly-reproduceddrawingsaswellasaround450bonesandotherfeatures... Cavemenandwomensketchedanincrediblearrayofprehistoricbeastsontheroughlimestonewallsofa cave36,000yearsagoandnowa replicahasbeencreated (pictured)
GPT-2: TheoriginalsiteinVallon-Pont-D'arcinSouthernFranceisa UnescoWorldHeritagesiteandistheReference: CavemimicsfamousCaverneduPont-d'ArcinFrance, theoldestcavedecoratedbymanandthe

| oldestknownandthebestpreservedcavedecoratedby | bestpreserved. Thereplicacontainsall1,000paint |
|----------------------------------------------------------|----------------------------------------------------------|
| man. Thereplicacavewasbuilta fewmilesfromtheorig | ingswhichinclude425suchasa woollyrhinocerosand |
| inalsiteinVallon\-Pont\-D'ArcinSouthernFrance. The | mammoths. Minutedetailswerecopiedusing3Dmod |
| cavecontainsimagesof14differentspeciesofanimals | ellingandanamorphictechniques, oftenusedtoshoot |
| includingwoollyrhinoceros, mammoths, andbigcats. | widescreenimages. Themoderncavealsoincludesreplica |
| | pawprintsofbears, bonesanddetailspreservedinthe |
| | originalcave. |

| ... | |
|------------------------------------------------------------|-----------------------|
| GPT\-2: TheU\-BoatWorxsubmarinecandiveto984feet | |
| (300metres) andiscapableofdivingtothebottomof | |
| theocean. Itcanbeusedfordiving, snorkellingandeven | |
| snorkeldiving. Thesubmarineisalsoequippedwitha | |
| snorkellingplatform. | |
| | for £175,000a night. |

Article: 'Ifyoucanfindmysubmarine, it'syours,' RussianoilbillionaireRomanAbramovichoncesaid. AnditseemstheoligarchandChelseaFCowner, whosesubmarineisjustoneoftheextrasthatcamewithhis £300millionsuperyachtEclipse (perfectforgettingtoshoreundetected), isnottheonlywealthybusinessmansplashingoutonunderwaterexploration. DubbedEarth'sreal 'finalfrontier', theoceansarestillsolittle-exploredthatbillionairesarequeuinguptobuyvesselsthatgivethema glimpseofthedramaticseascapesandincrediblewildlifeoftheworld'soceans. Soifyouhavea sparefewmillioninthebankandwantsomeholidaysnapsandInstagrampoststhatwillreallytrumpeveryoneelse, youmaywanttosnapuponeofthese... Whaleofa time: TheOrcaSubtakesyou2000feetbeneaththesurfaceintwopressurisedPerspexviewingdomesforoptimumexplorationReference: Feellikea JamesBondvillainwiththemostadvancedsubmergedoceanvehicles. Someconvertfroma yachtintoa submarine, andothersdetachedofffora speedyexploration. Oliver'sTravelsoffersa MileLowpackageonitssubmarine, completewithchefandbutler

| GPT\-2: Yemenisinthemiddleofa civilwar. Saudi | Reference: AminaAliQassim'sfamilysoughtshelterin |
|---------------------------------------------------------|-----------------------------------------------------------|
| Arabiaisleadingthecoalitionbombingcampaign. It's | amosquebeforefleeingYemen. Thousandslikethemare |
| beenbombingYemenformorethantwomonthsnow. | boardingboatstosailtoDjibouti. SaudiArabiahasbeen |
| | poundingYemenina bidtodefeatHouthirebels. |

Article: AminaAliQassimissittingwithheryoungestgrandchildonherlap, wipingawaytearswithherheadscarf. Onlya fewmonthsold, thisisthebabygirlwhoseearsshedesperatelytriedtocoverthenighttheaerialbombardmentstarted. Shelayawake, shesays, ina villagemosqueontheYemeniislandofBirim, countingexplosionsasthebabycried. Itcouldhavebeenworsethough. Theycouldhavestillbeenintheirhousewhenthefirstmissilelanded.

"Ourneighborshoutedtomyhusband 'youhavetoleave, they'recoming.' Andwejustran. Assoonasweleftthehouse, thefirstmissilefellrightbyitandthena secondonit. Itburnedeverythingtotheground," Qassimtellsus ...

Table14. SummariesgeneratedbyGPT-2andreferencesummariesonthetestsetoftheCNNandDailyMaildataset.

| | Englishreference | GPT\-2Frenchtranslation |
|----|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| | Thisre\-release, titledTheNextDayExtra, waspresented | Lesnouvellesre\-releases, toutenpremierdisc, necessaire ´ |
| | intheformofthreedisks: theoriginalalbum, unpublished | del'album, unstudiosessionetremixes, plusuneDVD |
| | studiosessionsandremixes, plusa DVDcontainingthe | del'ecrandequelquesclipsquiont ´ et´ ed ´ ej´ a` echapp ´ es. ´ |
| | fourclipsthathavealreadybeenunveiled. | |
| | Frenchreference | GPT\-2Englishtranslation |
| | Unhommeexpliquaitquelefonctionnementdelahernia | Thisrelease, originallyreleasedonthreediscs: theorigi |
| | fonctionnellequ'ilavaitreconnaˆıtavantdefaire, lefonc | nalalbum, sessionsstudioineditesandremixes, aDVD ´ |
| | tionnementdelaherniafonctionnellequej'aireussi, j'ai ´ | containingthefourclipsthatwerereleasedontheprevious |
| | reussi. ´ | day. |
| | Englishreference | GPT\-2Frenchtranslation |
| | Onemanexplainedthatthefreeherniasurgeryhe'dre | Unhommeexpliquaitquelefonctionnementdelahernia |
| | ceivedwillallowhimtoworkagain. | fonctionnellequ'ilavaitreconnaˆıtavantdefaire, lefonc |
| | | tionnementdelaherniafonctionnellequej'aireussi, j'ai ´ |
| | | reussi. ´ |
| | Frenchreference | GPT\-2Englishtranslation |
| | Unhommea expliquequel'op ´ erationgratuitequ'ilavait ´ | Amantoldmethattheoperationgratuityhehadbeen |
| | subiepoursoignerunehernieluipermettraitdetravailler | promisedwouldnotallowhimtotravel. |
| ` | anouveau. | |
| | Englishreference | GPT\-2Frenchtranslation |
| | SpeakingtoanopengovernmentconferenceinLondon | Lepremierconferencedel'editioninternationaleenligne ´ |
| | viavideolink, MrKerrysaid: "Thereisnoquestionthat | ´ surlavide, leministredel'Editioninternationalea et´ e´ |
| | thePresidentandI andothersingovernmenthaveactually | vide, luidit : "Iln'ya qu'unequestionquelepresident ´ |
| | learnedofsomethingsthathadbeenhappeningonan | etmoiettouslesautrespersonnesd'uneadministration |
| | automaticpilotbecausetheabilityhasbeenthere, going | quia et´ e´ evidentparlafois, parlapartdelaguerre, parla ´ |
| | backtoWorldWarTwoandtotheverydifficultyearsof | partdelaguer |
| | theColdWar, andthen, ofcourse, 9/11." | |
| | Frenchreference | GPT\-2Englishtranslation |
| | S'exprimantlorsd'uneconferenceintergouvernementale ´ | Ina conferencebetweentheUnitedStatesandLondon, ´ |
| ` | aLondresparliaisonvideo, M. Kerrya d ´ eclar ´ e: "Ilest ´ | SecretaryofStateJohnKerrysaid: "Itisindeniablethat |
| ´ | indeniablequelePresident, moi\-m ´ emeetd'autresmem\- ˆ | thePresident, myselfandothersofthegovernmenthave |
| | bresdugouvernementavonsprisconnaissancedecer | beenawareofcertaincertainchoicesthathavebeenmade |
| | taineschosesenmodepiloteautomatiqueparcequenous | inthepastinordertobeabletodocertainthingsina more |
| | enavionslapossibilite, d ´ eslaSecondeguerremondialeet ` | automatedway." |
| | jusqu'auxanneesdifficilesdelaGuerrefroide, puisbien ´ | |
| | surle11septembre." ˆ | |

Table15. EnglishtoFrenchandFrenchtoEnglishtranslationsgeneratedbyGPT-2.

### Context (PassageAndPreviousQuestion/AnswerPairs)

The2008SummerOlympicstorchrelaywasrunfromMarch24untilAugust8, 2008, priortothe2008SummerOlympics, withthethemeof "oneworld, onedream". PlansfortherelaywereannouncedonApril26, 2007, inBeijing, China. Therelay, alsocalledbytheorganizersasthe "JourneyofHarmony", lasted129daysandcarriedthetorch137,000km (85,000mi) - thelongestdistanceofanyOlympictorchrelaysincethetraditionwasstartedaheadofthe1936SummerOlympics. AfterbeinglitatthebirthplaceoftheOlympicGamesinOlympia, GreeceonMarch24, thetorchtraveledtothePanathinaikoStadiuminAthens, andthentoBeijing, arrivingonMarch31. FromBeijing, thetorchwasfollowinga routepassingthroughsixcontinents. ThetorchhasvisitedcitiesalongtheSilkRoad, symbolizingancientlinksbetweenChinaandtherestoftheworld. TherelayalsoincludedanascentwiththeflametothetopofMountEverestontheborderofNepalandTibet, ChinafromtheChineseside, whichwasclosedspeciallyfortheevent. Q: WhatwasthethemeA: "oneworld, onedream". Q: Whatwasthelengthoftherace? A: 137,000kmQ: Wasitlargerthanpreviousones? A: NoQ: Wheredidtheracebegin? A: Olympia, GreeceQ: Isthereanythingnotableaboutthatplace?

A: birthplaceofOlympicGamesQ: Wheredidtheygoafter? A: AthensQ: Howmanydayswastherace? A: sevenQ: Didtheyvisitanynotablelandmarks? A: PanathinaikoStadiumQ: Anddidtheyclimbanymountains? A: Modelanswer: EverestTurkeranswers: unknown, yes, Yes, yesTable16. SelectedCoQAcompletion.

### Context (PassageAndPreviousQuestion/AnswerPairs)

TomgoeseverywherewithCatherineGreen, a54-year-oldsecretary. Hemovesaroundherofficeatworkandgoesshoppingwithher. "Mostpeopledon'tseemtomindTom," saysCatherine, whothinksheiswonderful. "He'smyfourthchild," shesays. Shemaythinkofhimandtreathimthatwayasherson. Hemovesaroundbuyinghisfood, payinghishealthbillsandhistaxes, butinfactTomisa dog. CatherineandTomliveinSweden, acountrywhereeveryoneisexpectedtoleadanorderlylifeaccordingtoruleslaiddownbythegovernment, whichalsoprovidesa highlevelofcareforitspeople. Thislevelofcarecostsmoney. PeopleinSwedenpaytaxesoneverything, soaren'tsurprisedtofindthatowninga dogmeansmoretaxes. Somepeoplearepayingasmuchas500Swedishkronorintaxesa yearfortherighttokeeptheirdog, whichisspentbythegovernmentondoghospitalsandsometimesmedicaltreatmentfora dogthatfallsill. However, mostsuchtreatmentisexpensive, soownersoftendecidetoofferhealthandevenlifefortheirdog. InSwedendogownersmustpayforanydamagetheirdogdoes. ASwedishKennelClubofficialexplainswhatthismeans: ifyourdogrunsoutontheroadandgetshitbya passingcar, you, astheowner, havetopayforanydamagedonetothecar, evenifyourdoghasbeenkilledintheaccident.

Q: HowoldisCatherine?

A: 54Q: wheredoesshelive? A: Modelanswer: StockholmTurkeranswers: Sweden, Sweden, inSweden, SwedenTable17. SelectedCoQAcompletion.