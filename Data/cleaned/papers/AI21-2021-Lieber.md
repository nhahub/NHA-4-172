# Jurassic-1: TechnicalDetailsAndEvaluationWHITEPAPER

| OpherLieber | OrSharir | BarakLenz | YoavShoham |
|-----------------|--------------|-----------------|----------------|
| AI21Labs | AI21Labs | AI21Labs | AI21Labs |
| opherl@ai21.com | ors@ai21.com | barakl@ai21.com | yoavs@ai21.com |

# A**Bstract**

Jurassic-1isa pairofauto-regressivelanguagemodelsrecentlyreleasedbyAI21Labs, consistingofJ1-Jumbo, a178B-parametermodel, andJ1-Large, a7B-parametermodel. Wedescribetheirarchitectureandtraining, andevaluatetheirperformancerelativetoGPT-3. Theevaluationisintermsofperplexity, aswellaszero-shotandfew-shotlearning. Tothatend, wedevelopeda zeroshotandfew-shottestsuite, whichwemadepubliclyavailable (https://github.com/ai21labs/
lm-evaluation) asa sharedresourcefortheevaluationofmegalanguagemodels.

# 1IntroductionAI21LabshasrecentlyreleasedJurassic-1, thefirstina sequenceoflanguagemodels (LMs) thatwillbemadeavailabletotheresearchanddevelopmentcommunity. TohelpdevelopersassessthesuitabilityofJurassic-1fortheirneeds, thiswhitepaperprovidestechnicaldetailsregardingitsarchitectureandtraining, aswellasanevaluationofitsperformance. WebelievethatradicaladvancesinNLPcallforinnovationsbeyondincreasingthenetworksize, trainingdata, andtrainingtime. Butsizedoesmatter. Jurassic-1isa setofbaselinemodels, inspiredbyOpenAI'spioneeringworkonGPT-3 (Brownetal., 2020). SimilartoGPT-3, Jurassic-1consistsofauto-regressivemodelstrainedona mixofEnglishcorporathatscalesupto178Bparameters. Itdiverges, however, fromGPT-3inseveralimportantrespects, suchasthesizeofvocabularyandthedepth/widthratiooftheneuralnet. Wecoverthesearchitecturaldecisionsinthenextsection.

Inthatsectionwealsobrieflydiscussthetrainingprocess, coveringbasictopicssuchastrainingcorpusandlengthoftraining.1Finally, wepresentourevaluationofthemodel, includinga comparisonwithGPT-3. Evaluationofperplexityisstraightforward. Evaluationofzero-shotlearningrequiresspecification, andevaluationoffew-shotlearningisnotoriouslytricky, beingsubjecttothevagariesofpromptchoice. Wedevelopeda zero-shotandfew-shotevaluationsuitewhichwefounduseful, andhavepostedthesuitetoGitHub (https://github.com/ai21labs/lm-evaluation) asa sharedresourceforthecommunityfortheevaluationofmegaLMs.

# 2ModelAndTrainingDetailsInthissectionwedescribethedesignchoicesbehindourJurassic-1models. Insub-section2.1wedescribethearchitecturethatJurassic-1isbasedupon, howitdiffersfrompriorapproaches, andhowthattranslatesintofasterinference. Insub-section2.2wedescribethevocabularywedevelopedforincreasingthetokenizationefficiency, therebyreducingthecomputeneededtoprocessa givenstringoftext, aswellasprovidinga certainamountofsemanticinductivebias. Last, insub-section2.3webrieflydescribehowthemodelwastrained.

1Trainingsucha largemodel, onover800GPUsovermanymonths, isa non-trivialengineeringfeat, andraisesmanyissuesnotpresentinsmallermodels: Overflows, nullattentionheads, modelanddataparallelismthatrequiresolutionsontopofpackagessuchasDeepSpeed (Rasleyetal., 2020), hardwarefailures, rigorouscheckpointing, andmore. Theseadditionaldetailsarebeyondthescopeofthispaper.

| Model | nparams | nlayers | dmodel | nheads | dhead | nvocab |
|-------------|-----------|-----------|----------|----------|---------|----------|
| GPT\-36.7B | 6.7B | 32 | 4096 | 32 | 128 | 50K |
| J1\-Large | 7.5B | 32 | 4096 | 32 | 128 | 256K |
| GPT\-3175B | 175B | 96 | 12288 | 96 | 128 | 50K |
| J1\-Jumbo | 178B | 76 | 13824 | 96 | 144 | 256K |

| Tokenizer | nvocab | Wikipedia | OWT | Books | C4 | PileCC | Avg. | arXiv | GitHub | Avg. |
|------------------|----------|-------------|-------|---------|-------|----------|--------|---------|----------|--------|
| T5'sSP | 32K | 0.255 | 0.253 | 0.291 | 0.245 | 0.234 | 0.256 | 0.375 | 0.370 | 0.372 |
| GPT2/3'sBPE | 50K | 0.223 | 0.217 | 0.253 | 0.216 | 0.225 | 0.227 | 0.333 | 0.405 | 0.369 |
| Jurassic\-1'sSP | 256K | 0.171 | 0.158 | 0.172 | 0.154 | 0.159 | 0.163 | 0.257 | 0.322 | 0.290 |

Table1: ComparingthearchitectureofourJurassic-1modelstotheirGPT-3counterparts.

Table2: Comparingtheefficiencyofdifferenttokenizersonvariouscorpora, asmeasuredbytheaveragetokens-perbytes (TPB) ratio, i.e., numberoftokensdividedbynumberofbytesina samplefromthecorpus.

### 2.1ArchitectureWebasedourmodelsonthedecodermoduleoftheTransformerarchitecture (Vaswanietal., 2017) withthemodificationsproposedbyRadfordetal. (2019). Inputtokensarefirstconvertedtovectorrepresentationwithannvocab-by-dmodelembeddingmatrix (seefollowingsectionfora furtherdiscussion), andthenfedintotheTransformernetwork. ThearchitectureiscomposedofnlayersTransformerlayersusinga hiddendimensiondmodel, eachequippedwitha selfattentionmodulewithnheadsattentionheadsofsizedheadanda feed-forwardmodule. Seepriorworksforspecificdetailsoninner-modules.

Wetargetedtwomodelsizesthatwecallinshort **J1-Large** (7.5Bparameters) and **J1-Jumbo** (178Bparameters), whichroughlycorrespondtoGPT-36.7BandGPT-3175Bmodels (asspecifiedinBrownetal. (2020)), respectively. SeeTable1 forexactspecificationsofbothourJurassic-1modelsandtheirGPT-3counterparts. ForJ1-Jumbo, wedivergedfromtheTransformerarchitectureusedbyGPT-3175B. Instead, wedesignedourarchitectureinlightofa recentlyproposedtheory (Levineetal., 2020) forthedepth-to-widthexpressivitytradeofffoundinselfattentionnetworks. Accordingtosaidtheory, toleveragethefullextentofthepowerbroughtbydepth, thewidthmustbechosenappropriately - putanotherway, fora givenparameterbudgetthereisanoptimaldepth. Specifically, fora parameterbudgetof175B (notincludingembeddingmatrix), theoptimaldepthshouldbearound80layers, farfromthe96layersusedbyGPT-3175B. WeputthetheorytotestanddesignedourJ1-Jumbowith76layers. Weused76ratherthan80layersbecausewealsohavetoaccountforvarioushardwareconsiderationsduringbothtrainingandinference. Forexample, thehiddendimensionisconstrainedbythenumberofheadsandtheirsize, whiledheadshouldbea multipleof8 foroptimalmatrix-multiplicationoperationsonGPUsandnheadsshouldbedivisiblebythemodelparallelizationfactor. Similarly, thelayersaredividedintostagesofequalcomputeneedsforpipelineparallelization, introducinga constraintonnlayersaswell. Beyondthepotentialadvantagessuggestedbythetheory, anotherbyproductofourrefinedarchitectureisa significantgaininruntimeperformance. Byshiftingcomputeresourcesfromdepthtowidth, moreoperationscanbeperformedinparallel (width) ratherthansequentially (depth). Thisisespeciallyrelevanttotextgenerationwheretokensareprocessedoneata time, andsothereislessopportunityforparallelization, resultinginsub-optimalGPUutilization. Inourbenchmarks, comparingourarchitectureagainstGPT-3175Bonthesamehardwareconfiguration, ourarchitecturehasmodestbenefitsintrainingtime (1.5% speedupperiteration), butsignificantruntimegainsinbatchinference (7%)
andtextgeneration (upto23%).

### 2.2LargeVocabularyForTokenizationEfficiencyTheruntimeforprocessinga rawstringoftextwitha Transformerarchitectureisdirectlyrelatedtothenumberoftokens, denotedbyN, thestringisencodedinto, roughlyO(N) denselinearoperationsperTransformerlayerandO(N2) fortheself-attentionoperation. Giventhat, weinvestigatestrategiestoimprovethetokens-per-wordratio, therebyreducingthecostforprocessingthetext.

| Word\-Pieces | Words | Multi\-WordExpressions |
|----------------|----------------------|---------------------------|
| z | _cat | _ever_so_slightly |
| _pre | _descriptive | _Tony_Stark |
| ism | _System.out.println | _marketing_campaign |
| tion | _Pokemon | _Higgs_boson |
| \-ness | _e.g. | _production_capacity |
| \-in\-law | | _stepping_down |
| \-on\-demand | _https://github.com/ | _bank_accounts |
| \-of\-the\- | _LGBT | _National_Public_Radio |

Table3: ExamplesofitemsfromJ1'svocabulary, includingword-pieces, wholewords, andmulti-wordexpressions, whereweusethemoreloosesenseofa wordasa sequencesnotcontainingwhitespaceinthemiddleofit. Underscorerepresenta literalspaceinthetoken.

Figure1: Anexampleshowinghowmulti-wordtokensandanoveralllargervocabularycanbetterarticulatethevariousoptionsthemodelconsidersata givenpointinthetext.

TwoofthemostcommontokenizersusedforLMsareGPT2/3'sBytes-Pair-Encoding (BPE) tokenizer (Sennrichetal., 2016; Radfordetal., 2019; Brownetal., 2020) andT5'sSentencePiece (SP) tokenizer (KudoandRichardson, 2018; Raffeletal., 2020), eachemployinga tokenvocabularyofatmost50Ktokens, whereeachtokenrepresentseithera wordora wordpiece. GPT'sBPEtokenizerisa bitmoreefficientthanT5'sSPonEnglishtext, withanaverageof0.227tokensperbyte (TPB) vs. 0.256forT5 (lowerisbetter). Inotherdomains, e.g., source-codeorarticlesinLaTeX, bothtokenizersareonparwith0.37TPB. SeeTable2 forperdomainstatistics. Torepresenttextmoreefficiently, wetraineda SPtokenizerwitha largerbudgetof256Kvocabularyitemsandwithoutrestrictingittowordboundaries. Theresultingvocabularycontainsa richmixtureofwordpieces, wholewords, andmulti-wordexpressions, witha fallbacktounicodebytesforout-of-vocabularyinstances. Therehavebeenseveralpriorworks (Chenetal., 2016; Diaoetal., 2020; Zhangetal., 2021) onutilizinglargervocabularies, oraugmentingwordvocabularieswithphrasevocabularies, butincreasingthetokenefficiencywasnotthefocusoftheseearlierworks, andthisapproachwasneveremployedinmodelsofJurassic-1'sscale (tothebestofourknowledge).

Theresultofourimprovedtokenizerisa vastlymoreefficienttextrepresentationwitha 0.163TPBonEnglishtextand0.29onnon-Englishdomains (seeTable2 forfulldetails). Inotherwords, wecanrepresentthesametextwith28% fewertokensthanGPT-3, enablingustoprocessqueriesupto1.4× fasterwhenusingthesamearchitecture, andnearly1.8× whenaccountingforthearchitecturalspeedupsreportedintheprevioussection. Alternatively, ifweusethesamemaximalsequencelengthof2048tokens, wecanrepresent39% moretext, allowingourmodeltocovermorecontentduringtrainingandleveragelongerpromptsinfew-shotsettings. Seesection3 foreffectonfew-shotperformance.

Furthermore, byincorporatingmulti-wordexpressionsintoourvocabulary, itismorecloselyalignedwiththesemanticunitsofthetext, includingbothnamedentitiesandcommonphrases (seeexamplesinTable3). Tokenizingaccordingtothesemanticunitsofthetexthasseveraladvantages, suchasmoresample-efficienttraining (Levineetal., 2021), andmoreinterpretabledecoding, asillustratedinFigure1.

Increasingthevocabularydoeshavesomeveryminordrawbacks. Namely, itrequiresmorememorytostoretheadditionalparametersofthevocabularyembeddinglayer, aswellasmorecomputingresourcestocalculatethetokenprobabilitiesacrossthelargervocabulary. Yet, forlarge-scaleLMs, theseadditionalcostsarenegligibleincomparison

| Corpus | Curie (≈GPT\-36.7B) | J1\-Large | Davinci (≈GPT\-3175B) | J1\-Jumbo |
|-------------------|------------------------|-------------|--------------------------|-------------|
| arXiv | \-0.639 | \-0.515 | \-0.581 | \-0.471 |
| Books3 | \-0.611 | \-0.628 | \-0.556 | \-0.579 |
| C4 | \-0.547 | \-0.499 | \-0.501 | \-0.455 |
| DMMath | \-0.989 | \-0.734 | \-0.950 | \-0.719 |
| EnronEmails | \-0.718 | \-0.530 | \-0.664 | \-0.431 |
| Freelaw | \-0.470 | \-0.401 | \-0.424 | \-0.356 |
| GitHub | \-0.496 | \-0.307 | \-0.447 | \-0.248 |
| Gutenberg | \-0.873 | \-0.672 | \-0.806 | \-0.617 |
| Hackernews | \-0.726 | \-0.644 | \-0.676 | \-0.602 |
| NIHExporter | \-0.462 | \-0.440 | \-0.424 | \-0.409 |
| OpenSubtitles | \-0.695 | \-0.667 | \-0.646 | \-0.609 |
| PhilPapers | \-0.557 | \-0.571 | \-0.501 | \-0.514 |
| PileCC | \-0.534 | \-0.512 | \-0.484 | \-0.464 |
| PubmedAbstracts | \-0.475 | \-0.441 | \-0.433 | \-0.407 |
| PubmedCentral | \-0.523 | \-0.433 | \-0.478 | \-0.401 |
| StackExchange | \-0.583 | \-0.494 | \-0.536 | \-0.454 |
| UbuntuIRC | \-0.716 | \-0.644 | \-0.656 | \-0.594 |
| USPTO | \-0.426 | \-0.407 | \-0.392 | \-0.372 |
| YoutubeSubtitles | \-0.616 | \-0.620 | \-0.565 | \-0.572 |
| Average | \-0.613 | \-0.535 | \-0.564 | \-0.488 |

Table4: Wereportaveragelog-probabilitiesperbyteonvarietyofcorpora (Raffeletal., 2020; Gaoetal., 2020) toillustratethesuitabilityofourmodelsonvariousdomains. OnalmostallcorporaourJurassic-1modelsarewellaheadoftheirGPT-3'scounterparts.

withalltheotherlayers. ThevocabularyembeddingforJ1-Jumbo, forexample, requires3.6Bparameters, whicharejust2% ofallparameters.

### 2.3TrainingDetailsTrainingmodelsofthesescalesposesmanyengineeringchallengesbecauseoftheirimmensesize. Simplystoring178Bparametersrequiresmorethan356GBofmemoryinhalf-precision, whereaseventhelargestGPUsavailabletodayhavea maximummemoryof80GB, andthisisbeforetakingintoaccounttheoptimizer'sstateortheintermediatecalculationsusedbybackwardsimulations. Therefore, trainingmustbedistributedacrosstensorhundredsofnodes, eachwithmultipleGPUs, whichpresentsitsownsetofchallenges, e.g., loadingandsavingthesehugecheckpointsacrossmanynodescreatescommunicationbottlenecks. Toutilizetheavailablenodesefficiently, wereliedona combinationofdata, model, andpipelineparallelismstrategies, aswellasdistributivelyshardingtheoptimizer'sstateparametersasproposedinRajbhandarietal. (2020). WebasedourimplementationonbothDeepSpeed (Rasleyetal., 2020) andMegatronLM (Narayananetal., 2021). Ourmodelwastrainedwiththeconventionalself-supervisedauto-regressivetrainingobjectiveon300Btokensdrawnfrompubliclyavailableresources, attempting, inpart, toreplicatethestructureofthetrainingdataasreportedinBrownetal. (2020). Asfortheoptimizationprocedure, heretoowefollowedthehyper-parameterssuggestedinBrownetal.

(2020) foreachcorrespondingmodelsize. Namely, weuseda baselearningrateof1.2 × 10−4and0.6 × 10−4, anda batchsizeof2Mand3.2Mtokens, forJ1-LargeandJ1-Jumbo, respectively. Wealsouseda linearwarm-upoverroughlythefirst375milliontokens, andgraduallyincreasedthebatchsizefrom32Ktokensuptoitstargetvalueforthefirstfewbilliontokens.

# 3EvaluationOurJurassic-1models, aswellastheirGPT-3counterparts, wereextensivelytestedona varietyoftasks. Asa firststep, wediscussthemodel'sabilitytocompletetextacrossdifferentdomains. Theabilitytoperformzero-shotlearningisourprimarycriteriaforevaluatingthemodel'sabilitytosolvea widevarietyoftasks, sinceitislessvulnerabletopromptchoicethanthefew-shotsetting, andthusmorestableandconsistent. Finally, wedemonstratethefew-shotcapabilitiesofourmodelonselecttasks. WefindthatinallcasesourmodelsperformeitheronparorbetterthantheirGPT-3counterparts.

| QuestionFormat | AnswerFormat | Newlines | ARC\-Challenge | ARC\-Easy | RACE\-middle | RACE\-high |
|-------------------|--------------------|------------|------------------|-------------|----------------|--------------|
| Question: | Answer: | 1 | 48.1% | 67.1% | 56.6% | 46.5% |
| Question: | Answer: | 2 | 49.5% | 67.0% | 55.6% | 45.5% |
| question: | answer: | 1 | 48.5% | 67.2% | 56.3% | 46.5% |
| question: | answer: | 2 | 49.7% | 68.0% | 56.1% | 46.4% |
| Q: | A: | 1 | 47.7% | 66.1% | 56.6% | 45.9% |
| Q: | A: | 2 | 47.5% | 66.9% | 55.9% | 45.7% |
| | StandardDeviation | | 0.90% | 0.59% | 0.39% | 0.43% |

| Task | TestSize | Curie (≈GPT\-36.7B) | J1\-Large | Davinci (≈GPT\-3175B) | J1\-Jumbo |
|----------------|-------------|------------------------|-------------|--------------------------|-------------|
| ARC\-Challenge | 1172 | 41.7% | 41.7% | 50.2% | 48.1% |
| ARC\-Easy | 2376 | 60.3% | 62.2% | 69.2% | 67.1% |
| BoolQ | 3270 | 66.1% | 65.0% | 75.9% | 73.5% |
| HellaSwag | 10042 | 68.4% | 71.9% | 79.3% | 79.3% |
| PIQA | 1838 | 76.8% | 78.8% | 80.1% | 81.4% |
| RACE\-high | 3498 | 42.5% | 43.1% | 46.2% | 45.9% |
| RACE\-middle | 1436 | 53.2% | 53.9% | 56.0% | 56.6% |
| RTE | 277 | 55.2% | 59.2% | 57.4% | 62.8% |
| StoryCloze | 1871 | 77.6% | 80.2% | 83.1% | 83.1% |
| Winogrande | 1267 | 64.5% | 64.4% | 70.1% | 68.9% |
| Average | | 60.6% | 62.0% | 66.7% | 66.7% |

Table5: Variationsinaccuracyofquestion-answeringtasks (usingJ1-Jumbo) duetominorformatdifferencesinhowthequestionandanswersarepresentedtothemodel, andthenumberofseparatingnewlines. ResultsarebasedonourJ1-Jumbomodel, butsimilarvariationsoccurinJ1-LargeaswellasinGPT-3models.

Table6: Zero-shotresultsona selectsetoftasksfromBrownetal. (2020), accordingtotheformatsincludedintheirpaper.

Fortestingunderthesedifferentsettings, wedevelopedourownevaluationsuite, whichwepublishedathttps:
//github.com/ai21labs/lm-evaluation. WeutilizedOpenAI'scommercialAPI, notablyits *Curie* and *Davinci* endpoints, toevaluateGPT-3withoursuiteoftests. Itisnotspecifiedinitsdocumentation, butweestimatedbasedonourexperimentsandtheircorrelationtotheresultsreportedinBrownetal. (2020) thatCuriecorrespondstoGPT-36.7BandthatDavincicorrespondstoGPT-3175B. Forthetextcompletionevaluation, wemeasuredthelog-probabilityofa sampleofdocumentsofsimilarlengthsfroma givencorpus, normalizedbythenumberofbytesinthedocumentinordertobetokenizationindependent. Wetestedona varietyofdomainsfoundinthePiledataset (Gaoetal., 2020), includingweb (Pile'sCommonCrawlcorpus), academictextformattedinLaTeX (arXiv), fictionbooks (Books3, Gutenberg), computerprograms (GitHub), andmore. Inallbutthreecorpora, ourJurassic-1modelsoutperformtheirGPT-3counterparts. SeecompleteresultsinTable. 4. Ourmainevaluationiscenteredonzero-shotlearningcapabilities. Ourdecisiontousezero-shotlearningwasdrivenbyitssimplicityanddeterministicbehavior, whichdoesnotdependontheselectionofexamplesshownduringfew-shotlearning. Nevertheless, theformattingofa taskcanhaveanimpactonhowwella modelperforms, asdemonstratedinTable5, wherevariousformsofformattingquestion-answeringtaskshada significanteffectontheaccuracyofthecorrectanswer. Asthereisnostandardbenchmarkforzero-shotperformance, wetriedtoreplicatetheformatsusedbyBrownetal. (2020) fromthefewexamplesintheirfigures. Seetherepositoryforourevaluationsuitefortheformatspecificationweused. OurresultsonCurieandDavincideviateslightlyfromthosepresentedinBrownetal. (2020), perhapsduetotheslightdifferencesbetweenformatsusedinpresentingthetask, andthepossibilitiesthatOpenAI'sAPIisbasedona slightlydifferentsetofmodelsthanthosepresentedinBrownetal. (2020). Nevertheless, forthemostpartthereisa strongcorrelationwiththeresultsreportedintheirpaper, asmentionedabove. Whencomparedtoourmodels, weseevariedresults, whereonsometaskstheJurassic-1modelscomeaheadandinsomeGPT-3. Onaverage, weseethatbothmodelsattainthesameperformance. SeecompleteresultsinTable. 6. Whileonzero-shotbothGPT-3andJ1attainonparresults, themainadvantageofJ1isthatitreadstextmoreefficiently.

Oneofitsbenefitsisthatinfew-shotlearningsettingsmoretrainingexamplescanfitintheprompt. Noteverytaskrequiresmanyexamples, andinsometasksaddingtoomanycouldevenhurtperformance. Inourexperience, question-answeringtasksusuallyneedonlya fewexamples. However, inmanyreal-worldscenarios, havingmore

![5_image_0.png](5_image_0.png)

Figure2: Resultsforfew-shotlearningontheDBPedia-14andTREC-6text-classificationtasks. Theverticallinesignifythemaximalnumberofexamplesthatwouldfitinthepromptfora GPT-3model. Asshown, J1-Largeisabletoattainbetterresultsbyallowingformoretrainingexamplestofitinthepromptforthesamenumberoftokens.

examplescouldhavea significantimpactonperformance. Thisisespeciallytrueincaseswherethetaskhasmanyedgecasesorismorefuzzilydefined, andsomanyexamplesareneededtoconveytothelanguagemodeltheexpectedresult, e.g., paraphrasingsentencesorsummarizinga document. Sinceitcanbedifficulttoconsistentlymeasurethesekindoftaskswithouthumanevaluation, weinsteadturntotext-classificationtasksovera largesetofclasses. Whenmanyclassesareinvolved, moreexamplesareneededtoproperlyspecifyeachclass. WeevaluatedJ1andGPT-3ontwotext-classificationdatasets, namely, DBPedia-14andTREC-6. InbothcasesweseethattheaccuracycanbedramaticallyimprovedbyaddingmoreexamplesthancanfitintoGPT-3'scontextlength. SeeourresultsinFigure2. Asa finalremark, wewishtocommentontheissueofbiasandtoxicityfoundinourmodel, aswellaspracticallyalllanguagemodelsinusetoday. Aswasobservedincountlessotherworks (Shengetal., 2019; BordiaandBowman, 2019; deVassimonManelaetal., 2021; Nadeemetal., 2021), languagemodelsabsorbbiasesandtoxicityexpressedinthetextstheyweretrainedon, andarepronetoreplicatingthem. Ourmodelisnodifferent, andindeedmanylanguagebiasescanbeobservedwhenusingit, e.g., adoctorismorelikelytobeassociatedwiththepronoun "he", whilea nurseismorelikelytobeassociatedwiththepronoun "she". SeeTable7 fora thoroughcomparisonofourmodelsandtheirGPT3counterpartsontheStereoSet (Nadeemetal., 2021) biasbenchmark. WhileitappearstheJurassic-1modelsaremarginallylessbiasedthanGPT3, thisismerelyonebenchmarkandwedonotwishtooverstateourclaims. Whentheseissuesarecarelesslyignored, employinglanguagemodelscouldleadtounintendedanddiscriminativeconsequences, andifdeployedinwidespreadsettingscouldcauseactualharmtosociety. Manymethodscouldbeusedtomitigatetheseissues, e.g., bycarefullyengineeringprompts, orfilteringsuspectedresults - whichwestronglyrecommendusersofourmodelstoadopt - butthisisanactiveareaofresearch (Sunetal., 2019; Benderetal., 2021) andtheproblematlargeisfarfromsolved. Wearefully-committedtoengagingwiththecommunity, constantlymonitoringdevelopmentsinthisarea, andincorporatingsuchmethods, astheybecomemature, intoourmodels. WeinviteanyoneinterestedinconductingresearchonorotherwisepromotingAIethicsandsafetytocontactusatsafety@ai21.comandexploreopportunitiesforcollaboration.

# 4SummaryWereleasedJurassic-1, apairofauto-regressivelanguagemodels, includingboththe178B-parameterJ1-Jumbo, aswellasitssmallersibling, J1-Large, with7Bparameters. Themodelsutilizea moreefficientarchitectureandtokenizer, whichsignificantlyspeedsupinference. Inaddition, sinceourtokenizercanfitmoretextinthesamecontextlength, moreexamplescanbeincludedinfew-shotlearningsettings. WeevaluatedourJurassic-1modelsondatacompletion, zero-shotlearningandfew-shotlearning. Ourzero-shotandfew-shotevaluationcodeismadepubliclyavailable. WefindthatourJurassic-1modelscanpredicttextfroma broadersetofdomains (web, academic, legal, sourcecode, andmore) thanGPT-3, achievecomparableperformanceinzero-shotsettings, andcanbesuperiortoGPT-3infew-shot, duetotheirabilitytofitmoreexamplesintoa prompt.

# ReferencesTomBrown, BenjaminMann, NickRyder, MelanieSubbiah, JaredD Kaplan, PrafullaDhariwal, ArvindNeelakantan, PranavShyam, GirishSastry, AmandaAskell, SandhiniAgarwal, ArielHerbert-Voss, GretchenKrueger, TomHenighan, RewonChild, AdityaRamesh, DanielZiegler, JeffreyWu, ClemensWinter, ChrisHesse, MarkChen, EricSigler, MateuszLitwin, ScottGray, BenjaminChess, JackClark, ChristopherBerner, SamMcCandlish, Alec

| Task | Metric | Curie (≈GPT\-36.7B) | J1\-Large | Davinci (≈GPT\-3175B) | J1\-Jumbo |
|-----------------|----------|------------------------|-----------------|--------------------------|-------------|
| | | | Intra\-Sentence | | |
| Gender | LM | 92.7 | 92.6 | 93.9 | 92.9 |
| | SS | 72.2 | 72.1 | 73.3 | 75.4 |
| | ICAT | 51.6 | 51.6 | 50.1 | 45.7 |
| Profession | LM | 91.1 | 91.7 | 91.9 | 92.3 |
| | SS | 65.3 | 64.2 | 66.6 | 67.0 |
| | ICAT | 63.2 | 65.7 | 61.4 | 60.9 |
| Race | LM | 94.0 | 93.6 | 94.0 | 93.6 |
| | SS | 64.6 | 62.8 | 68.6 | 65.6 |
| | ICAT | 66.5 | 69.7 | 59.0 | 64.3 |
| Religion | LM | 92.0 | 92.1 | 92.8 | 91.4 |
| | SS | 69.5 | 65.2 | 65.8 | 66.2 |
| | ICAT | 56.2 | 64.0 | 63.4 | 61.8 |
| Intra\-Sentence | LM | 92.7 | 92.7 | 93.1 | 92.9 |
| Overall | SS | 66.1 | 64.6 | 68.3 | 67.4 |
| | ICAT | 62.9 | 65.7 | 59.0 | 60.6 |
| | | | Inter\-Sentence | | |
| Gender | LM | 83.7 | 83.2 | 82.6 | 80.9 |
| | SS | 59.5 | 58.9 | 63.2 | 59.5 |
| | ICAT | 67.9 | 68.4 | 60.7 | 65.4 |
| Profession | LM | 81.7 | 80.8 | 78.3 | 78.5 |
| | SS | 56.9 | 53.1 | 58.1 | 56.5 |
| | ICAT | 70.4 | 75.7 | 65.7 | 68.3 |
| Race | LM | 84.7 | 82.8 | 83.1 | 82.5 |
| | SS | 51.6 | 47.6 | 52.2 | 50.6 |
| | ICAT | 82.0 | 78.8 | 79.4 | 81.5 |
| Religion | LM | 87.1 | 84.9 | 89.0 | 85.2 |
| | SS | 54.2 | 53.2 | 52.9 | 54.5 |
| | ICAT | 79.8 | 79.4 | 83.9 | 77.5 |
| Inter\-Sentence | LM | 83.5 | 82.2 | 81.5 | 80.8 |
| Overall | SS | 54.7 | 51.3 | 55.9 | 54.1 |
| | ICAT | 75.7 | 80.0 | 71.9 | 74.2 |
| OverallScore | LM | 88.1 | 87.4 | 87.3 | 86.9 |
| | SS | 60.4 | 58.0 | 62.0 | 60.7 |
| | ICAT | 69.8 | 73.5 | 66.3 | 68.3 |

Table7: AbiasevaluationaccordingtotheStereoSet (Nadeemetal., 2021) benchmark, comparingthetendencyofthemodeltopreferstereotypestoanti-stereotypesonvariousaspects (gender, profession, race, andreligion). Theevaluationisdividedtointer-sentenceandintra-sentencetext-completiontasks. TheLMScoremeasurestheaccuracyofanLMtoprefera relevantcompletiontoa sentenceoveranunrelatedcompletion. TheSSScoremeasurestheprobabilityofanLMtoprefera stereotypecompletiontoananti-stereotype - an *ideal* scorewouldbe50, signifyingnobiaseitherway. TheICATScoreisanaggregatedscorethatisequaltoICAT = LMS ×
min(SS,100−SS)
50 , where100representstheidealmodel. Toaccommodateforourmulti-wordtokenizer, wenormalizelog-probsofcompletionsbynumberofcharactersratherthannumberoftokens, whichwefoundtohelpGPT-3modelsaswell.

Radford, IlyaSutskever, andDarioAmodei. Languagemodelsarefew-shotlearners. InH. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, andH. Lin, editors, *AdvancesinNeuralInformationProcessingSystems*, volume33, pages1877-1901. CurranAssociates, Inc., 2020. URLhttps://proceedings.neurips.cc/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

JeffRasley, SamyamRajbhandari, OlatunjiRuwase, andYuxiongHe. Deepspeed: Systemoptimizationsenabletrainingdeeplearningmodelswithover100billionparameters. In *Proceedingsofthe26thACMSIGKDDInternational* ConferenceonKnowledgeDiscovery & DataMining, KDD '20, page3505-3506, NewYork, NY, USA, 2020. AssociationforComputingMachinery. ISBN9781450379984. doi: 10.1145/3394486.3406703. URLhttps:
//doi.org/10.1145/3394486.3406703.

AshishVaswani, NoamShazeer, NikiParmar, JakobUszkoreit, LlionJones, AidanN Gomez, ŁukaszKaiser, andIlliaPolosukhin. Attentionisallyouneed. InI. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, andR. Garnett, editors, *AdvancesinNeuralInformationProcessingSystems*, volume30. CurranAssociates, Inc., 2017. URLhttps://proceedings.neurips.cc/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

AlecRadford, JeffWu, RewonChild, DavidLuan, DarioAmodei, andIlyaSutskever. Languagemodelsareunsupervisedmultitasklearners. 2019.

YoavLevine, NoamWies, OrSharir, HofitBata, andAmnonShashua. Limitstodepthefficienciesofself-attention. InH. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, andH. Lin, editors, AdvancesinNeuralInformationProcessingSystems, volume33, pages22640-22651. CurranAssociates, Inc., 2020. URLhttps://proceedings.neurips. cc/paper/2020/file/ff4dfdf5904e920ce52b48c1cef97829-Paper.pdf.

RicoSennrich, BarryHaddow, andAlexandraBirch. Neuralmachinetranslationofrarewordswithsubwordunits. InProceedingsofthe54thAnnualMeetingoftheAssociationforComputationalLinguistics (Volume1: LongPapers), pages1715-1725, Berlin, Germany, August2016. AssociationforComputationalLinguistics. doi: 10.18653/v1/ P16-1162. URLhttps://www.aclweb.org/anthology/P16-1162.

TakuKudoandJohnRichardson. SentencePiece: Asimpleandlanguageindependentsubwordtokenizeranddetokenizerforneuraltextprocessing. InProceedingsofthe2018ConferenceonEmpiricalMethodsinNaturalLanguageProcessing: SystemDemonstrations, pages66-71, Brussels, Belgium, November2018. AssociationforComputationalLinguistics. doi: 10.18653/v1/D18-2012. URLhttps://www.aclweb.org/anthology/D18-2012.

ColinRaffel, NoamShazeer, AdamRoberts, KatherineLee, SharanNarang, MichaelMatena, YanqiZhou, WeiLi, andPeterJ. Liu. Exploringthelimitsoftransferlearningwitha unifiedtext-to-texttransformer. JournalofMachineLearningResearch, 21(140):1-67, 2020. URLhttp://jmlr.org/papers/v21/20-074.html.

WenlinChen, DavidGrangier, andMichaelAuli. Strategiesfortraininglargevocabularyneurallanguagemodels. InProceedingsofthe54thAnnualMeetingoftheAssociationforComputationalLinguistics (Volume1: LongPapers), pages1975-1985, Berlin, Germany, August2016. AssociationforComputationalLinguistics. doi: 10.18653/v1/ P16-1186. URLhttps://aclanthology.org/P16-1186.

ShizheDiao, JiaxinBai, YanSong, TongZhang, andYonggangWang. ZEN: Pre-trainingChinesetextencoderenhancedbyn-gramrepresentations. In *FindingsoftheAssociationforComputationalLinguistics: EMNLP2020*, pages4729-4740, Online, November2020. AssociationforComputationalLinguistics. doi: 10.18653/v1/2020. findings-emnlp.425. URLhttps://aclanthology.org/2020.findings-emnlp.425.

XinsongZhang, PengshuaiLi, andHangLi. Ambert: Apre-trainedlanguagemodelwithmulti-grainedtokenization, 2021.

YoavLevine, BarakLenz, OpherLieber, OmriAbend, KevinLeyton-Brown, MosheTennenholtz, andYoavShoham.

{PMI}-masking: Principledmaskingofcorrelatedspans. In *InternationalConferenceonLearningRepresentations*,
2021. URLhttps://openreview.net/forum?id=3Aoft6NWFej.

SamyamRajbhandari, JeffRasley, OlatunjiRuwase, andYuxiongHe. Zero: Memoryoptimizationstowardtrainingtrillionparametermodels. InSC20: InternationalConferenceforHighPerformanceComputing, Networking, StorageandAnalysis, pages1-16, 2020. doi: 10.1109/SC41405.2020.00024.

DeepakNarayanan, MohammadShoeybi, JaredCasper, PatrickLeGresley, MostofaPatwary, VijayAnandKorthikanti, DmitriVainbrand, PrethviKashinkunti, JulieBernauer, BryanCatanzaro, AmarPhanishayee, andMateiZaharia. Efficientlarge-scalelanguagemodeltrainingongpuclusters, 2021.

LeoGao, StellaBiderman, SidBlack, LaurenceGolding, TravisHoppe, CharlesFoster, JasonPhang, HoraceHe, AnishThite, NoaNabeshima, ShawnPresser, andConnorLeahy. ThePile: An800gbdatasetofdiversetextforlanguagemodeling. *arXivpreprintarXiv:2101.00027*, 2020.

MoinNadeem, AnnaBethke, andSivaReddy. StereoSet: Measuringstereotypicalbiasinpretrainedlanguagemodels. In *Proceedingsofthe59thAnnualMeetingoftheAssociationforComputationalLinguisticsandthe* 11thInternationalJointConferenceonNaturalLanguageProcessing (Volume1: LongPapers), pages5356-5371, Online, August2021. AssociationforComputationalLinguistics. doi: 10.18653/v1/2021.acl-long.416. URLhttps://aclanthology.org/2021.acl-long.416.

EmilySheng, Kai-WeiChang, PremkumarNatarajan, andNanyunPeng. Thewomanworkedasa babysitter: Onbiasesinlanguagegeneration. In *Proceedingsofthe2019ConferenceonEmpiricalMethodsinNaturalLanguageProcessing* andthe9thInternationalJointConferenceonNaturalLanguageProcessing (EMNLP-IJCNLP), pages3407-3412, HongKong, China, November2019. AssociationforComputationalLinguistics. doi: 10.18653/v1/D19-1339. URLhttps://aclanthology.org/D19-1339.

ShikhaBordiaandSamuelR. Bowman. Identifyingandreducinggenderbiasinword-levellanguagemodels. InProceedingsofthe2019ConferenceoftheNorthAmericanChapteroftheAssociationforComputationalLinguistics: StudentResearchWorkshop, pages7-15, Minneapolis, Minnesota, June2019. AssociationforComputationalLinguistics. doi: 10.18653/v1/N19-3002. URLhttps://aclanthology.org/N19-3002.

DanieldeVassimonManela, DavidErrington, ThomasFisher, BorisvanBreugel, andPasqualeMinervini. Stereotypeandskew: Quantifyinggenderbiasinpre-trainedandfine-tunedlanguagemodels. InProceedingsofthe16thConferenceoftheEuropeanChapteroftheAssociationforComputationalLinguistics: MainVolume, pages2232- 2242, Online, April2021. AssociationforComputationalLinguistics. URLhttps://aclanthology.org/2021. eacl-main.190.

TonySun, AndrewGaut, ShirlynTang, YuxinHuang, MaiElSherief, JieyuZhao, DibaMirza, ElizabethBelding, Kai-WeiChang, andWilliamYangWang. Mitigatinggenderbiasinnaturallanguageprocessing: Literaturereview. In *Proceedingsofthe57thAnnualMeetingoftheAssociationforComputationalLinguistics*, pages1630-
1640, Florence, Italy, July2019. AssociationforComputationalLinguistics. doi: 10.18653/v1/P19-1159. URLhttps://aclanthology.org/P19-1159.

EmilyM. Bender, TimnitGebru, AngelinaMcMillan-Major, andShmargaretShmitchell. Onthedangersofstochasticparrots: Canlanguagemodelsbetoobig? InProceedingsofthe2021ACMConferenceonFairness, Accountability, andTransparency, FAccT '21, page610-623, NewYork, NY, USA, 2021. AssociationforComputingMachinery. ISBN9781450383097. doi: 10.1145/3442188.3445922. URLhttps://doi.org/10.1145/3442188.3445922.