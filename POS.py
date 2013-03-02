#################################################################################
#   						Main Program										#
#################################################################################
import sys
import os
import operator	
from collections import OrderedDict			
import random
import math
			
# Declaring all the variables
trainFile = './data/bc.train'
testFile = './data/bc.test'
S=["ADJ", "ADV", "ADP", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", "X","."]
pS1={"ADJ":0, "ADV":0, "ADP":0, "CONJ":0, "DET":0, "NOUN":0, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}
cPoS={"ADJ":0, "ADV":0, "ADP":0, "CONJ":0, "DET":0, "NOUN":0, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}
pPoS={"ADJ":0, "ADV":0, "ADP":0, "CONJ":0, "DET":0, "NOUN":0, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}
cS1={"ADJ":0, "ADV":0, "ADP":0, "CONJ":0, "DET":0, "NOUN":0, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}
sentenceDList={};wordDict={};pSplusS={};cSplusS={};cWordGivenTag={};pWordGivenTag={};probMarg={};mSmplngProbableTag={}
totalWCount=0;totalFirstWCount=0; epsilon=0.00000000000000000000000000001

# we use a very large prime number over here to scale
def sample(sampleProbability):
    a= random.choice([x for x in sampleProbability for y in range(int(1/(1-sampleProbability[x]))% 188748146801)])
    return a
	
def sampling():
	for sentIndx, eSentence in sentenceDList.items():
		if sentIndx >= len(sentenceDList):
			break;
		forsample={};
		previous = probMarg[sentIndx]  #we use the max marginal probability over here
		for wrdIndx, wrdPos in eSentence.items():
			key=(sentIndx, wrdIndx, wrdPos[0])
			mSmplngProbableTag[key] = previous
			if wrdIndx==len(eSentence):
				break
			i=wrdIndx+1
			for tags in S:
				observed=epsilon
				if (eSentence[i][0],tags) in pWordGivenTag:
					observed = pWordGivenTag[(eSentence[i][0],tags)]
					
				xyz =observed*pSplusS[tags,previous] #Calculate the probability distribution for next word
				forsample[tags]=xyz
				
			previous = sample(forsample) #call sample function which returns with weighted probability
	return mSmplngProbableTag		

'''
Method Description:
This method is to compute the accuracy of the sentences and words.
Arguments:
origSentence= The test sentence on which we are trying to test
computedPoS= This is a dictionary which contains the 
classifier= The model we used to classify the PoS (Part Of Speech)
'''
def computeAccuracy (origSentence, computedPoS, classifier):
	compCorrectlyW=0; compCorrectlyS=0;	totalSent=0; totalWords=0
	
	for sentIndx,eSentence in origSentence.items():
		totalSent += 1
		compSentCorr=1
		for wrdIndx, wrdPoS in eSentence.items():
			totalWords += 1
			compPoS = computedPoS[sentIndx, wrdIndx, wrdPoS[0]]
			grndTruthPoS = wrdPoS[1]
			if (grndTruthPoS == compPoS):
				compCorrectlyW += 1
			else:
				compSentCorr=0
		if (compSentCorr == 1):
			compCorrectlyS += 1
			
	print "\nTotal Words:", totalWords, "sentences:", totalSent
	if (classifier == "Naive"):
		print "Part 2, Naive Graphical Model:"
	elif (classifier == "Bayes Net"):
		print "Part 3, Bayes Net:"
	else:
		print "Part 4, Sampling:"
	
	print "Words correct\t\t :", compCorrectlyW/float(totalWords) * 100
	print "Sentence correct\t :", compCorrectlyS/float(totalSent) * 100

	
'''
Method Description:
This method is to print the appropiate statistics.
Arguments:
origSentence: The test sentence on which we are trying to test
naiveComputedPoS, bayesNComputedPoS, samplngComputedPoS: 
This is a dictionary which contains the PoS (Part Of Speech) computed by different model
Naive, Bayes Net and , Sampling resp.
'''
def printStats (origSentence, naiveComputedPoS, bayesNComputedPoS, samplngComputedPoS):
	'''
	i=0
	for sentIndx,eSentence in origSentence.items():
		print "\n\nConsidering sentence\t:" ,
		for wrdIndx, wrdPoS in eSentence.items():
			print wrdPoS[0],
		print "\nGround truth\t\t:",
		for wrdIndx, wrdPoS in eSentence.items():
			print wrdPoS[1],	
		print "\nNaive\t\t\t:",	
		for wrdIndx, wrdPoS in eSentence.items():
			print naiveComputedPoS[sentIndx, wrdIndx, wrdPoS[0]],	
		print "\nBayes Net\t\t:",	
		for wrdIndx, wrdPoS in eSentence.items():
			print bayesNComputedPoS[sentIndx, wrdIndx, wrdPoS[0]],	
		for j in range(1,6):
			print "\nSampling ",j, "\t\t:",
			for wrdIndx, wrdPoS in eSentence.items():
				print samplngComputedPoS[j][sentIndx, wrdIndx, wrdPoS[0]],		
	'''
	print "\n\n\n\nPERFORMANCE SUMMARY"
	print "----------------------------------------------------------------------------------"
	computeAccuracy (origSentence,naiveComputedPoS, "Naive")
	computeAccuracy (origSentence,bayesNComputedPoS, "Bayes Net")
	for j in range(1,5):	
		computeAccuracy (origSentence,samplngComputedPoS[j],"Sampling")

'''		
Description:
This part of code takes care of the Step 1 of the Assignment 2, i.e.
'''
#####################################################################################################
#							Step 1: Estimate Conditional Probability											#
#####################################################################################################	

# Reading the training file
f = open(trainFile, "r")
trainString=f.read()

# Trying the split the file into sentences.
sentlist=trainString.split("\n\n") # sentlist is a  list of all sentences.


#	This loop is going to create a dictionary of dictionary, of the whole file. and it of the below format#
#	{"SentenceIndx": {"WordIndx":"Word, PartofSpeech"}}
#	e.g.: 	{"1": {"1":"He", "NOUN", "2":"is", "VERB",........}}
#			{"2": {"1":"She", "NOUN", "2":"was", "VERB",........}}

sentIndex=1
for eSentence in sentlist:
	wordIndex=1
	wordDict={}
	wordlist=eSentence.split()
	for i in range(0, len(wordlist), 2):
		value=wordlist[i:i+2]
		wordDict[wordIndex]= (value[0], value[1])
		wordIndex += 1
	sentenceDList[sentIndex]=wordDict 
	sentIndex += 1
	
'''
# DEBUG: Printing the sentenceDList dicitionary
for eSentence, value in sentenceDList.items():
	print eSentence, value
'''

#	In the below loop we compute 3 things
#	the total number of words we have i.e. totalWCount
#	the total number of first words i.e.totalFirstWCount and, 
#	count of all the PoS we have from training data
for sentIndx,eSentence in sentenceDList.items():
	for wrdIndx, wrdPos in eSentence.items():
		totalWCount += 1
		pos=wrdPos[1]
		if (wrdIndx == 1):
			cS1[pos] += 1
			totalFirstWCount += 1
		cPoS[pos] += 1

#	Computing Prior P(S1) Probability
#	We are computing prior probabalities
#	pS1 and pPoS is a dictionary with key being the PoS and value being the probability of same.
#	e.g. pPoS=pS1 ={"NOUN": 0.23, "VERB": 0.12,........}
for tag in S:
	pS1[tag] =cS1[tag]/float(totalFirstWCount) # For this we just look the first words of all sentence
	pPoS[tag] =cPoS[tag]/float(totalWCount)

'''
#DEBUG: To print the pS1 to check for values
print "\n\n\n###########################################################"	
print "Step 1: Estimating the conditional probability tables....."
print "###########################################################"	
print "\nP(S1) = ", pS1	
'''

#	Computing P(Si+1|Si)
#	cSplusS, pSplusS is a dictionary with "PoS,PoS" is the key and 
#	the value being the counnt or probability of it respectively
#	e.g.: cSplusS={("NOUN","NOUN")=3, ("VERB","NOUN")=2,..... 144 entries}
for PoS1 in pS1.keys():
	for PoS2 in pS1.keys():
		pSplusS[PoS1,PoS2]=0.00000000001;
cSplusS={};
for sentIndx,eSentence in sentenceDList.items():
	for index, (wrdIndx, value) in enumerate(eSentence.items()):
		#print index;
		if (index != len(eSentence)-1 ):
			key1=eSentence[wrdIndx][1]
			key2=eSentence[wrdIndx+1][1]
			if (key1,key2) in cSplusS:
				cSplusS[key1, key2] = cSplusS[key1, key2]+1.0
			else:
				cSplusS[key1, key2] = 1.0;
			#print key1, key2, cSplusS[key1, key2]
		
for tagp in S:
	for tagfixed in S:
		if (tagp,tagfixed) in cSplusS:
			pSplusS[tagp, tagfixed]=cSplusS[tagp, tagfixed]/(float)(cPoS[tagfixed]);
			#print tagp, tagfixed, pSplusS[tagp, tagfixed];

'''
#DEBUG: To print the pS1 to check for values
# To Print the Probabilities which are <> 0			
print "\nOnly Printing Probabilities which are <> 0:"
print "------------------------------------------------------------"	
for keys in pSplusS.keys():
	if (pSplusS[keys] > 0):
		print "P(Si+1|Si)", keys, ":", pSplusS[keys]
'''

#	Computing P(W|Si) Probability
#	cWordGivenTag, pWordGivenTag is a dictionary with "Word,PoS" is the key and 
#	the value being the count or probability of it respectively
#	e.g.: cWordGivenTag={("He","NOUN")=3, ("He","VERB")=2,..... 12 entries}
for sentIndx,eSentence  in sentenceDList.items():
	for wrdIndx, wrdPos in eSentence.items():
		if (wrdPos in cWordGivenTag):
			cWordGivenTag[wrdPos] += 1;
		else:
			cWordGivenTag[wrdPos]=1;

for key in cWordGivenTag:
	# we do not need to check for "Divide by Zero" since we only have items only for non-zero values.
	posVal=key[1]
	pWordGivenTag[key] = cWordGivenTag[key]/float(cPoS[posVal]);
	
'''
#DEBUG: To Print and check the Probabilities
print "\nOnly Printing Probabilities P(W|Si) which are <> 0:"
print "------------------------------------------------------------"	
for keys,value in pWordGivenTag.items():
	if (pWordGivenTag[keys]>0):
	print "P(W|Si)", keys, ":", pWordGivenTag[keys]
'''


# *************************************************************************************************
# 							Reading Test file Step 3,4,5										
# *************************************************************************************************

#Declaring Variables
learnedEProb={};fLearnedBNetVar={};bNLearnedEachProb={};bLearnedBNetVar={};beta={};mProbableTag={};sentenceDList={};mBnProbableTag={}

# Reading the test file
f = open(testFile, "r")
testString=f.read()
testWordList=testString.split()
sentlist=testString.split("\n\n")
testotalWCount=0
index=1

sentIndex=1
for eSentence in sentlist:
	wordIndex=1
	wordDict={}
	wordlist=eSentence.split()
	for i in range(0, len(wordlist), 2):
		value=wordlist[i:i+2]
		wordDict[wordIndex]= (value[0], value[1])
		wordIndex += 1
	sentenceDList[sentIndex]=wordDict
	sentIndex += 1

#####################################################################################################
#							Step 2: Naive Bayes Learning											#
#####################################################################################################	
	
#	Creating variables to store the values:
#	key = sentIndx,wrdIndx,word
#	value = Probabilites
#	e.g.: 	learnedEProb={	(1,1,"He"): {"ADJ":0, "ADV":0, "ADP":0.23, "CONJ":0, "DET":0, "NOUN":0.11, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}, 
#							(1,1,"is"): {"ADJ":0, "ADV":0, "ADP":.012, "CONJ":0, "DET":0, "NOUN":0, "NUM":0.1, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}, 
#							....... total # of words}	

for sentIndx, eSentence in sentenceDList.items():
	for wrdIndx, wrdPos in eSentence.items():
		testotalWCount +=1
		learnedEProb[sentIndx,wrdIndx,wrdPos[0]]={"ADJ":0, "ADV":0, "ADP":0, "CONJ":0, "DET":0, "NOUN":0, "NUM":0, "PRON":0, "PRT":0, "VERB":0, "X":0,".":0}
		

for sentIndx, eSentence in sentenceDList.items():		
	for wrdIndx, wrdPos in eSentence.items():
		key=(sentIndx, wrdIndx, wrdPos[0])
		for eachPoS in S:
			if (wrdPos[0], eachPoS) in pWordGivenTag:
				#	Simply applying Bayes Rule
				#	But ignoring the normalizer since anyways we need to take the max among the probablities to find the PoS
				learnedEProb[key][eachPoS] = pWordGivenTag[wrdPos[0], eachPoS] * pPoS[eachPoS]
			else:
				learnedEProb[key][eachPoS] = epsilon

for key,value in learnedEProb.items():
	maxProbTag= max(value.items(), key=operator.itemgetter(1))[0]
	mProbableTag[key]=maxProbTag

naiveSentenceDList=sentenceDList
'''
# DEBUG: To check  the Maximum possible tags.
for key in sorted(mProbableTag.keys()):
	print key,mProbableTag[key]
'''

#####################################################################################################
#							Step 3: Bayes Network Learning											#
#####################################################################################################	

for sentIndx, eSentence  in sentenceDList.items():
	tau={} #this stores the forward probabilites
	for wrdIndx, wrdPos in eSentence.items():
		word = wrdPos[0]
		key=(sentIndx, wrdIndx, word)
		observed = epsilon

		if wrdIndx ==1 :   #if first word then probability is calculated as pS1 * P(word|tag)
			temp = {};
			for tag in S:
				if (word,tag) in pWordGivenTag:
						#print word,tag
						observed = pWordGivenTag[(word,tag)]
				temp[tag] = observed* pS1[tag];
			tau[wrdIndx] = temp;
			
		else:  #else we eliminate i-1
			temp={};
			for tag in S:
				sum=0.0
				observed=epsilon
				if (word,tag) in pWordGivenTag:
						#print word,tag
					observed=pWordGivenTag[(word,tag)]
				for tag2 in S:
					sum+=(pSplusS[(tag,tag2)]*tau[wrdIndx-1][tag2])
				temp[tag] = sum*observed
			tau[wrdIndx] = temp;

	beta={}
	for wrdIndx, wrdPos in reversed(eSentence.items()):
		word = wrdPos[0]
		key=(sentIndx, wrdIndx, word)
		observed = epsilon;
		if wrdIndx ==len(eSentence):
			temp = {}
			for tag in S:
				temp[tag] = 1.0 # assign beta(n) =1
			
			beta[wrdIndx] = temp
		else:
			temp={}
			for tag in S:
				sum=0.0
				word1 = eSentence[wrdIndx+1][0]
				for tag2 in S:
					if(word1,tag2) in pWordGivenTag:
						observed=pWordGivenTag[(word1,tag2)]  
					sum+= observed*pSplusS[(tag2,tag)]*beta[wrdIndx+1][tag2]
					observed=epsilon  # over here we use P(n+1word|tag)
				temp[tag] = sum
		beta[wrdIndx] = temp

	final ={}
	for i in tau.keys():
		final[i] ={}
		for tag in tau[i].keys():
			final[i][tag]=(tau[i][tag]*beta[i][tag])
		
			
	for wrdIndx, wrdPos in eSentence.items():
		key= (sentIndx, wrdIndx, wrdPos[0])
		maxProbTag=max(final[wrdIndx].items(),key=operator.itemgetter(1))[0]
		if (wrdIndx==1):
			probMarg[sentIndx]=maxProbTag
		mBnProbableTag[key]=maxProbTag

'''
for key, values in 	sorted(mBnProbableTag.items()):			
	print key, values
'''


#####################################################################################################
#							Step 4: Doing Sampling													#
#####################################################################################################	

#Here we simply call a method defined above, 5 times. 
mSmplngProbTag5={}
for sampleTimes in range(1,6):
	mSmplngProbTag5[sampleTimes]= sampling()


'''
for key, values in 	sorted(mSmplngProbTag5.items()):			
	print key, values
'''	
printStats(naiveSentenceDList,mProbableTag,mBnProbableTag,mSmplngProbTag5)
