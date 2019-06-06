import tarfile
import os
import xml.etree.ElementTree as etree
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

archived_xml = '../data/EuroSense/eurosense.v1.0.high-precision.xml'
mapping_file = "../resources/bn2wn_mapping.txt"

#BabelNet synset ids to WordNet offset ids
synset_id = pd.read_csv(mapping_file, sep = "\t", error_bad_lines=False, header = None)
synset_id.columns = ['BabelNet', 'WordNet']
BabelNet_id = list(synset_id['BabelNet'])

context = etree.iterparse(archived_xml, events=("start", "end"))

N_annotations_present, textExists = False, False
bigrams, unigrams = [], []
sentence = ''
#total = 153763675
total = 290000000
#Write to file the last sentence written so in case of a crash, you can write and process starting from that sentence

f = open("../resources/parsed_corpora_RECOVERY_final.txt", "r+")
annotations_per_sentence = open("../resources/parsed_corpora_annotations_final.txt", "a")
data = f.readlines()
#checks for last iteration if exists
last_iteration, written_lines = [int(i) for i in data[-1].split(",")]
print("starting processing from iteration # {}\t Written lines so far: {}".format(last_iteration, written_lines))

############################
## main iteration start ###
###########################
with open('../resources/parsed_corpora_final.txt', 'a', encoding='utf-8') as file:
    for idx, (event, elem) in enumerate(tqdm(context)):

        #checks current idx so if preprocessing crashes, It start processing from this iteration
        if last_iteration < idx:
            #taking start of each sentence
            if elem.tag=='sentence' and event == 'start':
                sentence_idx = elem.get("id")

            #taking the sentence text (English only)
            if elem.tag == 'text' and elem.get("lang") == 'en':
                #checking if the text is not a None
                if elem.text!= None:
                    sentence += elem.text
                    textExists =  True
                else:
                    textExists = False

            if textExists:
                #taking the start of the sentence annotations (English only)
                if elem.tag == 'annotation' and elem.get("lang") == 'en' and event == 'start':
                    # checking if annotation is in the mapping from BabelNet to WordNet

                    current_synset_id = elem.text
                    if current_synset_id in BabelNet_id:
                        N_annotations_present=True
                        anchor = elem.get("anchor")
                        replace_by = "_".join(elem.get("lemma").split(" ")) + "_" + elem.text

                        #write N-grams and unigrams into memory
                        if len(anchor.split(" "))>1:
                            bigrams.append([anchor, replace_by])
                        elif len(anchor.split(" "))==1:
                            unigrams.append([anchor, replace_by])

                #after iterating through all annotations, write the transformed sentence
                if elem.tag=='sentence' and event == 'end':
                    if N_annotations_present:
                        annotations = 0

                        #ensure longest n-grams dominate, then replace
                        bigrams = sorted(bigrams, key = lambda k: len(k[0].split(" ")), reverse = True)

                        #replace n-grams
                        sentence = sentence.replace("-"," ")
                        for orig, replace in bigrams:
                            annotations+=1
                            wrap = lambda x: " "+x+" "
                            sentence = sentence.replace(wrap(orig), wrap(replace))

                        #split before unigrams so nothing gets replaced twice
                        sentence = sentence.split(" ")

                        #UNIGRAMS replacement
                        for index, (orig, replace) in enumerate(unigrams):
                            if orig in sentence:
                                annotations+=1
                                sentence[sentence.index(orig)] = replace

                        #join back to write to file
                        sentence = " ".join(sentence)

                        #write to file
                        file.write(sentence+"\n")
                        annotations_per_sentence.writelines(str(annotations)+"\n")
                        written_lines+=1

                        #reset
                        bigrams, unigrams, N_annotations_present, sentence = [], [], False, ''

                    else:
                        sentence = ''
                    f.writelines("{},{}\n".format(str(idx), str(written_lines)))
        #debugging
        if (idx+1)%5000000==0:
            print("Number of actually written lines: {}\n {:.3f}% done".format(written_lines, ((idx+last_iteration)/total)*100))
            #break
            #delete to ease memory
        elem.clear()
    del context

##########################
## main iteration end ###
#########################
print("_"*120)
print("Number of actually written lines: {}".format(written_lines))

f.close()
annotations_per_sentence.close()
