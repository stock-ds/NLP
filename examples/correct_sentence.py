from sklearn.feature_extraction.text import CountVectorizer
from stock_ds.dictionary_tools import sc_correct_dictionary
from stock_ds.spellcorrect import correct_text
from stock_ds.spellcorrect.reconstruct import reconstruct_text

sentence = "Yoou cannot end a sentence with becausse becuse becaus is a conjunction."
cv = CountVectorizer().fit([sentence])
remap, corrections = sc_correct_dictionary(cv.vocabulary_)
print(cv.vocabulary_)
print(remap)
print(corrections)
print(reconstruct_text(sentence, corrections))
print(correct_text(sentence))
