# coding=utf8


import random
from sys import  getdefaultencoding

import sys

def GenerateRandomBanglaCharsOnly(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    C=0
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    stringlist=[]

    for i in range(num_words):
        NumberofAlphabets=random.randint(1,7)
        string=""
        for j in range(NumberofAlphabets):
            Character=random.randint(0,45)
            string+=charlist[Character]
        stringlist.append(string)
        # print(string)
    return stringlist
def GenerateRandomBanglaCharsWithModifier(num_words):

    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars="খ্র" # For generating the "Ri" Character
    b=(chars[3:6]+chars[6:9]) # The "Ri Character consits of two different charcters
    modifiers="া ে ি ী ু"
    charlist=[] # I have the chars in a list in specific index
    modlist=[] # Same for the modifiers
    for i in range(len(banglachars)/3):
        charlist.append(banglachars[i*3:(i+1)*3])
    for i in range(0,19,4):
        modlist.append(modifiers[i*3:(i+1)*3])
    modlist.append(b)
    stringlist = []
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    for i in range(num_words):
        string = ""
        NumberOfAlphabet = random.randint(1, 7)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier=random.randint(0,1)
            if(WithModifier_or_WithNoModifier):

                indexofchar=random.randint(0,32)
                string+=(charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier=random.randint(0,5)
                string+=(charlist[indexofchar]+modlist[indexofmodifier])
        stringlist.append(string)
    return stringlist

def GenerateRandomBanglaCharsWithModifierAndPunctuation(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character
    b = (chars[3:6] + chars[6:9])  # The "Ri Character consits of two different charcters
    modifiers = "া ে ি ী ু"
    puncuation=".,;\" :"
    punc = []
    for i in puncuation:
        punc.append(i)
    charlist = []  # I have the chars in a list in specific index
    modlist = []  # Same for the modifiers
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    for i in range(0, 19, 4):
        modlist.append(modifiers[i * 3:(i + 1) * 3])

    modlist.append(b)


    Total = charlist + modlist + punc



    stringlist = []
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    for i in range(num_words):
        string = ""
        NumberOfAlphabet = random.randint(1, 3)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier = random.randint(0, 1)
            if (WithModifier_or_WithNoModifier):

                indexofchar = random.randint(0, 32)
                string += (charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier = random.randint(0, 5)
                string += (charlist[indexofchar] + modlist[indexofmodifier])

        punrand=random.randint(0,5)
        string+=puncuation[punrand]

        NumberOfAlphabet = random.randint(1, 3)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier = random.randint(0, 1)
            if (WithModifier_or_WithNoModifier):

                indexofchar = random.randint(0, 32)
                string += (charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier = random.randint(0, 5)
                string += (charlist[indexofchar] + modlist[indexofmodifier])
        print(string)
        stringlist.append(string)
    return stringlist
def GenerateRandomEnglishLowerChars(num_words):
    stringlist=[]
    for i in range (num_words):
        string=""
        NumberofChars=random.randint(1,7)
        for j in range(NumberofChars):
            ind = random.randint(0,25)
            string+=chr(97+ind)
        print(string)
        stringlist.append(string)
    return stringlist

def GenerateRandomEnglishUpperChars(num_words):
    stringlist = []
    for i in range(num_words):
        string = ""
        NumberofChars = random.randint(1, 7)
        for j in range(NumberofChars):
            ind = random.randint(0, 25)
            string += chr(65 + ind)
        print(string)
        stringlist.append(string)
    return stringlist
def CombinedDataset(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character
    b = (chars[3:6] + chars[6:9])  # The "Ri Character consits of two different charcters
    modifiers = "া ে ি ী ু"
    puncuation = ".,;\" :"
    punc = []
    for i in puncuation:
        punc.append(i)
    charlist = []  # I have the chars in a list in specific index
    modlist = []  # Same for the modifiers
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    for i in range(0, 19, 4):
        modlist.append(modifiers[i * 3:(i + 1) * 3])

    modlist.append(b)
    lowerchar=[]
    upperchar=[]

    for i in range(97,123):
        lowerchar.append(chr(i))
    for i in range(65,91):
        upperchar.append(chr(i))
    print(len(charlist)+len(modlist)+len(punc))
    print(58+26)
    print(59+26+26)
    Total = charlist + modlist + punc + lowerchar + upperchar

    print(len(charlist)+len(modlist))

    stringlist = []
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    for i in range(num_words):
        string = ""
        NumberOfAlphabet = random.randint(1, 3)
        for j in range(NumberOfAlphabet):
            BanglaOrEnglish =random.randint(0,1)
            if(BanglaOrEnglish):
                WithModifier_or_WithNoModifier = random.randint(0, 1)
                if (WithModifier_or_WithNoModifier):

                    indexofchar = random.randint(0, 32)
                    string += (charlist[indexofchar])
                else:
                    indexofchar = random.choice(allowed_values)
                    indexofmodifier = random.randint(0, 5)
                    string += (charlist[indexofchar] + modlist[indexofmodifier])

            punrand = random.randint(0, 5)
            string += puncuation[punrand]

            NumberOfAlphabet = random.randint(1, 3)
            for j in range(NumberOfAlphabet):

                WithModifier_or_WithNoModifier = random.randint(0, 1)
                if (WithModifier_or_WithNoModifier):

                    indexofchar = random.randint(0, 32)
                    string += (charlist[indexofchar])
                else:
                    indexofchar = random.choice(allowed_values)
                    indexofmodifier = random.randint(0, 5)
                    string += (charlist[indexofchar] + modlist[indexofmodifier])
            else:
                upperorlower=random.randint(0,1)
                if(upperorlower):
                    string+=Total[random.randint(59,84)]
                else:
                    string+=Total[random.randint(85,109)]

        print(string)
        # print(len(string))
        stringlist.append(string)

    return stringlist
# CombinedDataset(1)
