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
def Jointchars():
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character

    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    str = "ন্ত"
    hosh = str[3:6]
    char=""
    jointchars = []

    allowed_for_Po = [31, 38, 41, 39]  # Allowed Character List
    for ind in allowed_for_Po:
        char = charlist[ind] + hosh + "প"
        jointchars.append(char)


    allowed_for_Do = [12, 13, 16, 23, 25, 30, 38, 39]
    for ind in allowed_for_Do:
        char = charlist[ind] + hosh + "ড"
        jointchars.append(char)


    allowed_for_To = [11, 12, 13, 16, 21, 25, 26, 31, 33, 35, 38, 39, 40, 41]
    for ind in allowed_for_To:
        char = charlist[ind] + hosh + "ট"
        jointchars.append(char)

    allowed_for_cho = [16, 18, 20, 21, 24, 25, 31, 38, 39, ]
    for ind in allowed_for_cho:
        char = charlist[ind] + hosh + "চ"
        jointchars.append(char)

    allowed_for_go = [13, 28, 29, 30]
    for ind in allowed_for_go:
        char = charlist[ind] + hosh + "গ"
        jointchars.append(char)

    allowed_for_ko = [11, 15, 38, 40, 41]
    for ind in allowed_for_ko:
        char = charlist[ind] + hosh + "ক"
        jointchars.append(char)

    allowed_for_bo = [11, 12, 13, 14, 17, 18, 21, 23, 25, 26, 27, 28, 29, 30, 33, 35, 38, 39, 40, 41]
    for ind in allowed_for_bo:
        char = charlist[ind] + hosh + "ব"
        jointchars.append(char)

    allowed_for_to = [11, 26, 31, 35, 39]
    for ind in allowed_for_to:
        char = charlist[ind] + hosh + "ত"
        jointchars.append(char)

    allowed_for_do = [13, 28, 30, 33, 35, 38]
    for ind in allowed_for_do:
        char = charlist[ind] + hosh + "দ"
        jointchars.append(char)

    allowed_for_no = [11, 13, 14, 16, 26, 28, 29, 30, 31, 35, 39, 42]
    for ind in allowed_for_no:
        char = charlist[ind] + hosh + "ন"
        jointchars.append(char)
    allowed_for_ro = list(range(11, 46))
    for ind in allowed_for_ro:
        char = charlist[ind] + hosh + "র"
        jointchars.append(char)


    return jointchars
def getTotalData():
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
    C = 0
    for i in range(0, 19, 4):
        # print(i)
        modlist.append(modifiers[i:i + 3])

    lowerchar = []
    upperchar = []

    for i in range(97, 123):
        lowerchar.append(chr(i))
    for i in range(65, 91):
        upperchar.append(chr(i))

    # print(len(modlist))
    Joint = Jointchars()

    Total = charlist + modlist + Joint + punc + lowerchar + upperchar
    return Total
def Makestr():

    '''
        Ranges:
        Character = 0-45
        Modifier = 46-50
        Joint = 51 - 172
        Punctuation = 173 - 178
        LowerChar= 179-204
        UpperChar = 205-230
    '''
    Total=getTotalData()
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    String=""
    BanglaOrEnglishOrPunc=random.randint(0,1)
    # print(BanglaOrEnglishOrPunc)
    if(BanglaOrEnglishOrPunc==0):
        NormalOrPunc=random.randint(0,3) # Punctuation or Normal Character
        if(NormalOrPunc==3):
            PuncInd=random.randint(173,178)
            String=Total[PuncInd]
        else:
            NormalOrJoinOrModifier=random.randint(0,2)
            if(NormalOrJoinOrModifier==0):
                IndexofNormal=random.randint(0,45)
                String=Total[IndexofNormal]
            elif(NormalOrJoinOrModifier==1):
                IndexofNormal=random.choice(allowed_values)
                IndexofMod=random.randint(46,50)
                String=Total[IndexofNormal]+Total[IndexofMod]
            else:
                IndexofJoint=random.randint(51,172)
                String=Total[IndexofJoint]
    elif(BanglaOrEnglishOrPunc==1):
        NormalorPunc=random.randint(0,3)
        if(NormalorPunc==3):
            PuncInd = random.randint(173, 178)
            String = Total[PuncInd]
        else:
            UpperOrLower=random.randint(0,1)
            if(UpperOrLower):
                IndexUp=random.randint(179,204)
                String=Total[IndexUp]
            else:
                IndexDown=random.randint(205,230)
                String=Total[IndexDown]
    # else:
    #     PuncInd = random.randint(173, 178)
    #     String = Total[PuncInd]

    return String
def decodetheData(string):
    Index = {}
    C = 0
    print(string)
    Total=getTotalData()
    for i in Total:
        Index[i] = C
        C += 1
    i=0
    print(len(Total))
    labels=[]
    while i<len(string):
        isJoint=0
        isChar=0
        if(i+9<=len(string)):
            ifjointornot=string[i:i+9]
            Flag=0
            for x in Total:
                if(ifjointornot==x):
                    isJoint=1
                    i+=9
                    labels.append(Index[x])
                    break

        if(isJoint==0):
            if(i+3<=len(string)):
                ifcharornot=string[i:i+3]
                Flag=0
                for x in Total:
                    if(ifcharornot==x):
                        Flag=1
                        isChar=1
                        i+=3
                        labels.append(Index[x])
                        break
        if(isJoint==0 and isChar==0):
            for x in Total:
                if(x==string[i]):
                    i+=1
                    labels.append(Index[x])
                    break
    print(labels)
    return labels


def CombinedDataset(num_words):
    strings=[]
    for i in range(num_words):
        numberofchars=random.randint(3,12)
        string=""
        for i in range(numberofchars):
            string+=Makestr()
        strings.append(string)
    # decodetheData(string)
    return strings





