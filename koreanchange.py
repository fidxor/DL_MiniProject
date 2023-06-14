import pandas as pd
from googletrans import Translator

def languageTrans(text):
    translator = Translator()

    translation = translator.translate(text, dest = 'ko')

    print(translation.text)



languageTrans('')