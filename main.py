#from kaggle import *
from tokenizer import *




if __name__ == "__main__":
    profanity = tipoArquivo("profanity_en.csv")
    print(profanity.pointer)
    

    #lista_profanidade = tokenizer(fil)