#from kaggle import *
from tokenizer import *




if __name__ == "__main__":
    profanity = tipoArquivo(
        nomeArquivo = "profanity_en.csv",
        separadores = [',', '\n', ' '],
        quantHeader = 9
    )
    print(profanity.header)
    

    #lista_profanidade = tokenizer(fil)