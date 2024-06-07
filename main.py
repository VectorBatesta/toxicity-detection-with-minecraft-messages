#from kaggle import *
from tokenizer import *




if __name__ == "__main__":
    profanity = profanityArquivo(
        nomeArquivo = "profanity_en.csv",
        separadores = [',', '\n'], #so separar tokens por virgula por causa do CSV
        quantHeader = 9, #quantidade de tokens no header
        toxicWords_amount = 1597 #1598 linhas em profanity tirando o header
    ) 
    print(profanity.header)

    print(profanity.toxicWordLIST[0])
    

    #lista_profanidade = tokenizer(fil)