input_file = 'profanity_en_old.csv'
output_file = 'profanity_en.csv'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        first_term = line.split(',')[0].strip()
        if len(first_term.split()) == 1:
            outfile.write(line)
