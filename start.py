import os
import nltk
nltk.download('punkt')

print('START')
print('PARSING')
os.system('python parse_files.py')
print('STATISTICS')
os.system('python gen_statistics.py')
print('SCORING')
os.system('python scorer.py')
print('END')