# MultipleChoiceParser

Simple scantron-like multiple choice image to text parser

## How it works

1. Print out scantron.jpg
2. Fill out scantron as desired
3. Scan or take picture of filled scantron
4. Run program `python mcparser.py -i <path-to-image>`
 

Result is sent to output.txt as json file

# Installation/Usage
```
git clone https://github.com/venkatabhishek/MultipleChoiceParser.git
pip install -r requirements.txt
python mcparser.py -i example/example.jpg
```
