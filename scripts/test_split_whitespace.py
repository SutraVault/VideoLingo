"""Whitespace in ASR tokens must not increase subtitle splitting."""
import ast
from pathlib import Path
import unittest
import spacy

ROOT=Path(__file__).resolve().parents[1]

class WhitespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((ROOT/'core/_3_2_split_meaning.py').read_text(encoding='utf-8'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tokenize_sentence']
        cls.scope={};exec(compile(tree,'split_meaning','exec'),cls.scope)
        cls.nlp=spacy.blank('en')

    def test_spacing_does_not_change_word_limit(self):
        text='So there was a whole series of experimental M1 Garand rifles.'
        normal=self.scope['tokenize_sentence'](text,self.nlp)
        doubled=self.scope['tokenize_sentence'](text.replace(' ','  '),self.nlp)
        self.assertEqual(normal,doubled)
        self.assertLessEqual(len(doubled),20)

    def test_newlines_and_tabs_are_not_counted(self):
        normal=self.scope['tokenize_sentence']('Earl Harvey is an engineer.',self.nlp)
        spaced=self.scope['tokenize_sentence']('  Earl\tHarvey\n is   an engineer.  ',self.nlp)
        self.assertEqual(normal,spaced)

    def test_real_short_clause_is_not_forced_into_two_parts(self):
        text='and the Armory was always interested in potential ways to simplify it.'
        self.assertLessEqual(len(self.scope['tokenize_sentence'](text.replace(' ','  '),self.nlp)),20)

if __name__=='__main__':unittest.main()
