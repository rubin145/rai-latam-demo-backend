#!/usr/bin/env python3
"""
Script para buscar los templates por defecto en langchain
"""
import os
from langchain.evaluation.criteria.eval_chain import PROMPT, PROMPT_WITH_REFERENCES
from langchain.evaluation.scoring.eval_chain import SCORING_TEMPLATE

def show_langchain_templates():
    print("🔍 TEMPLATES POR DEFECTO DE LANGCHAIN")
    print("=" * 60)
    
    print("🎯 CRITERIA EVALUATOR - PROMPT:")
    print("-" * 40)
    print(PROMPT.template)
    print()
    
    print("🎯 CRITERIA EVALUATOR - PROMPT_WITH_REFERENCES:")
    print("-" * 40)
    print(PROMPT_WITH_REFERENCES.template)
    print()
    
    print("🎯 SCORE STRING EVALUATOR - SCORING_TEMPLATE:")
    print("-" * 40) 
    # SCORING_TEMPLATE es un ChatPromptTemplate con 2 mensajes: System y Human
    # El template importante está en el HumanMessagePromptTemplate (mensaje 1)
    human_template = SCORING_TEMPLATE.messages[1].prompt.template
    print(human_template)
    print()
    
    print("✅ Templates extraídos!")

if __name__ == "__main__":
    show_langchain_templates()
