# -*- coding: utf-8 -*-
"""
Grammar and spelling analysis module.
Uses language_tool_python to find spelling, stylistic, and grammatical suggestions.
"""

from typing import List, Dict, Union
import language_tool_python
from analyzers.language import detect_language

# Global cache of LanguageTool instances to prevent slow restarts
_tools_cache: Dict[str, language_tool_python.LanguageTool] = {}

def get_tool_for_language(lang_code: str) -> language_tool_python.LanguageTool:
    """
    Returns a LanguageTool instance for the specified language.
    
    Args:
        lang_code (str): Language code ('pl', 'uk', 'en').
        
    Returns:
        language_tool_python.LanguageTool: LanguageTool instance.
    """
    lang_map = {
        'pl': 'pl-PL',
        'uk': 'uk-UA',
        'en': 'en-US'
    }
    code = lang_map.get(lang_code, 'en-US')
    
    if code not in _tools_cache:
        _tools_cache[code] = language_tool_python.LanguageTool(code)
    return _tools_cache[code]

def check_grammar(text: str) -> List[Dict[str, Union[int, str, List[str]]]]:
    """
    Checks the spelling and grammar of the input text.
    
    Args:
        text (str): The text to check.
        
    Returns:
        List[Dict[str, Union[int, str, List[str]]]]: List of dictionary suggestions with offset, bad text, and suggestions list.
    """
    if not text or len(text.strip()) < 3:
        return []
        
    lang_info = detect_language(text)
    lang_code = lang_info['code']
    
    try:
        tool = get_tool_for_language(lang_code)
        matches = tool.check(text)
    except Exception:
        # Fallback to English if tool initialization fails
        try:
            tool = get_tool_for_language('en')
            matches = tool.check(text)
        except Exception:
            return []
            
    results = []
    for match in matches:
        results.append({
            'offset': match.offset,
            'length': match.errorLength,
            'message': match.message,
            'ruleId': match.ruleId,
            'bad': text[match.offset : match.offset + match.errorLength],
            'suggestions': match.suggestions[:5]  # Limit to top 5 suggestions
        })
    return results
