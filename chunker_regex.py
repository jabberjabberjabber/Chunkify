# code source: https://gist.github.com/hanxiao/3f60354cf6dc5ac698bc9154163b4e6a
# link: https://jina.ai/tokenizer/

import regex as re

MAX_HEADING_LENGTH = 7
MAX_HEADING_CONTENT_LENGTH = 200
MAX_HEADING_UNDERLINE_LENGTH = 200
MAX_HTML_HEADING_ATTRIBUTES_LENGTH = 100
MAX_LIST_ITEM_LENGTH = 200
MAX_NESTED_LIST_ITEMS = 6
MAX_LIST_INDENT_SPACES = 7
MAX_BLOCKQUOTE_LINE_LENGTH = 200
MAX_BLOCKQUOTE_LINES = 15
MAX_CODE_BLOCK_LENGTH = 1500
MAX_CODE_LANGUAGE_LENGTH = 20
MAX_INDENTED_CODE_LINES = 20
MAX_TABLE_CELL_LENGTH = 200
MAX_TABLE_ROWS = 20
MAX_HTML_TABLE_LENGTH = 2000
MIN_HORIZONTAL_RULE_LENGTH = 3
MAX_SENTENCE_LENGTH = 400
MAX_QUOTED_TEXT_LENGTH = 600
MAX_PARENTHETICAL_CONTENT_LENGTH = 400
MAX_NESTED_PARENTHESES = 5
MAX_MATH_INLINE_LENGTH = 100
MAX_MATH_BLOCK_LENGTH = 500
MAX_PARAGRAPH_LENGTH = 1000
MAX_STANDALONE_LINE_LENGTH = 800
MAX_HTML_TAG_ATTRIBUTES_LENGTH = 100
MAX_HTML_TAG_CONTENT_LENGTH = 1000
LOOKAHEAD_RANGE = 200; 

chunk_regex = re.compile(
    r"(" +
    # 1. Headings (Setext-style, Markdown, and HTML-style)
    rf"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}|\w[^\r\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\r?\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>)[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}(?:</h[1-6]>)?(?:\r?\n|$))" +
    "|" +
    # 2. Citations
    rf"(?:\[[0-9]+\][^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})" +
    "|" +
    # 3. List items (Adjusted to handle indentation correctly)
    rf"(?:(?:^|\r?\n)[ \t]{{0,3}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}})(?:\r?\n[ \t]{{2,}}(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}))*)" +
    "|" +
    # 4. Block quotes (Handles nested quotes without chunking)
    rf"(?:(?:^>(?:>|\\s{{2,}}){{0,2}}(?:[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})(?:\r?\n[ \t]+[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})*?\r?\n?))" +
    "|" +
    # 5. Code blocks
    rf"(?:(?:^|\r?\n)(?:```|~~~)(?:\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\r?\n[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:```|~~~)\r?\n?)" +
    rf"|(?:(?:^|\r?\n)(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\r?\n(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\r?\n?)" +
    rf"|(?:<pre>(?:<code>)[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:</code>)?</pre>)" +
    "|" +
    # 6. Tables
    rf"(?:(?:^|\r?\n)\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|(?:\r?\n\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\|)?(?:\r?\n\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|){{0,{MAX_TABLE_ROWS}}})" +
    rf"|<table>[\s\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?</table>" +
    "|" +
    # 7. Horizontal rules
    rf"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\s*$|<hr\s*/?>)" +
    "|" +
    # 8. Standalone lines or phrases (Prevent chunking by treating indented lines as part of the same block)
    rf"(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?:</[a-zA-Z]+>)?(?:\r?\n|$))" +
    rf"(?:\r?\n[ \t]+[^\r\n]*)*)" +
    "|" +
    # 9. Sentences (Allow sentences to include multiple lines if they are indented)
    rf"(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$)(?:\r?\n[ \t]+[^\r\n]*)*)" +
    "|" +
    # 10. Quoted text, parentheticals, or bracketed content
    rf"(?<!\w)\"\"\"[^\"]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\"\"\"(?!\w)" +
    rf"|(?<!\w)(?:['\"\`])[^\r\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\g<1>(?!\w)" +
    rf"|\([^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}(?:\([^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}\)[^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}){{0,{MAX_NESTED_PARENTHESES}}}\)" +
    rf"|\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\][^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\]" +
    rf"|\$[^\r\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\$" +
    rf"|`[^\r\n`]{{0,{MAX_MATH_INLINE_LENGTH}}}`" +
    "|" +
   
    # 11. Paragraphs (Treats indented lines as part of the same paragraph)
    rf"(?:(?:^|\r?\n\r?\n)(?:<p>)?(?:(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))(?:</p>)?(?:\r?\n[ \t]+[^\r\n]*)*)" +
    "|" +
    
    # 12. HTML-like tags and their content
    rf"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:>[\s\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}</[a-zA-Z]+>|\s*/>))" +
    "|" +
    # 13. LaTeX-style math expressions
    rf"(?:(?:\$\$[\s\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\$\$)|(?:\$[^\$\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\$))" +
    "|" +
    # 14. Fallback for any remaining content (Keep content together if it's indented)
    rf"(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))(?:\r?\n[ \t]+[^\r\n]*)?))" +
    r")",
    re.MULTILINE | re.UNICODE
)



