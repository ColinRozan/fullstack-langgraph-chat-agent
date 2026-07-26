from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """You are given web search results about "{research_topic}". Synthesize them into a coherent, well-written summary.

Instructions:
- The current date is {current_date}.
- Only include information found in the provided search results. Do not make up any information.
- Consolidate key findings while tracking the sources.
- Include source citations using markdown format with the URLs provided in the search results.
- The output should be a well-written summary or report.

Research Topic:
{research_topic}

Search Results:
{search_results}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""


rag_reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries and knowledge base documents are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.
- Consider both the Web Research Summaries and the Knowledge Base Documents when evaluating sufficiency.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Web Research Summaries and Knowledge Base Documents to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Web Research Summaries:
{summaries}

Knowledge Base Documents:
{rag_documents}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries and knowledge base documents.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step.
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided web research summaries, knowledge base documents, and the user's question.

CRITICAL SOURCE DISTINCTION RULES:
1. For EVERY key fact, claim, or data point in your answer, you MUST append an inline source tag indicating where it came from.
2. Use these exact formats:
   - Web search: [🌐 Title](URL)
   - Knowledge base: [📄 source: filename.pdf]
3. If a fact appears in BOTH sources, cite BOTH like: [🌐 Title](URL) [📄 source: file.pdf]
4. If a paragraph or section is based PRIMARILY on knowledge base documents, prefix it with: **【基于知识库】**
5. If a paragraph or section is based PRIMARILY on web research, prefix it with: **【基于网络搜索】**
6. If NO knowledge base documents are relevant, explicitly state at the top: "⚠️ 未在知识库中找到相关文档，以下回答完全基于网络搜索。"
7. If web search returned no useful results, explicitly state: "⚠️ 网络搜索未返回有效结果，以下回答完全基于知识库。"

User Context:
- {research_topic}

Web Research Summaries:
{summaries}

Knowledge Base Documents:
{rag_documents}"""
