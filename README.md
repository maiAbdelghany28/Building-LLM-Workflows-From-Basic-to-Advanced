###########################################################
##           CSAI 422: Laboratory Assignment 5           ##
##     Building LLM Workflows: From Basic to Advanced    ##
###########################################################

## Overview

This repository implements four different workflows to repurpose a blog post into multiple formats: a summary, social media posts, and an email newsletter. Designed a modular system that breaks down complex tasks into smaller, more manageable components and then assemble them into effective workflows.The workflows use LLMs (via OpenAI-compatible API endpoints) and include:
   + Pipeline Workflow
   + DAG Workflow
   + Reflexion-based Workflow
   + Agent-driven Workflow

A comparative analysis is also included to evaluate their effectiveness based on output quality and execution time.

## Learning Objectives

• Implement tool definitions and function calling with the OpenAI API.

• Build a conversational agent capable of using external tools.

• Apply Chain of Thought and ReAct reasoning paradigms to improve agent performance.

• Integrate external APIs into your conversational agent.

## Prerequisites

 • Basic understanding of APIs and async programming

 • Python 3.10+

 • Understanding of APIs

 • Access to LLM API keys and models (OpenAI, Groq, etc.)

 • Familiarity with tool usage in LLMs 


## Setup Instructions

1. Clone the repository:
   git clone <repo-url>
   cd <repo-directory>

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare your environment variables: Create a .env file with the following (set according to your LLM provider):
   GROQ_API_KEY=your_groq_api_key
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   GROQ_MODEL=llama3-8b-8192
   MODEL_SERVER=GROQ  # or OPENAI, OPTOGPT

4. Add your sample blog post:
Create a file named sample_blog_post.json with this format in the project directory:
   {
   "title": "The Power of LLMs in Content Creation",
   "content": "Large Language Models (LLMs) are transforming how content is generated..."
   }

5. Run the program:
   python main.py


## Implementation Overview 

The system modularizes tasks using function-calling and OpenAI-compatible schemas:

<< Core Tasks >>

   • Extract Key Points – Identifies key ideas from blog post

   • Generate Summary – Synthesizes concise summary from key points

   • Create Social Media Posts – Generates tailored posts for Twitter/X, LinkedIn, and Facebook

   • Create Email Newsletter – Drafts email subject and body text

<< Workflow Types >>

   1) Pipeline Workflow: A straightforward linear sequence
   2) DAG Workflow: A structured graph (currently mirrors pipeline)
   3) Reflexion Workflow: Uses self-evaluation and improvement
   4) Agent-Driven Workflow: Uses LLM agent to reason and plan tool use autonomously

## Example Output (Pipeline Workflow)

   {
   "key_points": [
      "LLMs automate content generation",
      "They increase productivity across sectors",
      "Effective for summarization, social media, and newsletters"
   ],
   "summary": "LLMs are revolutionizing content creation by automating tasks, enhancing productivity, and streamlining communication across platforms.",
   "social_posts": {
      "twitter": "Discover how LLMs are revolutionizing content creation. #AI #LLM",
      "linkedin": "Learn how LLMs boost productivity and automate tasks in content creation workflows.",
      "facebook": "LLMs are changing the way we write and share content. Discover their full potential!"
   },
   "email": {
      "subject": "Revolutionize Your Content with LLMs",
      "body": "Large Language Models (LLMs) are transforming content workflows. From summaries to social media, see how automation boosts your efficiency."
   }
   }

Outputs from Reflexion and Agent workflows may differ due to evaluation and adaptive planning.


## Workflow Effectiveness Analysis

After running all workflows, the system evaluates outputs using LLM-based criteria for clarity, engagement, relevance, and platform-fit. Metrics are aggregated per workflow. 

< Best Workflow Per Task >

   • Summary → Reflexion

   • Social Media → Agent

   • Email Newsletter → Reflexion
< Recommendations >

   • For fast prototyping → Pipeline

   • For production-ready quality → Reflexion

   • For dynamic, complex content → Agent


## Challenges and Solutions

1| Tool Call Failures
   Problem: Occasional parsing errors when extracting function arguments.
   Fix: Added schema validation and robust fallback handling.

2| Reflexion Convergence
   Problem: Some outputs didn’t improve beyond a quality threshold.
   Fix: Introduced max_attempts and loop break conditions.

3| Agent Tool Looping
   Problem: The agent sometimes didn’t complete with finish tool.
   Fix: Added hard limit on iterations and a guiding message to finalize.

4| Variable LLM Behavior
   Problem: Different providers returned different levels of structure.
   Fix: Normalized message formatting and fallback handling across providers.


## Author
Mai Waheed AbdelMaksoud


## References
- Chapter 9: LLM Workflows from the textbook
- OpenAIAPIDocumentation: https://platform.openai.com/docs/guides/functioncalling
- ReAct Paper: “ReAct: Synergizing Reasoning and Acting in Language Models”
- Reflexion Paper: “Reflexion: Language Agents with Verbal Reinforcement Learning”